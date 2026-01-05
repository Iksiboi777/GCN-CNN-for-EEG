import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import scipy.io
from sklearn.preprocessing import RobustScaler

# Import existing model and utils
from Models.var_B import GCN_DE_Model
from Models.graph_construction import get_knn_adjacency_matrix
from utils.feature_engineering import SmartPreprocessor, get_standard_channel_names
from utils.training_utils import train_model_with_interrupt

# --- Configuration ---
LOCS_FILE = "utils/channel_62_pos.locs"
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-3
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def compute_rolling_variance(data, window_size=3):
    """Computes rolling variance (same as train_de.py)"""
    # Input: (62, samples, 5)
    pad_width = window_size // 2
    padded = np.pad(data, ((0,0), (pad_width, pad_width), (0,0)), mode='edge')
    vars_list = []
    for i in range(data.shape[1]):
        window = padded[:, i:i+window_size, :]
        vars_list.append(np.var(window, axis=1))
    return np.stack(vars_list, axis=1)

def evaluate(model, loader, base_edge_index, criterion, device, return_preds=False, return_embeddings=False):
    """
    Evaluates the model on a dataset. 
    Replicated from train_de.py to ensure compatibility with train_model_with_interrupt.
    """
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    all_preds = []
    all_labels = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            curr_batch_size = batch_X.size(0)
            batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(62)
            
            offsets = (torch.arange(curr_batch_size, device=device) * 62).view(-1, 1, 1)
            edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
            # Flatten features: (Batch, 62, 10) -> (Batch*62, 10)
            batch_X_flat = batch_X.view(-1, 10)

            if return_embeddings:
                outputs, embeddings = model(batch_X_flat, edge_index, batch_idx, return_embedding=True)
                all_embeddings.extend(embeddings.cpu().numpy())
            else:
                outputs = model(batch_X_flat, edge_index, batch_idx)
                
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
            
    acc = 100 * correct / total
    avg_loss = val_loss / len(loader)
    
    if return_embeddings:
        return avg_loss, acc, all_preds, all_labels, all_embeddings
    if return_preds:
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc

# --- 2. Data Loading (Modified for Binary Diagnostic) ---    

def load_binary_data(data_folder, label_file, ablate_gamma=False):
    print(f"Loading data from {data_folder}...")
    try:
        label_mat = scipy.io.loadmat(label_file)
    except FileNotFoundError:
        print("Label file not found.")
        return None, None, None, None

    # Load Labels (15 trials total)
    trial_labels = label_mat['label'][0]
    # Map: -1 (Neg) -> 0, 0 (Neu) -> 1, 1 (Pos) -> 2
    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = np.array([label_map[l] for l in trial_labels])

    # --- FILTER: Keep only Negative (0) and Neutral (1) ---
    keep_mask = (mapped_labels != 2) # Drop Positive
    
    print(f"Labels loaded. Keeping {np.sum(keep_mask)} out of {len(keep_mask)} trials per session.")

    files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
    
    subject_files = {}
    for f in files:
        parts = f.split('_')
        try: subj_id = int(parts[0])
        except: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)

    # --- CORRECT INITIALIZATION ---
    channel_names = get_standard_channel_names()
    preprocessor = SmartPreprocessor(channel_names)

    # Lists to store data
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for subj_id in sorted(subject_files.keys()):
        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1 # 1, 2, or 3
            
            # --- SPLIT STRATEGY: Session Holdout ---
            # Train on Sessions 1 & 2, Test on Session 3
            is_test_session = (session_id == 3)

            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            
            for trial_i in range(1, 16):
                # Calculate index for this specific trial (0-14)
                trial_idx = trial_i - 1
                
                # Check if this trial is in our "keep list" (Neg or Neu only)
                if not keep_mask[trial_idx]:
                    continue 

                key = f"de_LDS{trial_i}"
                if key not in mat: 
                    continue
                
                # Data Shape: (62, samples, 5)
                data = mat[key] 
                
                # --- CORRECT PREPROCESSING IMPLEMENTATION ---
                # 1. Detect Bads using average across bands
                avg_signal = np.mean(data, axis=2) # (62, samples)
                bads = preprocessor.detect_bad_channels(avg_signal)
                
                if bads:
                    # 2. Interpolate each band individually
                    cleaned_bands = []
                    for b in range(5):
                        band_data = data[:, :, b] # (62, samples)
                        cleaned_band = preprocessor.interpolate_bads(band_data, bads)
                        cleaned_bands.append(cleaned_band)
                    data = np.stack(cleaned_bands, axis=2) # Reassemble to (62, samples, 5)
                # --------------------------------------------

                # 3. Compute Variance (on cleaned data)
                data_var = compute_rolling_variance(data, window_size=3)
                
                # 4. Concatenate -> (62, samples, 10)
                data_combined = np.concatenate([data, data_var], axis=2)

                # 5. Transpose for Model -> (samples, 62, 10)
                data_combined = np.transpose(data_combined, (1, 0, 2))

                # --- ABLATION: KILL GAMMA MEAN ---
                if ablate_gamma:
                    # Index 4 is DE_Gamma. Set it to 0.
                    data_combined[:, :, 4] = 0.0
                # ---------------------------------

                num_samples = data_combined.shape[0] # Corrected axis after transpose
                current_label = mapped_labels[trial_idx]
                
                # Append to appropriate list based on session
                if is_test_session:
                    X_test_list.append(data_combined)
                    y_test_list.append(np.full(num_samples, current_label))
                else:
                    X_train_list.append(data_combined)
                    y_train_list.append(np.full(num_samples, current_label))

    # Concatenate everything
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Robust Scaling
    # Note: We must fit scaler ONLY on Train data to avoid leakage
    print("Applying RobustScaler (Fit on Train, Transform Both)...")
    scaler = RobustScaler()
    
    # Reshape Train
    N_tr, V, F = X_train.shape
    X_train_reshaped = X_train.reshape(-1, F)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(N_tr, V, F)
    
    # Reshape Test
    N_te, V, F = X_test.shape
    X_test_reshaped = X_test.reshape(-1, F)
    X_test_scaled = scaler.transform(X_test_reshaped) # Transform only!
    X_test = X_test_scaled.reshape(N_te, V, F)

    return X_train, y_train, X_test, y_test

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=str, default='1s')
    parser.add_argument('--ablate_gamma', action='store_true', default=False, 
                        help="If true, sets Gamma Mean feature to 0 to force model to use others.")
    args = parser.parse_args()

    data_folder = f"Data/ExtractedFeatures_{args.window_size}"
    label_file = os.path.join(data_folder, "label.mat")

    print(f"--- BINARY DIAGNOSTIC MODE (Neg vs Neu) ---")
    print(f"Split Strategy: Session Holdout (Train: Sess 1+2, Test: Sess 3)")
    print(f"Ablate Gamma Mean: {args.ablate_gamma}")

    # Load data already split
    X_train, y_train, X_test, y_test = load_binary_data(data_folder, label_file, args.ablate_gamma)
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape:  {X_test.shape}")

    # Convert to Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Setup Directories for TrainingManager ---
    run_name = f"BinaryDiag_Gamma{args.ablate_gamma}_SessionHoldout"
    results_dir = os.path.join("Results", "Diagnostic", run_name)
    params_dir = os.path.join("Params", "Diagnostic", run_name)
    errors_dir = os.path.join("Errors", "Diagnostic", run_name)

    # --- Construct Graph ---
    print("Constructing Graph...")
    base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)

    # --- Model Initialization ---
    IN_FEATURES = 10 # 5 Mean + 5 Variance
    print("Initializing GCN Model (Binary)...")
    model = GCN_DE_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=64, 
                         num_classes=2, dropout_rate=0.5, num_layers=3).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    # --- Master Training Loop ---
    train_model_with_interrupt(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        patience=PATIENCE,
        results_dir=results_dir,
        params_dir=params_dir,
        errors_dir=errors_dir,
        base_edge_index=base_edge_index,
        evaluate_fn=evaluate,
        hyperparams=args,
        in_features=IN_FEATURES
    )

if __name__ == "__main__":      
    main()
