import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.io
import os
import argparse
import sys
import json
from datetime import datetime

from Models.var_B import GCN_DE_Model
from Models.var_C import DGCNN_Model
from Models.graph_construction import get_knn_adjacency_matrix
from utils.training_utils import train_model_with_interrupt
from utils.feature_engineering import SmartPreprocessor, get_standard_channel_names
from sklearn.preprocessing import StandardScaler, RobustScaler

# --- Configuration ---
LOCS_FILE = "utils/channel_62_pos.locs"
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-3 # Stronger regularization
PATIENCE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_FILE = "run_config.json"

def get_next_run_id(window_size):
    """Reads and increments the run counter from run_config.json for the specific window size"""
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump({}, f)
            
    with open(CONFIG_FILE, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    
    key = f"run_counter_{window_size}"
    next_id = config.get(key, 0) + 1
    
    config[key] = next_id
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
        
    return next_id

def get_args():
    parser = argparse.ArgumentParser(description="Train GCN-DE for EEG Emotion Recognition")
    parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'],
                        help="Training mode: 'sub_dep' (Session split) or 'sub_indep' (LOSO)")
    parser.add_argument('--split_strategy', type=str, default='session_holdout', 
                        choices=['session_holdout', 'random'],
                        help="Data splitting strategy (only for sub_dep mode)")
    parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'],
                        help="Feature window size: '1s' or '4s'")
    parser.add_argument('--model_type', type=str, default = 'DGCNN', choices=['GCN', 'DGCNN'],
                        help="Type of GCN model to use")
    parser.add_argument('--test_subject', type=int, default=1, 
                        help="ID of the subject to leave out for testing (only for sub_indep mode)")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    return parser.parse_args()

def load_de_data(data_folder, label_file):
    print(f"Loading DE features from {data_folder}...")
    try:
        label_mat = scipy.io.loadmat(label_file)
        trial_labels = label_mat['label'][0]
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file}")
        sys.exit(1)
    
    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = [label_map[l] for l in trial_labels]
    
    X_list = []
    y_list = []
    session_list = []
    subject_list = []
    trial_list = []
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    for f in files:
        parts = f.split('_')
        try:
            subj_id = int(parts[0])
        except ValueError: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)

    # --- Initialize Smart Preprocessor ---
    channel_names = get_standard_channel_names()
    preprocessor = SmartPreprocessor(channel_names)
    print("Initialized Smart Preprocessor for Bad Channel Correction.")
    # -------------------------------------

    band_weights = None
    print("Warning: Manual band weights DISABLED. Using raw data for Learnable Attention.")
        
    for subj_id in sorted(subject_files.keys()):
        # if band_weights and str(subj_id) in band_weights:
        #     weights = np.array(band_weights[str(subj_id)]) # (5,)
        # else:
        # weights = np.ones(5) # No weighting - REMOVED

        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            for trial_i in range(1, 16):
                key = f"de_LDS{trial_i}"
                if key not in mat: continue
                data = mat[key]
                data = np.transpose(data, (1, 0, 2))

                                # --- SMART PREPROCESSING (Interpolation) ---
                # 1. Prepare for cleaning: (62, samples, 5)
                data_for_cleaning = np.transpose(data, (1, 0, 2)) 
                
                # 2. Detect Bads (using mean across bands to get a single time-series proxy)
                # We use the average of the 5 bands as the "signal" to check for variance/flatline
                avg_signal = np.mean(data_for_cleaning, axis=2) # (62, samples)
                bads = preprocessor.detect_bad_channels(avg_signal)
                
                if bads:
                    # Interpolate each band separately (spatial interpolation works on 2D maps)
                    cleaned_bands = []
                    for b in range(5):
                        band_data = data_for_cleaning[:, :, b] # (62, samples)
                        cleaned_band = preprocessor.interpolate_bads(band_data, bads)
                        cleaned_bands.append(cleaned_band)
                    
                    # Stack back: (62, samples, 5)
                    data_for_cleaning = np.stack(cleaned_bands, axis=2)
                    
                    # Transpose back to (samples, 62, 5) for storage
                    data = np.transpose(data_for_cleaning, (1, 0, 2))
                # -------------------------------------------

                num_samples = data.shape[0]
                X_list.append(data)
                y_list.append(np.full(num_samples, mapped_labels[trial_i - 1]))
                session_list.append(np.full(num_samples, session_id))
                subject_list.append(np.full(num_samples, subj_id))
                unique_trial_id = subj_id * 1000 + session_id * 100 + trial_i
                trial_list.append(np.full(num_samples, unique_trial_id))

    if not X_list:
        print("Error: No data loaded.")
        sys.exit(1)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    sessions = np.concatenate(session_list, axis=0)
    subjects = np.concatenate(subject_list, axis=0)
    trials = np.concatenate(trial_list, axis=0)
    
    # --- Subject-Specific & Session-Specific Normalization (Vectorized) ---
    print("Applying Subject-Specific & Session-Specific Normalization...")
    
    # Create unique identifiers for each subject-session pair
    # Assuming max session ID < 1000, which is safe here
    group_ids = subjects * 1000 + sessions
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    
    X_scaled = np.zeros_like(X)
    
    for gid in unique_groups:
        mask = (group_ids == gid)
        X_group = X[mask] # (N, 62, 5)
        
        # Flatten to (N, Features) for scaling
        # We scale each channel-band combination independently
        N, C, B = X_group.shape
        X_flat = X_group.reshape(N, -1)
        
        scaler = RobustScaler()
        X_flat_scaled = scaler.fit_transform(X_flat)
        
        X_scaled[mask] = X_flat_scaled.reshape(N, C, B)
        
    X = X_scaled
    print(f"Total Samples: {X.shape[0]}")
    return X, y, sessions, subjects, trials

# ...existing code...
def evaluate(model, loader, base_edge_index, criterion, device, return_preds=False, return_embeddings=False):
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
            
            batch_X_flat = batch_X.view(-1, 5)

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

def main():
    args = get_args()
    print(f"Using device: {DEVICE}")
    print(f"Mode: {args.mode} | Window: {args.window_size}")

    if args.window_size == '1s':
        data_folder = "Data/ExtractedFeatures_1s"
    else:
        data_folder = "Data/ExtractedFeatures_4s"
    label_file = os.path.join(data_folder, "label.mat")
    
    # --- NEW FOLDER LOGIC ---
    run_id = get_next_run_id(args.window_size)
    
    model_name = f"{args.model_type}_DE_{args.window_size}"

    if args.mode == 'sub_dep':
        run_name = f"Attempt_{run_id}_{args.split_strategy}"
    else:
        # For LOSO, include the test subject in the run name
        run_name = f"Attempt_{run_id}_LOSO_sub{args.test_subject}"
    
    results_dir = os.path.join("Results", model_name, run_name)
    params_dir = os.path.join("Params", model_name, run_name)
    errors_dir = os.path.join("Errors", model_name, run_name)

    # Ensure these directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    
    print(f"Directories:")
    print(f"  Results: {results_dir}")
    print(f"  Params:  {params_dir}")
    print(f"  Errors:  {errors_dir}")

    X, y, sessions, subjects, trials = load_de_data(data_folder, label_file)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    if args.mode == 'sub_dep':
        if args.split_strategy == 'session_holdout':
            print("  -> Strategy: Session Holdout (Train on S1+S2, Test on S3)")
            train_mask = (sessions == 1) | (sessions == 2)
            test_mask = (sessions == 3)
            
            X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
            X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
            
        elif args.split_strategy == 'random':
            print("  -> Strategy: Random Split (80% Train, 20% Test, shuffled across sessions)")
            # Random shuffle of all data
            # We want to select 3 trials (1 per class) from EACH session of EACH subject for testing.
            # This ensures we mix sessions but DO NOT mix windows from the same trial.
            
            test_indices_list = []
            
            # Get unique subject-session pairs
            unique_sub_sess = np.unique(np.stack((subjects, sessions), axis=1), axis=0)
            
            for sub, sess in unique_sub_sess:
                # Find all trials for this subject & session
                mask = (subjects == sub) & (sessions == sess)
                # Get unique trials in this session
                sess_trials = np.unique(trials[mask])
                
                # Group trials by label
                trials_by_label = {0: [], 1: [], 2: []}
                for t_id in sess_trials:
                    # Get label for this trial (take first sample's label)
                    t_label = y[trials == t_id][0]
                    trials_by_label[t_label].append(t_id)
                
                # Select 1 trial from each label for TEST
                for label in [0, 1, 2]:
                    available_trials = trials_by_label[label]
                    if len(available_trials) > 0:
                        # Deterministic selection for reproducibility (or random)
                        # Let's pick the LAST trial of each class for testing to be consistent
                        test_trial = available_trials[-1] 
                        
                        # Find indices of this trial
                        t_indices = np.where(trials == test_trial)[0]
                        test_indices_list.append(t_indices)
            
            test_indices = np.concatenate(test_indices_list)
            
            # Create boolean mask
            test_mask = np.zeros(len(y), dtype=bool)
            test_mask[test_indices] = True
            train_mask = ~test_mask
            
            X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
            X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
            
            print(f"     Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    elif args.mode == 'sub_indep':
        print(f"  -> Strategy: Leave-One-Subject-Out (Test Subject: {args.test_subject})")
        
        # 1. Identify Test Subject
        test_mask = (subjects == args.test_subject)
        
        # 2. Identify Validation Subject (Pick one from the remaining subjects)
        # We pick the subject with ID = (test_subject % 15) + 1
        # e.g., if Test=1, Val=2. If Test=15, Val=1.
        val_subject_id = (args.test_subject % 15) + 1
        val_mask = (subjects == val_subject_id)
        
        # 3. Identify Train Subjects (Everyone else)
        train_mask = ~(test_mask | val_mask)
        
        print(f"     Train Subjects: All except {args.test_subject} and {val_subject_id}")
        print(f"     Validation Subject: {val_subject_id}")
        print(f"     Test Subject: {args.test_subject}")
        
        X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
        X_val, y_val = X_tensor[val_mask], y_tensor[val_mask]
        X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
        
        # Create Loaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        # We use Val loader for Early Stopping
        test_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
        # We keep the real Test loader for final evaluation
        final_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    
    print("Constructing Graph...")
    base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)
    
    if args.model_type == 'GCN':
        print("Initializing Static GCN Model...")
        # SIMPLIFIED MODEL: 2 Layers, 64 Hidden, 0.5 Dropout
        model = GCN_DE_Model(num_nodes=62, in_features=5, hidden_dim=64, 
                             num_classes=3, dropout_rate=0.5, num_layers=3).to(DEVICE)
    elif args.model_type == 'DGCNN':
        print("Initializing Dynamic DGCNN Model (Learnable Graph)...")
        model = DGCNN_Model(num_nodes=62, in_features=5, hidden_dim=64, 
                            num_classes=3, dropout_rate=0.5).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # --- CHANGE 2: Add Class Weights to fix Negative Recall ---
    # The model ignores Negative (Class 0). We give it a higher weight.
    # Class 0 (Neg), Class 1 (Neu), Class 2 (Pos)
    # We double the penalty for getting Negative wrong.
    class_weights = torch.tensor([1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Note: In sub_indep mode, 'test_loader' passed here is actually the Validation Loader
    train_model_with_interrupt(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        device=DEVICE,
        patience=PATIENCE,
        results_dir=results_dir,
        params_dir=params_dir,
        errors_dir=errors_dir,
        base_edge_index=base_edge_index,
        evaluate_fn=evaluate)

    # --- Final Test for LOSO ---
    if args.mode == 'sub_indep':
        print("\n>>> Running Final Test on Held-out Subject...")
        # Load best model
        best_model_path = os.path.join(params_dir, "best_model_checkpoint.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model from validation phase.")
        
        test_loss, test_acc, preds, true_labels = evaluate(model, final_test_loader, base_edge_index, criterion, DEVICE, return_preds=True)
        print(f"FINAL TEST RESULTS (Subject {args.test_subject}):")
        print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")
        
        # Save specific test results
        np.save(os.path.join(results_dir, f"final_test_preds_sub{args.test_subject}.npy"), {'preds': preds, 'true': true_labels})

if __name__ == "__main__":
    main()