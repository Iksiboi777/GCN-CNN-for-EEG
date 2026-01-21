import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from torch_geometric.loader import TensorDataset, DataLoader # REMOVED: Incorrect import path
from torch_geometric.data import Data
import numpy as np
# ...existing code...
import numpy as np
import os
import argparse
from scipy.spatial.distance import pdist, squareform
import json
import scipy.io
import sys

# Import your custom utilities and the model we defined
from utils.inductive_graph import get_base_edge_index
from Models.var_ind_graph import GraphSAGE_EEG_Model
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.feature_engineering import get_standard_channel_names
# from Models.graph_construction import get_knn_adjacency_matrix

# --- Configuration ---
LOCS_FILE = "utils/channel_62_pos.locs"
BATCH_SIZE = 128
EPOCHS = 120
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3 
PATIENCE = 30
# --- NEW: Sparsity Penalty for Adaptive Layer ---
L1_LAMBDA = 1e-4  # Force gamma parameters towards zero
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_FILE = "run_config.json"

# --- STRATEGY CONFIGURATION ---
# HARD_SUBJECTS = [2, 7, 12, 13] # Subjects with systemic artifacts/sinkholes
ROLLING_VAR_WINDOW = 9         # Window size for generating variance features

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



def compute_rolling_variance(data, window_size=3):
    """
    Computes rolling variance along the time axis (axis 1).
    Input: (62, samples, 5)
    Output: (62, samples, 5)
    """
    pad_width = window_size // 2
    padded = np.pad(data, ((0,0), (pad_width, pad_width), (0,0)), mode='edge')
    
    vars_list = []
    for i in range(data.shape[1]):
        slice_data = padded[:, i : i + window_size, :]
        vars_list.append(np.var(slice_data, axis=1))
        
    return np.stack(vars_list, axis=1)

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
    # preprocessor = SmartPreprocessor(channel_names) # DISABLED for Attempt 18 reproduction
    # print("Initialized Smart Preprocessor for Bad Channel Correction.")
    # -------------------------------------

    band_weights = None
    print("Warning: Manual band weights DISABLED. Using raw data.")
        
    for subj_id in sorted(subject_files.keys()):
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
                
                # --- ROBUST SHAPE CORRECTION ---
                # Target: (62, samples, 5)
                # We identify dimensions by size: 62=Channels, 5=Bands
                shape = data.shape
                if shape[0] == 62:
                    if shape[2] == 5:
                        pass # Already (62, samples, 5)
                    elif shape[1] == 5:
                        data = np.transpose(data, (0, 2, 1)) # (62, 5, samples) -> (62, samples, 5)
                elif shape[1] == 62:
                    if shape[2] == 5:
                        data = np.transpose(data, (1, 0, 2)) # (samples, 62, 5) -> (62, samples, 5)
                    elif shape[0] == 5:
                        data = np.transpose(data, (1, 2, 0)) # (5, 62, samples) -> (62, samples, 5)
                elif shape[2] == 62:
                    if shape[1] == 5:
                        data = np.transpose(data, (2, 0, 1)) # (samples, 5, 62) -> (62, samples, 5)
                    elif shape[0] == 5:
                        data = np.transpose(data, (2, 1, 0)) # (5, samples, 62) -> (62, samples, 5)
                # -------------------------------

                # --- ATTEMPT 18 LOGIC: MANUAL CF2 FIX ONLY ---
                # We blindly apply the fix to CF2 because we know it is often broken.
                # if 'CF2' in channel_names:
                #     cf2_idx = channel_names.index('CF2')
                #     # Check if neighbors exist in the dataset
                #     n1, n2, n3 = 'FC2', 'C2', 'CP2'
                #     if n1 in channel_names and n2 in channel_names and n3 in channel_names:
                #         idx1 = channel_names.index(n1)
                #         idx2 = channel_names.index(n2)
                #         idx3 = channel_names.index(n3)
                        
                #         # Apply Triangulation Average
                #         data[cf2_idx, :, :] = (data[idx1, :, :] + data[idx2, :, :] + data[idx3, :, :]) / 3
                # -----------------------------------------------------------

                # --- RESTORED: Variance Calculation ---
                # Compute rolling variance (Strategy V2)
                data_var = compute_rolling_variance(data, window_size=ROLLING_VAR_WINDOW)
                
                # Stack: (62, samples, 5) + (62, samples, 5) -> (62, samples, 10)
                data_final = np.concatenate([data, data_var], axis=2)

                # Transpose to (samples, 62, 10) for storage/model input
                data_final = np.transpose(data_final, (1, 0, 2))

                # 5 ORIGINAL MEAN FEATURES ONLY (FOR ATTEMPT 18 REPRODUCTION)
                # data_final = np.transpose(data, (1, 0, 2))

                num_samples = data_final.shape[0]
                X_list.append(data_final)
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
    
    # --- ATTEMPT 18 REPRODUCTION: MANUAL Z-SCORE NORMALIZATION ---
    print("Applying Manual Subject-Specific & Session-Specific Z-Score Normalization...")
    
    group_ids = subjects * 1000 + sessions
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    
    n_groups = len(unique_groups)
    
    # Initialize arrays for stats
    # X shape: (N, 62, 5) -> Stats shape: (n_groups, 62, 5)
    group_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    group_sq_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    
    # Compute sums and counts using np.add.at (unbuffered in-place add)
    np.add.at(group_sums, group_indices, X)
    
    # Counts per group
    group_counts = np.bincount(group_indices)
    
    # Compute means: (n_groups, 62, 5)
    # Reshape counts to (n_groups, 1, 1) for broadcasting
    group_means = group_sums / group_counts[:, None, None]
    
    # Broadcast means back to original sample shape
    # shape: (N, 62, 5)
    expanded_means = group_means[group_indices]
    X_centered = X - expanded_means
    
    # Compute Stds
    np.add.at(group_sq_sums, group_indices, X_centered ** 2)
    group_stds = np.sqrt(group_sq_sums / group_counts[:, None, None])
    group_stds[group_stds < 1e-6] = 1.0 # Prevent div by zero
    
    # Apply Normalization
    expanded_stds = group_stds[group_indices]
    X = X_centered / expanded_stds
    print(f"Total Samples: {X.shape[0]}")
    return X, y, sessions, subjects, trials


def main():
    parser = argparse.ArgumentParser(description="Train GCN-DE for EEG Emotion Recognition")
    parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'])
    parser.add_argument('--aggregator', type=str, default='max', choices=['mean', 'max', 'lstm'])
    parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'])
    parser.add_argument('--test_subject', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Mode: {args.mode} | Window: {args.window_size}")

    if args.window_size == '1s':
        data_folder = "Data/ExtractedFeatures_1s"
    else:
        data_folder = "Data/ExtractedFeatures_4s"
    label_file = os.path.join(data_folder, "label.mat")
    
    run_id = get_next_run_id(args.window_size)
    
    model_name = f"GraphSAGE_DE_{args.window_size}"

    if args.mode == 'sub_dep':
        run_name = f"Attempt_{run_id}_Phase2"
    else:
        run_name = f"Attempt_{run_id}_LOSO_Phase2"
    
    results_dir = os.path.join("Results", model_name, run_name)
    params_dir = os.path.join("Params", model_name, run_name)
    errors_dir = os.path.join("Errors", model_name, run_name)

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
        train_mask = (sessions == 1) | (sessions == 2)
        test_mask = (sessions == 3)
        
        X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
        X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
            
        print(f"     Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    elif args.mode == 'sub_indep':
        print(f"  -> Strategy: Leave-One-Subject-Out (Test Subject: {args.test_subject})")
        test_mask = (subjects == args.test_subject)
        # val_subject_id = (args.test_subject % 15) + 1
        # val_mask = (subjects == val_subject_id)
        train_mask = ~(test_mask)
        
        print(f"     Train Subjects: All except {args.test_subject}")
        # print(f"     Validation Subject: {val_subject_id}")
        print(f"     Test Subject: {args.test_subject}")
        
        X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
        # X_val, y_val = X_tensor[val_mask], y_tensor[val_mask]
        X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
        print(f"     Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    
    # 3. Create the Inductive Template
    base_edge_index, coords = get_base_edge_index(LOCS_FILE, k=5)
    base_edge_index = base_edge_index.to(DEVICE)

    # 4. Standard Loaders (Compatible with training_utils.py)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # 4. Initialize Model
    in_features = X_train.shape[-1] # Should be 10 (DE + Var)
    print("Input feature dimension: ", in_features)
    model = GraphSAGE_EEG_Model(
        in_features=in_features, 
        hidden_dim=128, 
        aggregator=args.aggregator
    ).to(DEVICE)

    gamma_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'static_norm.gamma' in name:
            gamma_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.Adam([
        {'params': other_params, 'weight_decay': WEIGHT_DECAY}, # Normal L2
        {'params': gamma_params, 'weight_decay': 1e-2}          # Strong L2 (force small weights)
    ], lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=PATIENCE)
    
    # --- STRATEGY: Class Weights ---
    # Double penalty for Negative (Class 0) to fix Recall
    class_weights = torch.tensor([1.2, 0.9, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 5. Execute Training using existing training_utils.py
    print(f"Starting GraphSAGE Training with {args.aggregator} aggregator...")
    train_model_with_interrupt(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        device=DEVICE,
        results_dir=results_dir,
        params_dir=params_dir,
        errors_dir=errors_dir,
        base_edge_index=base_edge_index,
        evaluate_fn=evaluate,
        in_features=in_features
    )

if __name__ == "__main__":
    main()