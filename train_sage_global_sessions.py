import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import scipy.io
import argparse
import sys
import json
import time

# --- CUSTOM PROJECT IMPORTS ---
from Models.var_ind_graph import GraphSAGE_EEG_Model
from utils.inductive_graph import get_base_edge_index
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.feature_engineering import get_standard_channel_names

# --- CONFIGURATION ---
CACHE_FOLDER = "Data/Cache_GraphSAGE_1s"
LOCS_FILE = "utils/channel_62_pos.locs"
CONFIG_FILE = "run_config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Hyperparameters
BATCH_SIZE = 512  # Larger batch size for the larger pooled dataset
EPOCHS = 120
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-2 
PATIENCE = 30

# -----------------------------------------------------------------------------
# 1. RUN ID MANAGEMENT
# -----------------------------------------------------------------------------
def get_next_run_id(window_size):
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump({}, f)
    with open(CONFIG_FILE, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    key = f"global_run_counter_{window_size}"
    next_id = config.get(key, 0) + 1
    config[key] = next_id
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    return next_id

# -----------------------------------------------------------------------------
# 2. GLOBAL DOMAIN ADAPTATION: CORAL
# -----------------------------------------------------------------------------
def coral_alignment(source_X, target_X):
    """
    Correlation Alignment (CORAL):
    Aligns the covariance of the ENTIRE test-session population 
    to the ENTIRE training-session population.
    """
    from scipy.linalg import sqrtm, inv
    
    # Flatten to 2D: (Samples*Nodes, Features)
    s = source_X.reshape(-1, source_X.shape[-1]).cpu().numpy()
    t = target_X.reshape(-1, target_X.shape[-1]).cpu().numpy()

    # Calculate Covariances
    cov_s = np.cov(s, rowvar=False) + np.eye(s.shape[1]) * 1e-5
    cov_t = np.cov(t, rowvar=False) + np.eye(t.shape[1]) * 1e-5

    # Map C_target to C_source
    transformation = inv(sqrtm(cov_t)) @ sqrtm(cov_s)
    
    # Apply and reshape back to 3D
    t_aligned = (t @ transformation).real
    return torch.tensor(t_aligned.reshape(target_X.shape), dtype=torch.float)

# -----------------------------------------------------------------------------
# 3. POPULATION CACHE LOADING
# -----------------------------------------------------------------------------
def load_full_population():
    """
    Aggregates cached data from all 15 subjects into a single global dataset.
    """
    label_file = "Data/ExtractedFeatures_1s/label.mat"
    labels = scipy.io.loadmat(label_file)['label'][0]
    label_map = {-1: 0, 0: 1, 1: 2}
    
    X_all, y_all, sess_all, sub_all = [], [], [], []

    print("--- Loading Global Population Cache (Subjects 1-15) ---")
    for sub_id in range(1, 16):
        for sess_id in range(1, 4):
            for trial_id in range(1, 16):
                path = os.path.join(CACHE_FOLDER, f"Sub{sub_id}_Sess{sess_id}_Trial{trial_id}.npy")
                if os.path.exists(path):
                    data = np.load(path) # (Time, 62, 10)
                    X_all.append(data)
                    y_all.append(np.full(data.shape[0], label_map[labels[trial_id-1]]))
                    sess_all.append(np.full(data.shape[0], sess_id))
                    sub_all.append(np.full(data.shape[0], sub_id))
                    
    return (np.concatenate(X_all, axis=0), 
            np.concatenate(y_all, axis=0), 
            np.concatenate(sess_all, axis=0),
            np.concatenate(sub_all, axis=0))

# -----------------------------------------------------------------------------
# 4. THE GLOBAL WORKFLOW
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=str, default='1s')
    parser.add_argument('--aggregator', type=str, default='max')
    args = parser.parse_args()

    # 1. Load EVERYTHING into memory instantly
    X_raw, y_raw, sess_raw, sub_raw = load_full_population()
    X_tensor = torch.tensor(X_raw, dtype=torch.float)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
    # Graph Setup
    base_edge_index, _ = get_base_edge_index(LOCS_FILE, k=5)
    base_edge_index = base_edge_index.to(DEVICE)

    # 2. Global Permutations (Cross-Session Validation)
    permutations = [([1, 2], 3), ([1, 3], 2), ([2, 3], 1)]
    run_id = get_next_run_id(args.window_size)
    
    final_cross_session_scores = []

    for train_sess, test_sess in permutations:
        print(f"\n\n" + "="*70)
        print(f"TRAINING GLOBAL MODEL | TEST SESSION: {test_sess}")
        print("="*70)

        # Split the entire population at once
        train_mask = np.isin(sess_raw, train_sess)
        test_mask = (sess_raw == test_sess)

        X_train = X_tensor[train_mask].contiguous()
        X_test_unaligned = X_tensor[test_mask]

        # Apply Global CORAL
        print(f"  [CORAL] Aligning Global Test Population Cloud...")
        X_test = coral_alignment(X_train, X_test_unaligned).contiguous()

        # Loaders
        train_loader = DataLoader(TensorDataset(X_train, y_tensor[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_tensor[test_mask]), batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = GraphSAGE_EEG_Model(in_features=10, hidden_dim=128, aggregator=args.aggregator).to(DEVICE)
        
        # Optimization
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        # Defensive Loss
        class_weights = torch.tensor([1.5, 1.0, 1.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # Setup paths
        run_name = f"Global_SAGE_1s_TestSess{test_sess}"
        results_dir = f"Results/SAGE_Global/Attempt_{run_id}/{run_name}"
        params_dir = f"Params/SAGE_Global/Attempt_{run_id}/{run_name}"
        errors_dir = f"Errors/SAGE_Global/Attempt_{run_id}/{run_name}"
        for d in [results_dir, params_dir, errors_dir]: os.makedirs(d, exist_ok=True)

        # 3. Train 1 of the 3 Models
        train_model_with_interrupt(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=EPOCHS,
            device=DEVICE,
            results_dir=results_dir,
            params_dir=params_dir,
            errors_dir=errors_dir,
            base_edge_index=base_edge_index,
            evaluate_fn=evaluate,
            in_features=10
        )
        
        # Final Evaluation for this Session Holdout
        _, acc = evaluate(model, test_loader, base_edge_index, criterion, DEVICE, 10)
        final_cross_session_scores.append(acc)

    # 4. Summary
    print(f"\n\n" + "#"*70)
    print(f"FINAL GLOBAL DATASET ACCURACY: {np.mean(final_cross_session_scores):.2f}%")
    print(f"Individual Session Holdout Scores: {final_cross_session_scores}")
    print("#"*70)

if __name__ == "__main__":
    main()