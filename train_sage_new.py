import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.io
import os
import argparse
import json
import time

# --- CUSTOM PROJECT IMPORTS ---
from Models.var_ind_graph import GraphSAGE_EEG_Model
from utils.inductive_graph import get_base_edge_index
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.feature_engineering import get_standard_channel_names

# --- CONFIG ---
CACHE_FOLDER = "Data/Cache_GraphSAGE_1s"
LOCS_FILE = "utils/channel_62_pos.locs"
CONFIG_FILE = "run_config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attempt 18 Hyperparameters
BATCH_SIZE = 256 
EPOCHS = 120
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-3  
L1_LAMBDA = 1e-4
PATIENCE = 30

# -----------------------------------------------------------------------------
# 1. ATTEMPT 18: MANUAL Z-SCORE LOGIC
# -----------------------------------------------------------------------------
def apply_manual_z_score(train_data, test_data):
    """
    Implements Attempt 18 normalization properly.
    Fits the mean/std on training data to avoid leakage.
    """
    mu = torch.mean(train_data)
    sigma = torch.std(train_data)
    
    train_norm = (train_data - mu) / sigma
    test_norm = (test_data - mu) / sigma
    
    print(f"  [Attempt 18] Normalizing with Mu: {mu:.4f}, Sigma: {sigma:.4f}")
    return train_norm, test_norm

def apply_subject_wise_z_score(X, subjects):
    """
    Crucial for Subject-Independent (sub_indep).
    Centers each brain independently so the model sees universal patterns.
    """
    X_new = torch.zeros_like(X)
    unique_subs = np.unique(subjects)
    
    for sub in unique_subs:
        mask = (subjects == sub)
        sub_data = X[mask]
        mu = torch.mean(sub_data)
        sigma = torch.std(sub_data)
        X_new[mask] = (sub_data - mu) / sigma
        
    print(f"  [LOSO-Opt] Subject-wise Z-scoring completed for {len(unique_subs)} subjects.")
    return X_new

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
def load_global_cache():
    label_file = "Data/ExtractedFeatures_1s/label.mat"
    labels = scipy.io.loadmat(label_file)['label'][0]
    label_map = {-1: 0, 0: 1, 1: 2}
    
    X_all, y_all, sess_all, sub_all = [], [], [], []
    
    for sub_id in range(1, 16):
        for sess_id in range(1, 4):
            for trial_id in range(1, 16):
                path = os.path.join(CACHE_FOLDER, f"Sub{sub_id}_Sess{sess_id}_Trial{trial_id}.npy")
                if os.path.exists(path):
                    data = np.load(path)
                    X_all.append(data)
                    y_all.append(np.full(data.shape[0], label_map[labels[trial_id-1]]))
                    sess_all.append(np.full(data.shape[0], sess_id))
                    sub_all.append(np.full(data.shape[0], sub_id))
                    
    return (np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), 
            np.concatenate(sess_all, axis=0), np.concatenate(sub_all, axis=0))

# -----------------------------------------------------------------------------
# 3. MASTER EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sub_indep', choices=['sub_dep', 'sub_indep'])
    parser.add_argument('--test_subject', type=int, default=1)
    parser.add_argument('--aggregator', type=str, default='max', choices=['mean', 'max', 'pool'])
    args = parser.parse_args()

    # Load Data
    X_raw, y_raw, sess_raw, sub_raw = load_global_cache()
    X_tensor = torch.tensor(X_raw, dtype=torch.float)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
    base_edge_index, _ = get_base_edge_index(LOCS_FILE, k=5)
    base_edge_index = base_edge_index.to(DEVICE)

    # A. SUBJECT-DEPENDENT (The 3 Global Session Models)
    if args.mode == 'sub_dep':
        permutations = [([1, 2], 3), ([1, 3], 2), ([2, 3], 1)]
        
        for train_sess, test_sess in permutations:
            print(f"\n>>> MODE: SUB_DEP | PERM: Train {train_sess} -> Test {test_sess}")
            
            train_mask = np.isin(sess_raw, train_sess)
            test_mask = (sess_raw == test_sess)
            
            # --- ATTEMPT 18 NORMALIZATION ---
            X_train, X_test = apply_manual_z_score(X_tensor[train_mask], X_tensor[test_mask])
            
            run_training(X_train, y_tensor[train_mask], X_test, y_tensor[test_mask], 
                        base_edge_index, f"Dep_TestSess{test_sess}", args.aggregator)

    # B. SUBJECT-INDEPENDENT (The LOSO Strategy)
    elif args.mode == 'sub_indep':
        print(f"\n>>> MODE: SUB_INDEP | HELD-OUT SUBJECT: {args.test_subject}")
        
        # 1. OPTIMIZATION: Subject-wise Normalization BEFORE splitting
        # This makes different subjects comparable while keeping the LDS energy relative.
        X_norm = apply_subject_wise_z_score(X_tensor, sub_raw)
        
        test_mask = (sub_raw == args.test_subject)
        
        X_train = X_norm[~test_mask]
        X_test = X_norm[test_mask]
        
        run_training(X_train, y_tensor[~test_mask], X_test, y_tensor[test_mask], 
                    base_edge_index, f"Indep_Sub{args.test_subject}", args.aggregator)

def run_training(X_train, y_train, X_test, y_test, edge_index, tag, aggregator):
    model = GraphSAGE_EEG_Model(in_features=10, hidden_dim=128, aggregator=aggregator).to(DEVICE)
    
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

    res_dir = f"Results/Master/Run_{tag}"
    os.makedirs(res_dir, exist_ok=True)

    train_model_with_interrupt(
        model=model,
        train_loader=DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
        test_loader=DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        results_dir=res_dir,
        params_dir=res_dir,
        errors_dir=res_dir,
        base_edge_index=edge_index,
        evaluate_fn=evaluate,
        in_features=10
    )

if __name__ == "__main__":
    main()