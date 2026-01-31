# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np
# import scipy.io
# import os
# import argparse
# import sys
# import json
# import time
# from datetime import datetime

# # --- CUSTOM PROJECT IMPORTS ---
# from Models.var_ind_graph import GraphSAGE_EEG_Model
# from utils.inductive_graph import get_base_edge_index
# from utils.training_utils import train_model_with_interrupt, evaluate
# from utils.feature_engineering import SmartPreprocessor, get_standard_channel_names

# # --- Configuration ---
# LOCS_FILE = "utils/channel_62_pos.locs"
# BATCH_SIZE = 128
# EPOCHS = 120
# LEARNING_RATE = 0.0001
# WEIGHT_DECAY = 1e-3 
# PATIENCE = 30
# # --- NEW: Sparsity Penalty for Adaptive Layer ---
# L1_LAMBDA = 1e-4  # Force gamma parameters towards zero
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIG_FILE = "run_config.json"

# # --- STRATEGY CONFIGURATION ---
# # HARD_SUBJECTS = [2, 7, 12, 13] # Subjects with systemic artifacts/sinkholes
# ROLLING_VAR_WINDOW = 9         # Window size for generating variance features

# def get_next_run_id(window_size):
#     """Reads and increments the run counter from run_config.json for the specific window size"""
#     if not os.path.exists(CONFIG_FILE):
#         with open(CONFIG_FILE, 'w') as f:
#             json.dump({}, f)
            
#     with open(CONFIG_FILE, 'r') as f:
#         try:
#             config = json.load(f)
#         except json.JSONDecodeError:
#             config = {}
    
#     key = f"run_counter_{window_size}"
#     next_id = config.get(key, 0) + 1
    
#     config[key] = next_id
#     with open(CONFIG_FILE, 'w') as f:
#         json.dump(config, f, indent=4)
        
#     return next_id



# def compute_rolling_variance(data, window_size=3):
#     """
#     Computes rolling variance along the time axis (axis 1).
#     Input: (62, samples, 5)
#     Output: (62, samples, 5)
#     """
#     pad_width = window_size // 2
#     padded = np.pad(data, ((0,0), (pad_width, pad_width), (0,0)), mode='edge')
    
#     vars_list = []
#     for i in range(data.shape[1]):
#         slice_data = padded[:, i : i + window_size, :]
#         vars_list.append(np.var(slice_data, axis=1))
        
#     return np.stack(vars_list, axis=1)

# def load_de_data(data_folder, label_file):
#     print(f"Loading DE features from {data_folder}...")
#     try:
#         label_mat = scipy.io.loadmat(label_file)
#         trial_labels = label_mat['label'][0]
#     except FileNotFoundError:
#         print(f"Error: Label file not found at {label_file}")
#         sys.exit(1)
    
#     label_map = {-1: 0, 0: 1, 1: 2}
#     mapped_labels = [label_map[l] for l in trial_labels]
    
#     X_list = []
#     y_list = []
#     session_list = []
#     subject_list = []
#     trial_list = []
    
#     files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
#     subject_files = {}
#     for f in files:
#         parts = f.split('_')
#         try:
#             subj_id = int(parts[0])
#         except ValueError: continue
#         if subj_id not in subject_files: subject_files[subj_id] = []
#         subject_files[subj_id].append(f)

#     # --- Initialize Smart Preprocessor ---
#     channel_names = get_standard_channel_names()
#     # preprocessor = SmartPreprocessor(channel_names) # DISABLED for Attempt 18 reproduction
#     # print("Initialized Smart Preprocessor for Bad Channel Correction.")
#     # -------------------------------------

#     band_weights = None
#     print("Warning: Manual band weights DISABLED. Using raw data.")
        
#     for subj_id in sorted(subject_files.keys()):
#         s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
#         for sess_idx, fname in enumerate(s_files):
#             session_id = sess_idx + 1
#             file_path = os.path.join(data_folder, fname)
#             try: mat = scipy.io.loadmat(file_path)
#             except: continue
#             for trial_i in range(1, 16):
#                 key = f"de_LDS{trial_i}"
#                 if key not in mat: continue
#                 data = mat[key]
                
#                 # --- ROBUST SHAPE CORRECTION ---
#                 # Target: (62, samples, 5)
#                 # We identify dimensions by size: 62=Channels, 5=Bands
#                 shape = data.shape
#                 if shape[0] == 62:
#                     if shape[2] == 5:
#                         pass # Already (62, samples, 5)
#                     elif shape[1] == 5:
#                         data = np.transpose(data, (0, 2, 1)) # (62, 5, samples) -> (62, samples, 5)
#                 elif shape[1] == 62:
#                     if shape[2] == 5:
#                         data = np.transpose(data, (1, 0, 2)) # (samples, 62, 5) -> (62, samples, 5)
#                     elif shape[0] == 5:
#                         data = np.transpose(data, (1, 2, 0)) # (5, 62, samples) -> (62, samples, 5)
#                 elif shape[2] == 62:
#                     if shape[1] == 5:
#                         data = np.transpose(data, (2, 0, 1)) # (samples, 5, 62) -> (62, samples, 5)
#                     elif shape[0] == 5:
#                         data = np.transpose(data, (2, 1, 0)) # (5, samples, 62) -> (62, samples, 5)
#                 # -------------------------------

#                 # --- ATTEMPT 18 LOGIC: MANUAL CF2 FIX ONLY ---
#                 # We blindly apply the fix to CF2 because we know it is often broken.
#                 # if 'CF2' in channel_names:
#                 #     cf2_idx = channel_names.index('CF2')
#                 #     # Check if neighbors exist in the dataset
#                 #     n1, n2, n3 = 'FC2', 'C2', 'CP2'
#                 #     if n1 in channel_names and n2 in channel_names and n3 in channel_names:
#                 #         idx1 = channel_names.index(n1)
#                 #         idx2 = channel_names.index(n2)
#                 #         idx3 = channel_names.index(n3)
                        
#                 #         # Apply Triangulation Average
#                 #         data[cf2_idx, :, :] = (data[idx1, :, :] + data[idx2, :, :] + data[idx3, :, :]) / 3
#                 # -----------------------------------------------------------

#                 # --- RESTORED: Variance Calculation ---
#                 # Compute rolling variance (Strategy V2)
#                 data_var = compute_rolling_variance(data, window_size=ROLLING_VAR_WINDOW)
                
#                 # Stack: (62, samples, 5) + (62, samples, 5) -> (62, samples, 10)
#                 data_final = np.concatenate([data, data_var], axis=2)

#                 # Transpose to (samples, 62, 10) for storage/model input
#                 data_final = np.transpose(data_final, (1, 0, 2))

#                 # 5 ORIGINAL MEAN FEATURES ONLY (FOR ATTEMPT 18 REPRODUCTION)
#                 # data_final = np.transpose(data, (1, 0, 2))

#                 num_samples = data_final.shape[0]
#                 X_list.append(data_final)
#                 y_list.append(np.full(num_samples, mapped_labels[trial_i - 1]))
#                 session_list.append(np.full(num_samples, session_id))
#                 subject_list.append(np.full(num_samples, subj_id))
#                 unique_trial_id = subj_id * 1000 + session_id * 100 + trial_i
#                 trial_list.append(np.full(num_samples, unique_trial_id))


#     if not X_list:
#         print("Error: No data loaded.")
#         sys.exit(1)

#     X = np.concatenate(X_list, axis=0)
#     y = np.concatenate(y_list, axis=0)
#     sessions = np.concatenate(session_list, axis=0)
#     subjects = np.concatenate(subject_list, axis=0)
#     trials = np.concatenate(trial_list, axis=0)
#     return X, y, sessions, subjects, trials
# # -----------------------------------------------------------------------------
# # DOMAIN ADAPTATION: SESSION-AWARE NORMALIZATION
# # -----------------------------------------------------------------------------

# def session_aware_normalize(X, sessions, channel_names):
#     """
#     Battles Euclidean Drift by centering each session's feature cloud independently.
#     Now includes Scaling Factor diagnostics to track session intensity drift.
#     """
#     preprocessor = SmartPreprocessor(channel_names)
#     X_normalized = np.zeros_like(X)
    
#     unique_sessions = np.unique(sessions)
#     for sess in unique_sessions:
#         mask = (sessions == sess)
#         if np.sum(mask) == 0: continue
        
#         # --- NEW: INTENSITY DIAGNOSTICS ---
#         # Calculate stats BEFORE normalization to see the raw drift
#         # X[mask] contains all timepoints for this session (e.g., 3394)
#         q75, q25 = np.percentile(X[mask], [75, 25])
#         iqr = q75 - q25
#         raw_mean = np.mean(X[mask])
#         raw_std = np.std(X[mask])
        
#         print(f"  [Domain Stats] Session {sess} | Raw Mean: {raw_mean:.4f} | IQR (Intensity): {iqr:.4f} | Std: {raw_std:.4f}")
        
#         # Run the actual preprocessing
#         # This will trigger the 'Mapping non-standard channels' message each time
#         X_normalized[mask] = preprocessor.process_subject(X[mask])
        
#     return torch.tensor(X_normalized, dtype=torch.float)

# # -----------------------------------------------------------------------------
# # DOMAIN ADAPTATION: CORAL ALIGNMENT
# # -----------------------------------------------------------------------------

# def coral_alignment(source_X, target_X):
#     """
#     CORAL (Correlation Alignment):
#     Think of this as a 'Warping' tool. It takes the shape of the test data
#     distribution and warps it to match the shape of the training data.
#     """
#     from scipy.linalg import sqrtm, inv
    
#     # 1. Flatten the data into 2D (Total_Nodes, Features)
#     # We treat all electrodes and timepoints as one big cloud of data points
#     s = source_X.reshape(-1, source_X.shape[-1]).cpu().numpy()
#     t = target_X.reshape(-1, target_X.shape[-1]).cpu().numpy()

#     # 2. Calculate the 'Covariance' (The shape/direction of the cloud)
#     # We add a tiny bit of identity matrix (1e-5) so the math doesn't explode
#     cov_s = np.cov(s, rowvar=False) + np.eye(s.shape[1]) * 1e-5
#     cov_t = np.cov(t, rowvar=False) + np.eye(t.shape[1]) * 1e-5

#     # 3. Create the 'Translation Map' (The Transformation Matrix)
#     # This math finds how to turn Cov_T into Cov_S
#     whitening = inv(sqrtm(cov_t))
#     coloring = sqrtm(cov_s)
#     transformation = whitening @ coloring

#     # 4. Apply the map to the target data
#     # We use .real because sqrtm can sometimes produce tiny imaginary numbers
#     t_aligned = (t @ transformation).real
    
#     # 5. Reshape it back to the original 3D shape (T, 62, 10)
#     return torch.tensor(t_aligned.reshape(target_X.shape), dtype=torch.float)

# # -----------------------------------------------------------------------------
# # MAIN EXECUTION
# # -----------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'])
#     parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'])
#     parser.add_argument('--test_subject', type=int, default=1)
#     parser.add_argument('--aggregator', type=str, default='max')
#     args = parser.parse_args()

#     # 1. Directory & Run ID
#     data_folder = f"Data/ExtractedFeatures_{args.window_size}"
#     label_file = os.path.join(data_folder, "label.mat")
#     run_id = get_next_run_id(args.window_size)
    
#     model_name = f"GraphSAGE_Advanced_{args.window_size}"
#     strategy_tag = "Phase2" if args.mode == 'sub_dep' else "LOSO_Phase2"
#     run_name = f"Attempt_{run_id}_{strategy_tag}"
    
#     results_root = os.path.join("Results", model_name, run_name)
#     params_root = os.path.join("Params", model_name, run_name)
#     errors_root = os.path.join("Errors", model_name, run_name)

#     # 2. Bulk Data Loading
#     try:
#         X_full, y_full, sessions_full, subjects_full, trials_full = load_de_data(data_folder, label_file)
#     except SystemExit:
#         print("Stopping execution due to loading failure.")
#         return

#     channel_names = get_standard_channel_names()
#     base_edge_index, coords = get_base_edge_index(LOCS_FILE, k=5)
#     base_edge_index = base_edge_index.to(DEVICE)
    
#     print(f"Loaded Data: {X_full.shape} samples across {len(np.unique(subjects_full))} subjects.")

#     # 3. EXPERIMENT: SUBJECT-DEPENDENT PERMUTATION LOOP
#     if args.mode == 'sub_dep':
        
#         # We average results across all session-holdout combinations
#         permutations = [([1, 2], 3), ([1, 3], 2), ([2, 3], 1)]

#         for train_sess, test_sess in permutations:
#             print(f"\n>>> PERMUTATION: Train {train_sess} -> Test {test_sess}")
            
#             p_tag = f"TestSess_{test_sess}"
#             p_res = os.path.join(results_root, p_tag)
#             p_par = os.path.join(params_root, p_tag)
#             p_err = os.path.join(errors_root, p_tag)
#             for d in [p_res, p_par, p_err]: os.makedirs(d, exist_ok=True)

#             X_train = session_aware_normalize(X_full[np.isin(sessions_full, train_sess)], 
#                                              sessions_full[np.isin(sessions_full, train_sess)], 
#                                              channel_names).contiguous()  
#             X_test = session_aware_normalize(X_full[sessions_full == test_sess], 
#                                             sessions_full[sessions_full == test_sess], channel_names)
            
#             # 2. NEW: Now we 'Warp' the test session to match the train sessions
#             # This is where CORAL happens!
#             print(f"  [CORAL] Aligning Test Session {test_sess} to Training Distribution...")
#             X_test = coral_alignment(X_train, X_test)

#             y_train = torch.tensor(y_full[np.isin(sessions_full, train_sess)], dtype=torch.long)
#             y_test = torch.tensor(y_full[sessions_full == test_sess], dtype=torch.long)

#             if len(X_train) == 0 or len(X_test) == 0:
#                 print("Warning: Empty train or test set. Skipping permutation.")
#                 continue

#             model = GraphSAGE_EEG_Model(in_features=10, hidden_dim=128, aggregator=args.aggregator).to(DEVICE)
#             gamma_params = []
#             other_params = []
            
#             for name, param in model.named_parameters():
#                 if 'static_norm.gamma' in name:
#                     gamma_params.append(param)
#                 else:
#                     other_params.append(param)
                    
#             optimizer = optim.Adam([
#                 {'params': other_params, 'weight_decay': WEIGHT_DECAY}, # Normal L2
#                 {'params': gamma_params, 'weight_decay': 1e-2}          # Strong L2 (force small weights)
#             ], lr=LEARNING_RATE)
            
#             scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
#                                                             factor=0.5, patience=PATIENCE)
            
#             # --- STRATEGY: Class Weights ---
#             # Double penalty for Negative (Class 0) to fix Recall
#             class_weights = torch.tensor([1.2, 0.9, 1.0]).to(DEVICE)
#             criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

#             # --- DATA INTEGRITY DIAGNOSTIC ---
#             print(f"\n[DIAGNOSTIC] Verification for Permutation: Test Session {test_sess}")
#             print(f"  X_train Shape: {X_train.shape} | y_train Shape: {y_train.shape}")

#             # 1. Check for Label Imbalance
#             unique, counts = np.unique(y_train.numpy(), return_counts=True)
#             label_dist = dict(zip(unique, counts))
#             print(f"  Label Distribution in Train: {label_dist}")
#             if 0 not in label_dist:
#                 print("  !!! ERROR: CLASS 0 (NEGATIVE) IS MISSING FROM TRAINING DATA !!!")

#             # 2. Check for Data Corruption (NaNs or Infinity)
#             if torch.isnan(X_train).any() or torch.isinf(X_train).any():
#                 print("  !!! ERROR: X_train CONTAINS NaNs OR INF (CORAL might have exploded) !!!")

#             # 3. Check Feature Ranges (Is CORAL squashing the data?)
#             print(f"  Feature 0 (Delta) - Mean: {X_train[:,:,0].mean():.4f}, Std: {X_train[:,:,0].std():.4f}")
#             print(f"  Feature 4 (Gamma) - Mean: {X_train[:,:,4].mean():.4f}, Std: {X_train[:,:,4].std():.4f}")

#             # 4. Check for 'Dead' Data (Is it all zeros?)
#             if X_train.std() < 1e-6:
#                 print("  !!! ERROR: X_train IS CONSTANT. Normalization/CORAL killed the signal !!!")

#             # pause to let you read
#             time.sleep(2)

#             train_model_with_interrupt(
#                 model=model,
#                 train_loader=DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
#                 test_loader=DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
#                 optimizer=optimizer,
#                 criterion=criterion,
#                 scheduler=scheduler,
#                 epochs=EPOCHS,
#                 device=DEVICE,
#                 results_dir=p_res,
#                 params_dir=p_par,
#                 errors_dir=p_err,
#                 base_edge_index=base_edge_index,
#                 evaluate_fn=evaluate,
#                 in_features=10
#             )

#     # 4. EXPERIMENT: SUBJECT-INDEPENDENT (LOSO)
#     elif args.mode == 'sub_indep':
#         for d in [results_root, params_root, errors_root]: os.makedirs(d, exist_ok=True)
#         print(f"\n>>> LOSO Strategy: Testing on Subject {args.test_subject}")
        
#         test_mask = (subjects_full == args.test_subject)
        
#         # Center Domains for the whole subject split
#         X_train = session_aware_normalize(X_full[~test_mask], sessions_full[~test_mask], channel_names)
#         X_test = session_aware_normalize(X_full[test_mask], sessions_full[test_mask], channel_names)
        
#         y_train = torch.tensor(y_full[~test_mask], dtype=torch.long)
#         y_test = torch.tensor(y_full[test_mask], dtype=torch.long)

#         model = GraphSAGE_EEG_Model(in_features=10, hidden_dim=128, aggregator=args.aggregator).to(DEVICE)
#         gamma_params = []
#         other_params = []
        
#         for name, param in model.named_parameters():
#             if 'static_norm.gamma' in name:
#                 gamma_params.append(param)
#             else:
#                 other_params.append(param)
                
#         optimizer = optim.Adam([
#             {'params': other_params, 'weight_decay': WEIGHT_DECAY}, # Normal L2
#             {'params': gamma_params, 'weight_decay': 1e-2}          # Strong L2 (force small weights)
#         ], lr=LEARNING_RATE)
        
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
#                                                         factor=0.5, patience=PATIENCE)
        
#         # --- STRATEGY: Class Weights ---
#         # Double penalty for Negative (Class 0) to fix Recall
#         class_weights = torch.tensor([1.2, 0.9, 1.0]).to(DEVICE)
#         criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

#         train_model_with_interrupt(
#             model=model,
#             train_loader=DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
#             test_loader=DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
#             optimizer=optimizer,
#             criterion=criterion,
#             scheduler=scheduler,
#             epochs=args.epochs,
#             device=DEVICE,
#             results_dir=results_root,
#             params_dir=params_root,
#             errors_dir=errors_root,
#             base_edge_index=base_edge_index,
#             evaluate_fn=evaluate,
#             in_features=10
#         )


# if __name__ == "__main__":
#     main()





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
import time
from datetime import datetime

# --- CUSTOM PROJECT IMPORTS ---
from Models.var_ind_graph import GraphSAGE_EEG_Model
from utils.inductive_graph import get_base_edge_index
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.feature_engineering import SmartPreprocessor, get_standard_channel_names

# --- GLOBAL CONFIGURATION ---
LOCS_FILE = "utils/channel_62_pos.locs"
CACHE_FOLDER = "Data/Cache_GraphSAGE_1s"
CONFIG_FILE = "run_config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 120
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-3      # Strong L2 to prevent stimulus-specific overfitting
L1_LAMBDA = 1e-4        # Sparsity penalty for the Adaptive Layer (AGLI)
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
    key = f"run_counter_{window_size}"
    next_id = config.get(key, 0) + 1
    config[key] = next_id
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    return next_id

# -----------------------------------------------------------------------------
# 2. DOMAIN ADAPTATION: CORAL ALIGNMENT
# -----------------------------------------------------------------------------
def coral_alignment(source_X, target_X):
    """
    Correlation Alignment (CORAL):
    Aligns the second-order statistics (covariance) of the test session 
    to the training distribution to minimize geometric session drift.
    """
    from scipy.linalg import sqrtm, inv
    
    # Flatten: (Samples, Nodes, Features) -> (Samples*Nodes, Features)
    # Using .reshape() and .numpy() for compatibility
    s = source_X.reshape(-1, source_X.shape[-1]).cpu().numpy()
    t = target_X.reshape(-1, target_X.shape[-1]).cpu().numpy()

    # Calculate Covariance Matrices
    # 1e-5 added to identity for numerical stability during inversion
    cov_s = np.cov(s, rowvar=False) + np.eye(s.shape[1]) * 1e-5
    cov_t = np.cov(t, rowvar=False) + np.eye(t.shape[1]) * 1e-5

    # Transformation Matrix: T = C_t^(-1/2) * C_s^(1/2)
    whitening = inv(sqrtm(cov_t))
    coloring = sqrtm(cov_s)
    transformation = whitening @ coloring

    # Apply transformation and return to original 3D shape
    t_aligned = (t @ transformation).real
    return torch.tensor(t_aligned.reshape(target_X.shape), dtype=torch.float)

# -----------------------------------------------------------------------------
# 3. CACHE LOADING LOGIC
# -----------------------------------------------------------------------------
def load_subject_from_cache(sub_id):
    """
    Instantly loads pre-processed (10-feature) tensors from the cache folder.
    """
    label_file = "Data/ExtractedFeatures_1s/label.mat"
    labels = scipy.io.loadmat(label_file)['label'][0]
    label_map = {-1: 0, 0: 1, 1: 2} # Map SEED to PyTorch labels
    
    X_list, y_list, sess_list = [], [], []

    for sess_id in range(1, 4):
        for trial_id in range(1, 16):
            cache_path = os.path.join(CACHE_FOLDER, f"Sub{sub_id}_Sess{sess_id}_Trial{trial_id}.npy")
            if os.path.exists(cache_path):
                data = np.load(cache_path) # Shape: (Time, 62, 10)
                X_list.append(data)
                y_list.append(np.full(data.shape[0], label_map[labels[trial_id-1]]))
                sess_list.append(np.full(data.shape[0], sess_id))
                
    if not X_list:
        return None, None, None
        
    return (np.concatenate(X_list, axis=0), 
            np.concatenate(y_list, axis=0), 
            np.concatenate(sess_list, axis=0))

# -----------------------------------------------------------------------------
# 4. FORENSIC DIAGNOSTICS
# -----------------------------------------------------------------------------
def run_diagnostics(X_train, y_train, test_sess):
    print(f"\n[DIAGNOSTIC] Verification for Test Session {test_sess}")
    
    # Label Distribution Check
    unique, counts = np.unique(y_train.numpy(), return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"  Label Dist: {dist}")
    
    # Feature Range Check
    print(f"  Delta Mean: {X_train[:,:,0].mean():.4f} | Gamma Mean: {X_train[:,:,4].mean():.4f}")
    
    # Signal Collapse Check
    if X_train.std() < 1e-5:
        print("  !!! CRITICAL: SIGNAL COLLAPSE DETECTED (Check Normalization) !!!")
    
    time.sleep(1)

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION LOOP (15 SUBJECTS)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=str, default='1s')
    parser.add_argument('--aggregator', type=str, default='max')
    args = parser.parse_args()

    # Setup directories
    run_id = get_next_run_id(args.window_size)
    model_name = f"GraphSAGE_Advanced_{args.window_size}"
    run_name = f"Attempt_{run_id}_FullDataset"
    
    # Graph Template
    channel_names = get_standard_channel_names()
    base_edge_index, _ = get_base_edge_index(LOCS_FILE, k=5)
    base_edge_index = base_edge_index.to(DEVICE)

    global_results = []

    # START THE 15-SUBJECT MARATHON
    for sub_id in range(1, 16):
        print(f"\n" + "#"*70)
        print(f"##  PROCESSING SUBJECT {sub_id} OF 15")
        print("#"*70)

        # 1. Load from Cache
        X_sub_raw, y_sub_raw, sess_sub = load_subject_from_cache(sub_id)
        if X_sub_raw is None:
            print(f"  Skipping Sub {sub_id}: No cache found.")
            continue
            
        X_sub = torch.tensor(X_sub_raw, dtype=torch.float)
        y_sub = torch.tensor(y_sub_raw, dtype=torch.long)

        # 2. Permutation Looping (Cross-Session CV)
        permutations = [([1, 2], 3), ([1, 3], 2), ([2, 3], 1)]
        sub_accuracies = []

        for train_sess, test_sess in permutations:
            print(f"\n>>> Permutation: Train {train_sess} -> Test {test_sess}")
            
            # Create sub-folders for this specific permutation
            p_dir = f"Results/{model_name}/{run_name}/Sub{sub_id}/Test{test_sess}"
            m_dir = f"Params/{model_name}/{run_name}/Sub{sub_id}/Test{test_sess}"
            e_dir = f"Errors/{model_name}/{run_name}/Sub{sub_id}/Test{test_sess}"
            for d in [p_dir, m_dir, e_dir]: os.makedirs(d, exist_ok=True)

            # Split Data
            train_mask = np.isin(sess_sub, train_sess)
            test_mask = (sess_sub == test_sess)
            
            X_train = X_sub[train_mask].contiguous()
            X_test_raw = X_sub[test_mask]

            # 3. Apply CORAL Alignment
            print(f"  [CORAL] Warping Test Session {test_sess} to Training Feature Cloud...")
            X_test = coral_alignment(X_train, X_test_raw).contiguous()

            # Diagnostic Check
            run_diagnostics(X_train, y_sub[train_mask], test_sess)

            # DataLoaders
            train_loader = DataLoader(TensorDataset(X_train, y_sub[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_sub[test_mask]), batch_size=BATCH_SIZE, shuffle=False)

            # 4. Model & Optimizer Setup
            model = GraphSAGE_EEG_Model(in_features=10, hidden_dim=128, aggregator=args.aggregator).to(DEVICE)
            
            # Split parameters for specialized weight decay on AGLI
            gamma_params = [p for n, p in model.named_parameters() if 'agli.gamma' in n]
            other_params = [p for n, p in model.named_parameters() if 'agli.gamma' not in n]
            
            optimizer = optim.Adam([
                {'params': other_params, 'weight_decay': WEIGHT_DECAY},
                {'params': gamma_params, 'weight_decay': 1e-2} # Force sparsity on sensor trust
            ], lr=LEARNING_RATE)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
            
            # Defensive Loss: Weights protect Class 0, Smoothing prevents stimulus-overfitting
            class_weights = torch.tensor([1.2, 0.9, 1.0]).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

            # 5. Training Call
            train_model_with_interrupt(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                epochs=EPOCHS,
                device=DEVICE,
                results_dir=p_dir,
                params_dir=m_dir,
                errors_dir=e_dir,
                base_edge_index=base_edge_index,
                evaluate_fn=evaluate,
                in_features=10
            )
            
            # Capture results
            _, acc = evaluate(model, test_loader, base_edge_index, criterion, DEVICE, 10)
            sub_accuracies.append(acc)

        print(f"\nSubject {sub_id} Mean Cross-Session Accuracy: {np.mean(sub_accuracies):.2f}%")
        global_results.append(np.mean(sub_accuracies))

    print(f"\n\n" + "="*70)
    print(f"FINAL DATASET REPORT: Global Mean Accuracy: {np.mean(global_results):.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()