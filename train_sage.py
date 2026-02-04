# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import scipy.io
# import os
# import argparse
# import sys
# import json
# from datetime import datetime

# from Models.var_B import GCN_DE_Model
# from Models.var_C import DGCNN_Model
# from Models.var_D import Adaptive_DGCNN
# from Models.var_ind_graph import GraphSAGE_EEG_Model
# from Models.graph_construction import get_knn_adjacency_matrix
# from utils.training_utils import train_model_with_interrupt, evaluate
# from utils.feature_engineering import SmartPreprocessor, get_standard_channel_names
# # from sklearn.preprocessing import RobustScaler # REMOVED to match Attempt 18
# from utils.focal_loss import FocalLoss

# import torch.multiprocessing as mp


# # --- Configuration ---
# LOCS_FILE = "utils/channel_62_pos.locs"
# BATCH_SIZE = 128
# EPOCHS = 100
# LEARNING_RATE = 0.0001
# WEIGHT_DECAY = 1e-3 
# PATIENCE = 25
# # --- NEW: Sparsity Penalty for Adaptive Layer ---
# L1_LAMBDA = 1e-4  # Force gamma parameters towards zero
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIG_FILE = "run_config.json"

# ROLLING_VAR_WINDOW = 3      # Window size for generating variance features


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

# def get_args():
#     parser = argparse.ArgumentParser(description="Train GCN-DE for EEG Emotion Recognition")
#     parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'],
#                         help="Training mode: 'sub_dep' (Session split) or 'sub_indep' (LOSO)")
#     parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'],
#                         help="Feature window size: '1s' or '4s'")
#     parser.add_argument('--model_type', type=str, default = 'GRAPH_SAGE', 
#                         choices=['GCN', 'DGCNN', 'ADAPTIVE_DGCNN', 'GRAPH_SAGE'],
#                         help="Type of GCN model to use")
#     parser.add_argument('--max_parallel', type=int, default=4, 
#                         help="Maximum number of parallel processes")
#     return parser.parse_args()

# def compute_rolling_variance(data, window_size=ROLLING_VAR_WINDOW):
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


# def compute_frontal_alpha_asymmetry(data, channel_names):
#     """
#     Computes FAS = ln(Right Alpha) - ln(Left Alpha)
#     Specifically using F4 (Right) and F3 (Left) Alpha band (index 2).
#     Input Data: (62, Samples, Bands)
#     Output: (1, Samples, 1) -> Broadcastable Feature
#     """
#     if 'F4' not in channel_names or 'F3' not in channel_names:
#         # Fallback if specific channels missing, return zeros
#         return np.zeros((1, data.shape[1], 1))
        
#     f4_idx = channel_names.index('F4')
#     f3_idx = channel_names.index('F3')
    
#     # Alpha band is index 2 (Delta, Theta, Alpha, Beta, Gamma)
#     # We use a small epsilon for log stability
#     alpha_right = data[f4_idx, :, 2] + 1e-6
#     alpha_left  = data[f3_idx, :, 2] + 1e-6
    
#     # Formula: ln(Right) - ln(Left)
#     fas = np.log(alpha_right) - np.log(alpha_left)
    
#     # Reshape to (1, Samples, 1) to match dimensionality requirements for concatenation
#     # We pretend this is a "global" feature for the whole brain or broadcast it
#     fas = fas.reshape(1, -1, 1) 
#     return fas


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

#                 # --- 1. Compute Rolling Variance (Feature 6-10) ---
#                 data_var = compute_rolling_variance(data, window_size=ROLLING_VAR_WINDOW)
                
#                 # --- 2. Compute Frontal Alpha Asymmetry (Feature 11) ---
#                 # Returns (1, Samples, 1)
#                 # fas_feature = compute_frontal_alpha_asymmetry(data, channel_names)
#                 # Broadcast FAS to all 62 nodes: (62, Samples, 1)
#                 # fas_feature = np.repeat(fas_feature, 62, axis=0)

#                 # --- 3. Stack All Features ---
#                 # Data: (62, S, 5) | Var: (62, S, 5) | FAS: (62, S, 1)
#                 # Total Features = 11
#                 data_combined = np.concatenate([data, data_var], axis=2)

#                 # Transpose to (samples, 62, 11) for storage/model input
#                 data_combined = np.transpose(data_combined, (1, 0, 2))


#                 # 5 ORIGINAL MEAN FEATURES ONLY (FOR ATTEMPT 18 REPRODUCTION)
#                 # data_final = np.transpose(data, (1, 0, 2))

#                 # 3. Calculate correct number of samples (dimension 0 of transposed data)
#                 num_samples = data_combined.shape[0]
                
#                 X_list.append(data_combined)
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
    
#     # --- ATTEMPT 18 REPRODUCTION: MANUAL Z-SCORE NORMALIZATION ---
#     print("Applying Manual Subject-Specific & Session-Specific Z-Score Normalization...")
    
#     group_ids = subjects * 1000 + sessions
#     unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    
#     n_groups = len(unique_groups)
    
#     # Initialize arrays for stats
#     # X shape: (N, 62, 10) -> Stats shape: (n_groups, 62, 10)
#     group_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
#     group_sq_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    
#     # Compute sums and counts using np.add.at (unbuffered in-place add)
#     np.add.at(group_sums, group_indices, X)
    
#     # Counts per group
#     group_counts = np.bincount(group_indices)
    
#     # Compute means: (n_groups, 62, 10)
#     # Reshape counts to (n_groups, 1, 1) for broadcasting
#     group_means = group_sums / group_counts[:, None, None]
    
#     # Broadcast means back to original sample shape
#     # shape: (N, 62, 10)
#     expanded_means = group_means[group_indices]
#     X_centered = X - expanded_means
    
#     # Compute Stds
#     np.add.at(group_sq_sums, group_indices, X_centered ** 2)
#     group_stds = np.sqrt(group_sq_sums / group_counts[:, None, None])
#     group_stds[group_stds < 1e-6] = 1.0 # Prevent div by zero
    
#     # Apply Normalization
#     expanded_stds = group_stds[group_indices]
#     X = X_centered / expanded_stds
#     print(f"Total Samples: {X.shape[0]}")
#     return X, y, sessions, subjects, trials



# # -----------------------------------------------------------------------------
# # WRAPPER FUNCTION FOR A SINGLE SUBJECT FOLD (Isolated Process)
# # -----------------------------------------------------------------------------
# def run_single_subject_fold(subject_id, args, X_full, y_full, sub_full, 
#                             base_edge_index, run_id, model_name):
#     """
#     Runs training and evaluation for one specific subject in a separate process.
#     """
#     # 1. Device Assignment
#     num_gpus = torch.cuda.device_count()
#     local_device = torch.device(f"cuda:{subject_id % num_gpus}" if num_gpus > 0 else "cpu")
#     print(f"\n[PROCESS START] Subject {subject_id} assigned to {local_device}")

#     # 2. Data Splitting (Train on 14, Test on 1)
#     test_mask = (sub_full == subject_id)
#     X_train, y_train = X_full[~test_mask], y_full[~test_mask]
#     X_test, y_test = X_full[test_mask], y_full[test_mask]

#     IN_FEATURES = 10

#     # 3. Model Initialization (Fresh weights, zero leakage)
#     if args.model_type == 'GCN':
#         model = GCN_DE_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=128, 
#                              num_classes=3, num_layers=2, use_overlap_logic=args.use_overlap_logic).to(local_device)
#     elif args.model_type == 'DGCNN':
#         model = DGCNN_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=128, 
#                             num_classes=3, num_layers=2).to(local_device)
#     elif args.model_type == 'ADAPTIVE_DGCNN':
#         model = Adaptive_DGCNN(num_nodes=62, in_features=IN_FEATURES, num_classes=3, hidden_dim=128,
#                                num_layers=2).to(local_device)
#     elif args.model_type == 'GRAPH_SAGE':
#         model = GraphSAGE_EEG_Model(
#             in_features=IN_FEATURES, 
#             hidden_dim=64, 
#             aggregator='max'
#         ).to(local_device)

#     # 4. Optimizer Setup (Split for Gamma regularization)
#     gamma_params = [p for n, p in model.named_parameters() if 'static_norm.gamma' in n]
#     other_params = [p for n, p in model.named_parameters() if 'static_norm.gamma' not in n]
    
#     optimizer = optim.Adam([
#         {'params': other_params, 'weight_decay': WEIGHT_DECAY},
#         {'params': gamma_params, 'weight_decay': 1e-2} 
#     ], lr=LEARNING_RATE)

#     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
#     # Alpha weights prioritize Negative (Class 0) and Neutral (Class 1) slightly over Positive
#     alpha_weights = torch.tensor([1.2, 1.1, 0.9]).to(local_device)
#     criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
#     # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.2, 0.8]).to(local_device), label_smoothing=0.1)

#     # 5. Directory Setup
#     run_name = f"Attempt_{run_id}_LOSO_Parallel"
#     results_dir = os.path.join("Results", model_name, run_name, f"Subject_{subject_id}")
#     errors_dir = os.path.join("Errors", model_name, run_name, f"Subject_{subject_id}")
#     params_dir = os.path.join("Params", model_name, run_name, f"Subject_{subject_id}")
    
#     os.makedirs(results_dir, exist_ok=True)
#     os.makedirs(params_dir, exist_ok=True)
#     os.makedirs(errors_dir, exist_ok=True)


#     # Filter subject tensors for split
#     sub_train = sub_full[~test_mask]
#     sub_test = sub_full[test_mask]

#     train_loader = DataLoader(TensorDataset(X_train, y_train, sub_train), 
#                                 batch_size=BATCH_SIZE, 
#                                 shuffle=True,
#                                 num_workers=2,      # <--- INCREASE THIS
#                                 pin_memory=True,    # <--- ENABLE THIS
#                                 persistent_workers=True)
#     test_loader = DataLoader(TensorDataset(X_test, y_test, sub_test), 
#                                 batch_size=BATCH_SIZE, 
#                                 shuffle=False,
#                                 num_workers=2,      # <--- INCREASE THIS
#                                 pin_memory=True,    # <--- ENABLE THIS
#                                 persistent_workers=True)

#     # 6. Training Call
#     train_model_with_interrupt(
#         model=model,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         optimizer=optimizer,
#         criterion=criterion,
#         scheduler=scheduler,
#         epochs=EPOCHS,
#         device=local_device,
#         results_dir=results_dir,
#         params_dir=params_dir,
#         errors_dir=errors_dir,
#         subject_tag = f"Subject_{subject_id}",
#         base_edge_index=base_edge_index.to(local_device),
#         evaluate_fn=evaluate,
#         in_features=IN_FEATURES
#     )


# def main():
#     args = get_args()
    
#     # 1. Data Prep
#     # data_folder = f"Data/ExtractedFeatures_{args.window_size}"
#     data_folder = os.path.join("Data", f"ExtractedFeatures_4s")
#     label_file = os.path.join(data_folder, "label.mat")
    
#     X, y, sessions, subjects, _ = load_de_data(data_folder, label_file)

#     # 3. Setup Runs
#     run_id = get_next_run_id(args.window_size)
#     model_name = f"{args.model_type}_DE_{args.window_size}"

#     if args.mode == 'sub_indep':
#         print("Running Leave-One-Subject-Out with Parallel Processing...")    
#         processes = []
#         subject_list = list(range(1, 16))

#         # 2. Share Tensors in Memory (Crucial for multiprocessing)
#         X_tensor = torch.tensor(X, dtype=torch.float32).share_memory_()
#         y_tensor = torch.tensor(y, dtype=torch.long).share_memory_()
#         sub_tensor = torch.tensor(subjects, dtype=torch.long).share_memory_()
#         base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).share_memory_()
        
#         # Chunk the 15 subjects into groups of MAX_PARALLEL
#         for i in range(0, 15, args.max_parallel):
#             chunk = subject_list[i : i + args.max_parallel]
#             print(f"\n--- Starting Chunk: Subjects {chunk} ---")
            
#             for sub_id in chunk:
#                 # REMOVED shared_results from args
#                 p = mp.Process(target=run_single_subject_fold, 
#                                args=(sub_id, args, X_tensor, y_tensor, 
#                                      sub_tensor, base_edge_index, run_id, 
#                                      model_name))
#                 p.start()
#                 processes.append(p)
            
#             try:
#                 for p in processes: p.join()
#             except KeyboardInterrupt:
#                 print(f"\n\n{'!'*40}")
#                 print(f"MAIN PROCESS INTERRUPTED ON CHUNK {chunk}")
#                 print(f"Waiting for children to save state and exit...")
#                 print(f"{'!'*40}\n")
                
#                 # Wait for children to finish their handle_interrupt() routines
#                 for p in processes: 
#                     if p.is_alive(): p.join()
                
#                 print("\n>>> Moving to next subject chunk... (Press Ctrl+C again quickly to stop completely)")
#                 processes = [] # Clear chunk list
#                 continue 

#             processes = [] # Clear chunk list
        
#         print("\n" + "="*40)
#         print("AGGREGATING GLOBAL RESULTS FROM DISK...")
#         print("="*40 + "\n")
        
#         all_preds = []
#         all_trues = []
#         subject_accuracies = {}
        
#         root_res_dir = os.path.join("Results", model_name, f"Attempt_{run_id}_LOSO_Parallel")
        
#         # Iterate over all subjects to load their saved .npy files
#         for sub_id in subject_list:
#             res_file = os.path.join(root_res_dir, f"Subject_{sub_id}", f"final_test_preds_sub{sub_id}.npy")
            
#             if os.path.exists(res_file):
#                 data = np.load(res_file, allow_pickle=True).item()
#                 preds = data['preds']
#                 trues = data['true']
#                 acc = data['acc']
                
#                 subject_accuracies[sub_id] = acc
#                 all_preds.extend(preds)
#                 all_trues.extend(trues)
#                 print(f"Loaded Subject {sub_id}: {acc:.2f}%")
#             else:
#                 print(f"Warning: Results missing for Subject {sub_id}")
#                 subject_accuracies[sub_id] = 0.0

#         # --- CALCULATE GLOBAL METRICS ---
#         all_preds = np.array(all_preds)
#         all_trues = np.array(all_trues)
        
#         if len(all_preds) == 0:
#             print("\n[ERROR] No predictions were aggregated. Skipping report generation.")
#             print("Did the training processes finish and save their .npy files?")
#         else:
#             global_acc = np.mean(list(subject_accuracies.values()))
#             std_acc = np.std(list(subject_accuracies.values()))
            
#             print(f"\n[LOSO COMPLETE] Mean Acc: {global_acc:.2f}% (+/- {std_acc:.2f}%)")
            
#             # Generate Advanced Reports
#             class_names = ['Negative', 'Neutral', 'Positive']
            
#             try:
#                 cls_report = classification_report(all_trues, all_preds, target_names=class_names)
#                 conf_matrix = confusion_matrix(all_trues, all_preds)

#                 print("\nGlobal Classification Report:")
#                 print(cls_report)
#                 print("\nGlobal Confusion Matrix:")
#                 print(conf_matrix)
                
#                 # Save Global Summary
#                 with open(os.path.join(root_res_dir, "LOSO_Global_Summary.txt"), "w") as f:
#                     f.write(f"Global LOSO Average: {global_acc:.2f}% (+/- {std_acc:.2f}%)\n")
#                     f.write("-" * 30 + "\n")
#                     f.write("Per Subject Accuracies:\n")
#                     for sub, acc in subject_accuracies.items():
#                         f.write(f"Subject {sub}: {acc:.2f}%\n")
#                     f.write("\n" + "="*30 + "\n")
#                     f.write("GLOBAL CLASSIFICATION REPORT:\n")
#                     f.write(cls_report)
#                     f.write("\n" + "="*30 + "\n")
#                     f.write("GLOBAL CONFUSION MATRIX:\n")
#                     f.write(str(conf_matrix))
                    
#                 print(f"Global summary saved to {root_res_dir}")
#             except Exception as e:
#                 print(f"Error generating report: {e}")


#     else:
#         print("  -> Strategy: Session Holdout (Train on S1+S2, Test on S3)")
#         train_mask = (sessions == 1) | (sessions == 2)
#         test_mask = (sessions == 3)

#         # 2. Share Tensors in Memory (Crucial for multiprocessing)
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.long)
#         sub_tensor = torch.tensor(subjects, dtype=torch.long)
#         base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5)

        
#         X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
#         X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
        
#         train_loader = DataLoader(TensorDataset(X_train, y_train), 
#                                   batch_size=64, 
#                                   shuffle=True)
#         test_loader = DataLoader(TensorDataset(X_test, y_test), 
#                                  batch_size=64, 
#                                  shuffle=False)

#         run_name = f"Attempt_{run_id}_Phase2"
#         results_dir = os.path.join("Results", model_name, run_name)
#         params_dir = os.path.join("Params", model_name, run_name)
#         errors_dir = os.path.join("Errors", model_name, run_name)

#         os.makedirs(results_dir, exist_ok=True)
#         os.makedirs(params_dir, exist_ok=True)
#         os.makedirs(errors_dir, exist_ok=True)

#         # --- UPDATE: Input Features = 10 (Mean + Variance Features) ---
#         IN_FEATURES = X_train.shape[-1]
        
#         if args.model_type == 'GCN':
#             print("Initializing Static GCN Model...")
#             model = GCN_DE_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=64, 
#                                 num_classes=3, dropout_rate=0.5, num_layers=3).to(DEVICE)
#         elif args.model_type == 'DGCNN':
#             print("Initializing Dynamic DGCNN Model (Learnable Graph)...")
#             model = DGCNN_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=64, 
#                                 num_classes=3, dropout_rate=0.5).to(DEVICE)
#         elif args.model_type == 'ADAPTIVE_DGCNN':
#             # This is the new model with var_B's Gatekeepers and var_C's Dynamic Brain
#             model = Adaptive_DGCNN(num_nodes=62, in_features=IN_FEATURES, num_classes=3).to(DEVICE)
#             print("Using Adaptive DGCNN (var_D) - The 83% Hybrid Architecture")
#         elif args.model_type == 'GRAPH_SAGE':
#             model = GraphSAGE_EEG_Model(
#                 in_features=IN_FEATURES, 
#                 hidden_dim=128, 
#                 aggregator='max'
#             ).to(DEVICE)
            
#         # --- UPDATED OPTIMIZER: STRONG REGULARIZATION FOR GAMMA ---
#         # We split parameters into "gamma" (needs strong regularization) and "rest".
        
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


#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)
#         # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
#         # Alpha weights prioritize Negative (Class 0) and Neutral (Class 1) slightly over Positive
#         alpha_weights = torch.tensor([1.2, 1.1, 0.9]).to(DEVICE)
#         # criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
#         criterion = nn.CrossEntropyLoss(weight=alpha_weights, label_smoothing=0.1)

#         # --- CLEAN CALL TO MASTER TRAINING FUNCTION ---
#         train_model_with_interrupt(
#             model=model,
#             train_loader=train_loader,
#             test_loader=test_loader,
#             optimizer=optimizer,
#             criterion=criterion,
#             scheduler=scheduler,
#             epochs=EPOCHS,
#             device=DEVICE,
#             # patience=PATIENCE,
#             results_dir=results_dir,
#             params_dir=params_dir,
#             errors_dir=errors_dir,
#             subject_tag="SessionHoldout",
#             base_edge_index=base_edge_index.to(DEVICE),
#             evaluate_fn=evaluate,
#             hyperparams=args,
#             in_features=IN_FEATURES  # Passing the critical 10 features argument
#         )        


# if __name__ == "__main__":
#     # REQUIRED for Windows and CUDA multiprocessing
#     mp.set_start_method('spawn', force=True)
#     main()


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
ROLLING_VAR_WINDOW = 3         # Window size for generating variance features

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
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
    
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
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, 
                              num_workers=4,
                              persistent_workers=True,
                              pin_memory=True,
                              shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, 
                             num_workers=4,
                             persistent_workers=True,
                             pin_memory=True,
                             shuffle=False)

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
        subject_tag=f"Subject_{args.test_subject}" if args.mode == 'sub_indep' else "SessionHoldout",
        base_edge_index=base_edge_index.to(DEVICE),
        evaluate_fn=evaluate,
        in_features=in_features
    )

if __name__ == "__main__":
    main()