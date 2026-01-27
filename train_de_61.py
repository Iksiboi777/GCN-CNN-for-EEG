# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
# import argparse
# import joblib
# import torch.multiprocessing as mp
# import json

# from Models.var_B import GCN_DE_Model
# from Models.var_C import DGCNN_Model
# from Models.var_D import Adaptive_DGCNN
# from Models.graph_construction import get_knn_adjacency_matrix
# from utils.training_utils import train_model_with_interrupt, evaluate
# from utils.focal_loss import FocalLoss

# # --- Configuration ---
# LOCS_FILE = "utils/channel_62_pos.locs"
# BATCH_SIZE = 256 # Slightly smaller batch for Phase 2 stability
# EPOCHS = 100
# LEARNING_RATE = 0.0004
# WEIGHT_DECAY = 5e-3
# GAMMA_REG = 1e-2
# CONFIG_FILE = "run_config.json"

# # --- Helper: Get Run ID ---
# def get_next_run_id(model_type):
#     if not os.path.exists(CONFIG_FILE):
#         with open(CONFIG_FILE, 'w') as f: json.dump({}, f)
#     with open(CONFIG_FILE, 'r') as f:
#         try: config = json.load(f)
#         except: config = {}
#     key = f"run_counter_{model_type}"
#     next_id = config.get(key, 0) + 1
#     config[key] = next_id
#     with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=4)
#     return next_id


# # --- New Dataset Class for Memmap ---
# class SEEDMemmapDataset(Dataset):
#     def __init__(self, data_dir):
#         super().__init__()
#         self.shape = joblib.load(os.path.join(data_dir, "X_shape.pkl"))
#         self.X = np.memmap(os.path.join(data_dir, "X_custom.dat"), dtype='float32', mode='r', shape=self.shape)
#         self.y = np.load(os.path.join(data_dir, "y_labels.npy"))
#         self.subjects = np.load(os.path.join(data_dir, "subject_ids.npy"))
#         if np.min(self.y) == 1: self.y = self.y - 1 
        
#         # Adjust labels to 0-indexed if they are 1-3
#         if np.min(self.y) == 1:
#             self.y = self.y - 1 

#     def __len__(self):
#         return self.shape[0]

#     def __getitem__(self, idx):
#         # Return tensors: (62, 11), Label, SubjectID
#         x_sample = torch.tensor(self.X[idx], dtype=torch.float32)
#         y_sample = torch.tensor(self.y[idx], dtype=torch.long)
#         sub_sample = torch.tensor(self.subjects[idx], dtype=torch.long)
#         return x_sample, y_sample, sub_sample

# # --- Filtered Wrapper for Subsets ---
# class SubsetMemmapDataset(Dataset):
#     def __init__(self, parent_dataset, indices):
#         self.parent = parent_dataset
#         self.indices = indices    
#     def __len__(self):
#         return len(self.indices)    
#     def __getitem__(self, idx):
#         real_idx = self.indices[idx]
#         return self.parent[real_idx]

# # --- Worker Function ---
# def run_fold(subject_id, dataset, base_edge_index, args, run_id, model_name_str):
#     subject_id = int(subject_id)

#     # Setup Device
#     num_gpus = torch.cuda.device_count()
#     device = torch.device(f"cuda:{subject_id % num_gpus}" if num_gpus > 0 else "cpu")
#     print(f"Subject {subject_id} running on {device}")
    
#     all_subjects = dataset.subjects
#     test_indices = np.where(all_subjects == subject_id)[0]
#     train_indices = np.where(all_subjects != subject_id)[0]
#     run_name = f"Attempt_{run_id}_LOSO_Parallel"

#     # Init Loaders
#     train_ds = SubsetMemmapDataset(dataset, train_indices)
#     test_ds = SubsetMemmapDataset(dataset, test_indices)
    
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
#                               num_workers=0, pin_memory=True) # Workers=0 usually safer with Memmap on Windows
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
#     # Init Model
#     IN_FEAT = 11

#     # --- MODEL & HYPERPARAM SETUP ---
#     model_config = {
#         "LR": LEARNING_RATE,
#         "BATCH_SIZE": BATCH_SIZE,
#         "WEIGHT_DECAY": WEIGHT_DECAY,
#         "GAMMA_REG": GAMMA_REG,
#         "EPOCHS": EPOCHS,
#         "num_layers": 2,
#         "hidden_dim": 128,
#         "num_classes": 3,
#         "Loss_Function": "FocalLoss",
#         "Class_Weights": "[1.2, 1.0, 1.0]",
#         # Feature Flags (Boolean)
#         "Use_Gated_FAS": 0,
#         "Use_Regular_FAS": 0,
#         "Use_Adaptive_Subject_Bias": 1,
#         "Use_Subject_Specific_Norm": 1,
#         "Use_BatchNorm": 0, # Replaced by SSBN
#         "Use_AGLI": 1, 
#         "Use_SE_Block": 1
#     }

#     if args.model_type == 'GCN':
#         model = GCN_DE_Model(num_nodes=62, in_features=IN_FEAT, hidden_dim=128, 
#                              num_classes=3, num_subjects=15, num_layers=2).to(device)
#         model_config["Use_Gated_FAS"] = 1 # Assuming GCN_DE_Model now has the FAS Head
#     elif args.model_type == 'DGCNN':
#         model = Adaptive_DGCNN(num_nodes=62, in_features=IN_FEAT, hidden_dim=128,
#                                num_classes=3, num_subjects=15, num_layers=2).to(device)
#         # Adaptive DGCNN usually has SE and AGLI but not necessarily Gated FAS head unless added
    
#     # --- OPTIMIZER WITH GAMMA SPLIT ---
#     gamma_params = []
#     other_params = []
#     for name, param in model.named_parameters():
#         if 'gamma' in name or 'agli' in name:
#             gamma_params.append(param)
#         else:
#             other_params.append(param)
            
#     optimizer = optim.AdamW([
#         {'params': other_params, 'weight_decay': WEIGHT_DECAY}, 
#         {'params': gamma_params, 'weight_decay': GAMMA_REG} 
#     ], lr=LEARNING_RATE)
    
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
#     criterion = FocalLoss(alpha=torch.tensor([1.2, 1.0, 1.0]).to(device), gamma=2.0)
   
#     # 6. Directories (Exact Match to train_de.py structure)
#     results_dir = os.path.join("Results", model_name_str, run_name, f"Subject_{subject_id}")
#     params_dir = os.path.join("Params", model_name_str, run_name, f"Subject_{subject_id}")
#     errors_dir = os.path.join("Errors", model_name_str, run_name, f"Subject_{subject_id}")

#     # 7. Execute Training
#     train_model_with_interrupt(
#         model, train_loader, test_loader, optimizer, criterion, scheduler, 
#         EPOCHS, device, results_dir, params_dir, errors_dir, f"Sub {subject_id}",
#         base_edge_index.to(device), evaluate, hyperparams=model_config, in_features=IN_FEAT
#     )

#     # 8. Final Save for Aggregation
#     _, acc, preds, trues = evaluate(model, test_loader, base_edge_index.to(device), criterion, device, IN_FEAT, return_preds=True)
#     res_file = os.path.join(results_dir, f"final_test_preds_sub{subject_id}.npy")
#     np.save(res_file, {'preds': preds, 'true': trues, 'acc': acc})
#     print(f"Subject {subject_id} finished: {acc:.2f}%")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_name', type=str, default='Custom_2s_25overlap_FAST2')
#     parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'DGCNN'])
#     parser.add_argument('--mode', type=str, default='sub_indep', choices=['sub_indep', 'sub_deš'])
#     parser.add_argument('--max_parallel', type=int, default=3, help='Max parallel processes')
#     args = parser.parse_args()
    
#     data_path = f"Data/{args.dataset_name}"
#     print(f"Loading Memmap from {data_path}...")
#     full_dataset = SEEDMemmapDataset(data_path)
#     all_subjects = full_dataset.subjects
#     base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5)
    
#     # --- GLOBAL RUN ID (Generated Once) ---
#     run_id = get_next_run_id(args.model_type)
#     model_name_str = f"{args.model_type}_Phase2"
    
#     print(f"--- STARTING RUN: {model_name_str} | Attempt {run_id} | Mode: {args.mode} ---")
    
#     unique_subs = np.unique(all_subjects)
#     processes = []
    
#     for i in range(0, len(unique_subs), args.max_parallel):
#         chunk = unique_subs[i : i + args.max_parallel]
#         print(f"\nProcessing Chunk: {chunk}")
        
#         for sub_id in chunk:
#             p = mp.Process(target=run_fold, args=(
#                 sub_id, full_dataset, base_edge_index, args, run_id, model_name_str
#             ))
#             p.start()
#             processes.append(p)
            
#         for p in processes: p.join()
#         processes = []

#     # --- GLOBAL AGGREGATION (Restored from your request) ---
#     print("\n" + "="*40)
#     print("AGGREGATING GLOBAL RESULTS...")
#     print("="*40 + "\n")
    
#     all_preds = []
#     all_trues = []
#     subject_accuracies = {}
    
#     run_name = f"Attempt_{run_id}_LOSO_Parallel" if args.mode == 'sub_indep' else f"Attempt_{run_id}_SessionHoldout_Parallel"
#     root_res_dir = os.path.join("Results", model_name_str, run_name)
    
#     for sub_id in unique_subs:
#         sub_id = int(sub_id)
#         res_file = os.path.join(root_res_dir, f"Subject_{sub_id}", f"final_test_preds_sub{sub_id}.npy")
#         if os.path.exists(res_file):
#             data = np.load(res_file, allow_pickle=True).item()
#             subject_accuracies[sub_id] = data['acc']
#             all_preds.extend(data['preds'])
#             all_trues.extend(data['true'])
#             print(f"Subject {sub_id}: {data['acc']:.2f}%")
#         else:
#             print(f"Warning: Missing results for Subject {sub_id}")

#     if len(all_preds) > 0:
#         global_acc = np.mean(list(subject_accuracies.values()))
#         print(f"\nGlobal Mean Acc: {global_acc:.2f}%")
        
#         cls_report = classification_report(all_trues, all_preds, target_names=['Negative', 'Neutral', 'Positive'])
#         print(cls_report)
        
#         with open(os.path.join(root_res_dir, "Global_Summary.txt"), "w") as f:
#             f.write(cls_report)
#             f.write(f"\nMean Acc: {global_acc:.2f}%")
            

# if __name__ == "__main__":
#     mp.set_start_method('spawn', force=True)
#     main()





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np
import scipy.io
import os
import argparse
import sys
import json
import torch.multiprocessing as mp

from Models.var_B import GCN_DE_Model
from Models.var_C import DGCNN_Model
from Models.var_D import Adaptive_DGCNN
from Models.graph_construction import get_knn_adjacency_matrix
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.focal_loss import FocalLoss

# --- Configuration Constants ---
LOCS_FILE = "utils/channel_62_pos.locs"
CONFIG_FILE = "run_config.json"
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 5e-3
GAMMA_REG = 1e-2

def get_next_run_id(model_type):
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f: json.dump({}, f)
    with open(CONFIG_FILE, 'r') as f:
        try: config = json.load(f)
        except: config = {}
    key = f"run_counter_{model_type}"
    next_id = config.get(key, 0) + 1
    config[key] = next_id
    with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=4)
    return next_id

# -----------------------------------------------------------------------------
# 1. LOAD DATA (DEBUGGING VERSION)
# -----------------------------------------------------------------------------
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
    sess_list = []
    sub_list = []
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

        
    for subj_id in sorted(subject_files.keys()):
        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            # --- FIX: Dynamic Key Discovery ---
            # Find all keys that look like EEG or DE data
            all_keys = [k for k in mat.keys() if ('eeg' in k.lower() or 'de_lds' in k.lower()) and not k.startswith('_')]
            
            # Sort keys numerically based on the number at the end (e.g., 'djc_de_LDS1' -> 1)
            # This handles prefixes like 'djc_' automatically
            def extract_trial_num(k):
                nums = ''.join(filter(str.isdigit, k))
                return int(nums) if nums else 999
            
            all_keys.sort(key=extract_trial_num)
            
            # We expect exactly 15 trials. If fewer, process what we have.
            for i, key in enumerate(all_keys):
                if i >= 15: break # Only take first 15 mapped to labels
                
                data = mat[key] # Raw shape: (N, 62, 11) usually
                
                # --- ROBUST SHAPE CORRECTION ---
                # Goal: (N_samples, 62, 11)
                shape = data.shape
                
                # Case 1: Already correct (N, 62, 11)
                # We assume N is usually > 62. If shape[1]==62 and shape[2]==11, it is likely correct.
                if len(shape) == 3 and shape[1] == 62 and shape[2] == 11:
                    pass 
                
                # Case 2: (62, N, 11) -> Transpose to (N, 62, 11)
                elif len(shape) == 3 and shape[0] == 62 and shape[2] == 11:
                    data = np.transpose(data, (1, 0, 2))
                    
                # Case 3: (62, 11, N) -> Transpose to (N, 62, 11) 
                elif len(shape) == 3 and shape[0] == 62 and shape[1] == 11:
                    data = np.transpose(data, (2, 0, 1))

                # Case 4: (11, 62, N) -> Transpose to (N, 62, 11)
                elif len(shape) == 3 and shape[0] == 11 and shape[1] == 62:
                    data = np.transpose(data, (2, 1, 0))

                # Final Validity Check
                if data.shape[1] != 62 or data.shape[2] != 11:
                     print(f"Warning: {fname} key {key} has weird shape {shape}, skipping.")
                     continue

                num_samples = data.shape[0]                
                X_list.append(data)
                y_list.append(np.full(num_samples, mapped_labels[i])) # Label matches trial index
                sess_list.append(np.full(num_samples, session_id))
                sub_list.append(np.full(num_samples, subj_id))

    if not X_list: 
        raise ValueError("No valid data found in folder! Check if folder path is correct and files contain 'de_LDS' or 'eeg' keys.")
        
    X_all = np.concatenate(X_list, axis=0) 
    y_all = np.concatenate(y_list, axis=0)
    sub_all = np.concatenate(sub_list, axis=0)
    sess_all = np.concatenate(sess_list, axis=0)
    
    print(f"SUCCESS: Loaded Data Shape: {X_all.shape}")
    print(f"Subjects: {len(np.unique(sub_all))}, Sessions Found: {np.unique(sess_all)}")
    
    return X_all, y_all, sub_all, sess_all

# -----------------------------------------------------------------------------
# 2. WORKER FUNCTION (Matches run_single_subject_fold from train_de.py)
# -----------------------------------------------------------------------------
def run_fold(subject_id, args, X_full, y_full, sub_full, sess_full, base_edge_index, run_id, model_name_str):
    subject_id = int(subject_id)
    
    # 1. Device Setup
    num_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{subject_id % num_gpus}" if num_gpus > 0 else "cpu")
    print(f"Subject {subject_id} running on {device}")

    # 2. Train/Test Split logic
    
    if args.mode == 'sub_indep':
        # LOSO: Test on Subject X, Train on Everyone Else
        test_mask = (sub_full == subject_id)
        train_mask = ~test_mask
        
        path_suffix = f"Subject_{subject_id}"
        run_name = f"Attempt_{run_id}_LOSO_Parallel"
        
    elif args.mode == 'sub_dep':
        # Session Holdout: Train S1+S2, Test S3 (ONLY for this subject)
        # Note: In parallel Sub-Dep, "run_fold(subject_id)" implies models are individualized PER SUBJECT.
        # We discard all other subjects' data.
        
        subj_mask = (sub_full == subject_id)
        
        # Of this subject's data, which is test (Sess 3) and which is train (Sess 1,2)?
        is_session_3 = (sess_full == 3)
        
        test_mask = (subj_mask & is_session_3)
        train_mask = (subj_mask & ~is_session_3)
        
        path_suffix = f"Subject_{subject_id}"
        run_name = f"Attempt_{run_id}_SessionHoldout_Parallel"

    X_train, y_train, sub_train = X_full[train_mask], y_full[train_mask], sub_full[train_mask]
    X_test, y_test, sub_test = X_full[test_mask], y_full[test_mask], sub_full[test_mask]
    print(f"Subject {subject_id} | Train Samples: {len(y_train)} | Test Samples: {len(y_test)}")

    # 3. Create Loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train, sub_train), 
                              batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test, sub_test), 
                             batch_size=BATCH_SIZE, shuffle=False)
    
    IN_FEAT = 11

    # 4. Model Setup
    model_config = {
        "LR": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, "WEIGHT_DECAY": WEIGHT_DECAY,
        "GAMMA_REG": GAMMA_REG, "EPOCHS": EPOCHS, "Mode": args.mode,
        "Structure": "Batch-Parallel",
        "Use_Gated_FAS": 1 if args.model_type == 'GCN' else 0,
        "Use_Adaptive_Subject_Bias": 1, "Use_Subject_Specific_Norm": 1,
        "Use_AGLI": 1, "Use_SE_Block": 1
    }

    if args.model_type == 'GCN':
        model = GCN_DE_Model(num_nodes=62, in_features=IN_FEAT, hidden_dim=128, 
                             num_classes=3, num_subjects=15, num_layers=2).to(device)
    elif args.model_type == 'DGCNN':
        model = Adaptive_DGCNN(num_nodes=62, in_features=IN_FEAT, hidden_dim=128,
                               num_classes=3, num_subjects=15, num_layers=2).to(device)
    
    # 5. Optimizer
    gamma_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gamma' in name or 'agli' in name:
            gamma_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': other_params, 'weight_decay': WEIGHT_DECAY}, 
        {'params': gamma_params, 'weight_decay': GAMMA_REG} 
    ], lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(alpha=torch.tensor([1.2, 1.0, 1.0]).to(device), gamma=2.0)

    # 6. Directories
    results_dir = os.path.join("Results", model_name_str, run_name, path_suffix)
    params_dir = os.path.join("Params", model_name_str, run_name, path_suffix)
    errors_dir = os.path.join("Errors", model_name_str, run_name, path_suffix)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)

    # 7. Execute Training
    train_model_with_interrupt(
        model, train_loader, test_loader, optimizer, criterion, scheduler, 
        EPOCHS, device, results_dir, params_dir, errors_dir, f"Sub {subject_id}",
        base_edge_index.to(device), evaluate, hyperparams=model_config, in_features=IN_FEAT
    )

    # 8. Final Evaluation & Save
    _, acc, preds, trues = evaluate(model, test_loader, base_edge_index.to(device), criterion, device, IN_FEAT, return_preds=True)
    res_file = os.path.join(results_dir, f"final_test_preds_sub{subject_id}.npy")
    np.save(res_file, {'preds': preds, 'true': trues, 'acc': acc})
    print(f"Subject {subject_id} finished: {acc:.2f}%")

# -----------------------------------------------------------------------------
# 3. MAIN (Orchestration)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ExtractedFeatures_2s_25overlap')
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'DGCNN'])
    parser.add_argument('--mode', type=str, default='sub_indep', choices=['sub_indep', 'sub_dep'])
    parser.add_argument('--max_parallel', type=int, default=4)
    args = parser.parse_args()
    
    data_path = f"Data/{args.dataset_name}"
    label_file = f"Data/{args.dataset_name}/label.mat" # Assuming built by script
    
    # 1. Load Everything
    X, y, subjects, sessions = load_de_data(data_path, label_file)
    
    # 2. Share Memory for Multiprocessing
    X_tensor = torch.tensor(X, dtype=torch.float32).share_memory_()
    y_tensor = torch.tensor(y, dtype=torch.long).share_memory_()
    sub_tensor = torch.tensor(subjects, dtype=torch.long).share_memory_()
    sess_tensor = torch.tensor(sessions, dtype=torch.long).share_memory_()  # Not used in this version, but could be added if needed
    base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).share_memory_()

    # 3. Run Setup
    run_id = get_next_run_id(args.model_type)
    model_name_str = f"{args.model_type}_Phase2"
    
    print(f"--- STARTING RUN: {model_name_str} | Attempt {run_id} | Mode: {args.mode} ---")
    
    unique_subs = np.unique(subjects)
    processes = []
    
    # 4. Processing Loop
    for i in range(0, len(unique_subs), args.max_parallel):
        chunk = unique_subs[i : i + args.max_parallel]
        print(f"\nProcessing Chunk: {chunk}")
        
        for sub_id in chunk:
            p = mp.Process(target=run_fold, args=(
                sub_id, args, X_tensor, y_tensor, sub_tensor, sess_tensor, base_edge_index, run_id, model_name_str
            ))
            p.start()
            processes.append(p)
            
        for p in processes: p.join()
        processes = []

    # 5. Global Aggregation
    print("\n" + "="*40 + "\nAGGREGATING RESULTS...\n" + "="*40)
    
    run_name = f"Attempt_{run_id}_LOSO_Parallel"
    root_res_dir = os.path.join("Results", model_name_str, run_name)
    os.makedirs(root_res_dir, exist_ok=True)
    
    all_preds, all_trues, subject_accuracies = [], [], {}
    
    for sub_id in unique_subs:
        sub_id = int(sub_id)
        res_file = os.path.join(root_res_dir, f"Subject_{sub_id}", f"final_test_preds_sub{sub_id}.npy")
        if os.path.exists(res_file):
            data = np.load(res_file, allow_pickle=True).item()
            subject_accuracies[sub_id] = data['acc']
            all_preds.extend(data['preds'])
            all_trues.extend(data['true'])
            print(f"Subject {sub_id}: {data['acc']:.2f}%")
        else:
            print(f"Warning: Missing results for Subject {sub_id}")

    if len(all_preds) > 0:
        global_acc = np.mean(list(subject_accuracies.values()))
        print(f"\nGlobal Mean Acc: {global_acc:.2f}%")
        with open(os.path.join(root_res_dir, "Global_Summary.txt"), "w") as f:
            f.write(classification_report(all_trues, all_preds, target_names=['Negative', 'Neutral', 'Positive']))
            f.write(f"\nMean Acc: {global_acc:.2f}%")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()