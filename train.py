# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np
# import os
# import argparse
# import time
# import copy
# from sklearn.metrics import classification_report, confusion_matrix, f1_score

# from Models.var_A import CNNGCNModel
# from Models.graph_construction import get_knn_adjacency_matrix

# # --- Configuration ---
# DATA_FOLDER = "Data/Raw_Data_w_Bands"
# OUTPUT_FOLDER = "Models/Trained_Models_w_Bands"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LOCS_FILE = "channel_62_pos.locs"
# BATCH_SIZE = 32
# EPOCHS = 50
# LEARNING_RATE = 0.0005 # Kept low
# WEIGHT_DECAY = 1e-4
# PATIENCE = 15 
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_args():
#     parser = argparse.ArgumentParser(description="Train GCN-CNN for EEG Emotion Recognition")
#     parser.add_argument('--mode', type=str, required=True, choices=['sub_dep', 'sub_indep'],
#                         help="Training mode: 'subject_dependent' (Session split) or 'subject_independent' (LOSO)")
#     parser.add_argument('--test_subject', type=int, default=1, 
#                         help="ID of the subject to leave out for testing (only for subject_independent mode)")
#     parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
#     parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
#     parser.add_argument('--band', type=str, default='standard', choices=['standard', 'gamma'], 
#                         help="Frequency band to use: 'standard' (1-49Hz) or 'gamma' (50-75Hz)")
#     return parser.parse_args()

# def main():
#     args = get_args()
#     num_workers = 0 
#     print(f"Using device: {DEVICE} with {num_workers} workers.")
#     print(f"Mode: {args.mode} | Band: {args.band}")

#     # 1. Load Data
#     print(f"Loading data for band: {args.band}...")
#     X = np.load(os.path.join(DATA_FOLDER, f"X_raw_{args.band}.npy")) 
#     y = np.load(os.path.join(DATA_FOLDER, f"y_labels_{args.band}.npy"))
#     sessions = np.load(os.path.join(DATA_FOLDER, f"sessions_{args.band}.npy"))
#     subjects = np.load(os.path.join(DATA_FOLDER, f"subjects_{args.band}.npy"))
    
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.long)
    
#     # 2. Define Split Strategy
#     if args.mode == 'sub_dep':
#         print("Splitting data (Train: Sess 1+2, Test: Sess 3)...")
#         train_mask = (sessions == 1) | (sessions == 2)
#         test_mask = (sessions == 3)
#     elif args.mode == 'sub_indep':
#         print(f"Splitting data (LOSO): Leaving out Subject {args.test_subject}...")
#         train_mask = (subjects != args.test_subject)
#         test_mask = (subjects == args.test_subject)
        
#     X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
#     X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
    
#     print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

#     # 3. Construct Graph
#     print("Constructing Graph...")
#     base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)
    
#     # 4. Initialize Model
#     model = CNNGCNModel(num_nodes=62, time_steps=400).to(DEVICE)

#     # Verify output dimension
#     print("Verifying model output shape...")
#     dummy_input = torch.randn(2, 62, 400).to(DEVICE)
#     dummy_batch = torch.arange(2, device=DEVICE).repeat_interleave(62)
#     offsets = (torch.arange(2, device=DEVICE) * 62).view(-1, 1, 1)
#     dummy_edge_expanded = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

#     with torch.no_grad():
#         dummy_out = model(dummy_input, dummy_edge_expanded, dummy_batch)
#     print(f"Model Output Shape: {dummy_out.shape} (Should be [2, 3])")

#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)
    
#     # NUCLEAR OPTION: Force the model to care about all classes equally or slightly boost class 2
#     # Even if data is balanced, this helps break the "ignore class 2" habit.
#     # Weights: [1.0, 1.0, 1.2] -> 20% higher penalty for missing Class 2 (Positive)
#     class_weights = torch.tensor([1.0, 1.0, 1.2]).to(DEVICE) 
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
    
#     # 5. Training Loop
#     print("Starting Training...")
#     start_time = time.time()

#     best_val_loss = float('inf')
#     patience_counter = 0
#     best_model_wts = copy.deepcopy(model.state_dict())

#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

#     for epoch in range(args.epochs):
#         epoch_start = time.time()

#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         for batch_X, batch_y in train_loader:
#             batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
#             curr_batch_size = batch_X.size(0)
#             batch_idx = torch.arange(curr_batch_size, device=DEVICE).repeat_interleave(62)
            
#             offsets = (torch.arange(curr_batch_size, device=DEVICE) * 62).view(-1, 1, 1)
#             edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
#             optimizer.zero_grad()
#             outputs = model(batch_X, edge_index, batch_idx)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
            
#         train_acc = 100 * correct / total
#         avg_train_loss = total_loss / len(train_loader)
        
#         # Evaluate with detailed report every epoch
#         val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, base_edge_index, criterion, return_preds=True)
        
#         epoch_duration = time.time() - epoch_start
#         current_lr = optimizer.param_groups[0]['lr']
        
#         history['train_loss'].append(avg_train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
#         history['lr'].append(current_lr)
        
#         print(f"Epoch [{epoch+1}/{args.epochs}] ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
#         # PRINT REPORT EVERY EPOCH to see if Class 2 is being predicted
#         print(classification_report(val_labels, val_preds, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))
        
#         scheduler.step(val_loss)
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_wts = copy.deepcopy(model.state_dict())
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             print(f"EarlyStopping counter: {patience_counter} out of {PATIENCE}")
            
#         if patience_counter >= PATIENCE:
#             print("Early stopping triggered.")
#             break

#     # --- SAVE & EVALUATE BEST MODEL ---
#     print("\nLoading best model weights...")
#     model.load_state_dict(best_model_wts)
    
#     # Save to disk
#     model_path = os.path.join(OUTPUT_FOLDER, f"best_model_{args.mode}_{args.band}.pth")
#     torch.save(model.state_dict(), model_path)
#     print(f"Best model saved to {model_path}")

#     # Detailed Evaluation
#     print("\n--- Final Evaluation on Test Set ---")
#     test_loss, test_acc, preds, true_labels = evaluate(model, test_loader, base_edge_index, criterion, return_preds=True)
    
#     print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
#     print("\nClassification Report:")
#     print(classification_report(true_labels, preds, target_names=['Negative', 'Neutral', 'Positive']))
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(true_labels, preds))

#     # Save History
#     history_path = os.path.join(OUTPUT_FOLDER, f"training_history_{args.mode}_{args.band}.npy")
#     np.save(history_path, history)
#     print(f"History saved to {history_path}")
    
#     # Save Detailed Predictions for Debugging
#     debug_data = {
#         'y_true': true_labels,
#         'y_pred': preds
#     }
#     np.save(os.path.join(OUTPUT_FOLDER, f"debug_predictions_{args.mode}_{args.band}.npy"), debug_data)
#     print(f"Debug predictions saved to debug_predictions_{args.mode}_{args.band}.npy")
    
#     print(f"Total Time: {time.time() - start_time:.2f}s")


# def evaluate(model, loader, base_edge_index, criterion, return_preds=False):
#     model.eval()
#     correct = 0
#     total = 0
#     val_loss = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch_X, batch_y in loader:
#             batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
#             curr_batch_size = batch_X.size(0)
#             batch_idx = torch.arange(curr_batch_size, device=DEVICE).repeat_interleave(62)
            
#             # Expand edge_index for the batch
#             offsets = (torch.arange(curr_batch_size, device=DEVICE) * 62).view(-1, 1, 1)
#             edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            

#             outputs = model(batch_X, edge_index, batch_idx)
#             loss = criterion(outputs, batch_y)
#             val_loss += loss.item()
            
#             _, predicted = torch.max(outputs.data, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
            
#             if return_preds:
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(batch_y.cpu().numpy())
            
#     acc = 100 * correct / total
#     avg_loss = val_loss / len(loader)
    
#     if return_preds:
#         return avg_loss, acc, all_preds, all_labels
#     return avg_loss, acc

# if __name__ == "__main__":
#     main()



from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import scipy.io
import os
import argparse
import sys
import json
import torch.multiprocessing as mp

from Models.var_A import Attempt61_CNNGCN
from Models.graph_construction import get_knn_adjacency_matrix
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.focal_loss import FocalLoss

class ScaledSharedDataset(torch.utils.data.Dataset):
    def __init__(self, X_tensor, sub_tensor, y_tensor, indices):
        self.X = X_tensor
        self.sub = sub_tensor
        self.y = y_tensor
        self.indices = indices
        
        # We don't scale everything. We just pre-calculate 
        # a rough global median/std if needed, or scale per-sample.
        # For Raw Voltage, per-sample Robust Scaling is actually safer.

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.X[real_idx] # (62, 400) - No copy made yet
        y = self.y[real_idx]
        sub = self.sub[real_idx]
        
        # Scale only this ONE sample (62, 400)
        # Robust scale: (x - median) / interquartile_range
        median = x.median()
        q75, q25 = torch.quantile(x, 0.75), torch.quantile(x, 0.25)
        iqr = q75 - q25 + 1e-6
        x_scaled = (x - median) / iqr

        
        return x_scaled, y, sub
    

# --- Configuration ---
LOCS_FILE = "utils/channel_62_pos.locs"
CONFIG_FILE = "run_config.json"
BATCH_SIZE = 32 # Smaller batch size for Raw Data (memory heavy)
EPOCHS = 60
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
PATIENCE = 10


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
# 1. LOAD DATA (Directory Scanner for Raw Data)
# -----------------------------------------------------------------------------
def load_raw_data(data_folder, label_file):
    print(f"Loading RAW SEGMENTED data from {data_folder}...")
    
    # 1. Load Labels
    try:
        label_mat = scipy.io.loadmat(label_file)
        if 'label' in label_mat:
            trial_labels = label_mat['label'][0]
        else:
            # Fallback
            trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1] 
    except Exception as e:
        print(f"Label file issue: {e}. Using standard SEED labels.")
        trial_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = [label_map.get(l, 1) for l in trial_labels] 

    X_list = []
    y_list = []
    sub_list = []
    sess_list = []

    files = sorted([f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat'])
    
    # Group files by Subject
    files_per_sub = {}
    for fname in files:
        try:
            sid = int(fname.split('_')[0])
            if sid not in files_per_sub: files_per_sub[sid] = []
            files_per_sub[sid].append(fname)
        except: continue
    
    print(f"Found {len(files_per_sub)} subjects.")

    for sid in sorted(files_per_sub.keys()):
        # Sort by session date to ensure Sess 1, 2, 3 order
        sub_files = sorted(files_per_sub[sid], key=lambda x: x.split('_')[1])
        
        for sess_idx, fname in enumerate(sub_files):
            session_id = sess_idx + 1
            file_path = os.path.join(data_folder, fname)
            
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            
            # Find relevant keys
            keys = [k for k in mat.keys() if 'eeg' in k.lower() or 'de' in k.lower()]
            keys.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 999)
            
            for i, key in enumerate(keys):
                if i >= len(mapped_labels): break 
                
                data = mat[key] 
                
                # --- CRITICAL: Enforce Float32 Immediately ---
                # Raw data is huge. Float64 will crash your RAM.
                if data.dtype != np.float32:
                    data = data.astype(np.float32)

                # --- Shape Check: Expect (N, 62, 400) ---
                if data.ndim == 3:
                    if data.shape[1] == 62 and data.shape[2] == 400: pass
                    elif data.shape[0] == 62 and data.shape[2] == 400: 
                        data = np.transpose(data, (1, 0, 2))
                    elif data.shape[0] == 62 and data.shape[1] == 400:
                         # (62, 400, N) -> (N, 62, 400)
                         data = np.transpose(data, (2, 0, 1))
                
                if data.shape[1] == 62 and data.shape[2] == 400:
                    num_samples = data.shape[0]
                    X_list.append(data)
                    y_list.append(np.full(num_samples, mapped_labels[i]))
                    sub_list.append(np.full(num_samples, sid))
                    sess_list.append(np.full(num_samples, session_id))
    
    if not X_list: raise ValueError("No valid data found!")
        
    X_all = np.concatenate(X_list, axis=0) 
    y_all = np.concatenate(y_list, axis=0)
    sub_all = np.concatenate(sub_list, axis=0)
    sess_all = np.concatenate(sess_list, axis=0)
    
    print(f"Loaded Raw Data: {X_all.shape} | {X_all.nbytes / 1e9:.2f} GB | Dtype: {X_all.dtype}")
    return X_all, y_all, sub_all, sess_all

# -----------------------------------------------------------------------------
# 2. WORKER FUNCTION (Handles Both Sub-Dep and Sub-Indep Logic)
# -----------------------------------------------------------------------------
def run_fold(subject_id, args, X_full, y_full, sub_full, sess_full, base_edge_index, run_id):
    subject_id = int(subject_id)
    
    # 1. Device Setup
    num_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{subject_id % num_gpus}" if num_gpus > 0 else "cpu")
    print(f"Subject {subject_id} running on {device}")

    # 2. Split Strategy
    if args.mode == 'sub_indep':
        # LOSO: Train = All Subs except X, Test = Subject X
        test_mask = (sub_full == subject_id)
        train_mask = (sub_full != subject_id)
        
        run_name = f"Attempt_{run_id}_LOSO_Raw"
        path_suffix = f"Subject_{subject_id}"
        
    elif args.mode == 'sub_dep':
        # Personalized: Use ONLY Subject X data.
        # Train = Sess 1+2, Test = Sess 3
        subj_mask = (sub_full == subject_id)
        test_mask = subj_mask & (sess_full == 3)
        train_mask = subj_mask & (sess_full != 3) # Sess 1 & 2
        
        run_name = f"Attempt_{run_id}_SessionHoldout_Raw"
        path_suffix = f"Subject_{subject_id}"

    # Get indices for train/test instead of slicing the actual data
    train_indices = torch.where(train_mask)[0]
    test_indices = torch.where(test_mask)[0]
    # Initialize the "Smart" Loaders (No 8GB copies!)
    train_ds = ScaledSharedDataset(X_full, sub_full, y_full, train_indices)
    test_ds = ScaledSharedDataset(X_full, sub_full, y_full, test_indices)

    # 3. Loaders (Pin Memory is vital for Raw data)
    train_loader = DataLoader(train_ds, 
                              batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, 
                             batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=0, pin_memory=True)
    
    # 4. Model Setup
    # Model Config for Logging
    model_config = {
        "LR": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
        "WEIGHT_DECAY": WEIGHT_DECAY, "EPOCHS": EPOCHS, 
        "Mode": args.mode, "Model": "CNNGCN_Raw", 
        "Pattern": "2s_25overlap",
        "Test_Subject": subject_id
    }

    model = Attempt61_CNNGCN(num_nodes=62, time_steps=400).to(device)
    
    # 5. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Slight weighting to help class balance if needed
    criterion = FocalLoss(alpha=torch.tensor([1.2, 1.0, 1.0]).to(device), gamma=2.0)

    # 6. Directories
    results_dir = os.path.join("Results", "CNNGCN_Raw", run_name, path_suffix)
    params_dir = os.path.join("Params", "CNNGCN_Raw", run_name, path_suffix)
    errors_dir = os.path.join("Errors", "CNNGCN_Raw", run_name, path_suffix)

    # 7. Execute Training matches train_de.py workflow
    train_model_with_interrupt(
        model, train_loader, test_loader, optimizer, criterion, scheduler, 
        EPOCHS, device, results_dir, params_dir, errors_dir, f"Sub {subject_id}",
        base_edge_index.to(device), evaluate, hyperparams=model_config, 
        in_features=400 # Just for logging, not used by raw model
    )

    # 8. Final Save
    _, acc, preds, trues = evaluate(model, test_loader, base_edge_index.to(device), criterion, device, 0, return_preds=True)
    res_file = os.path.join(results_dir, f"final_test_preds_sub{subject_id}.npy")
    np.save(res_file, {'preds': preds, 'true': trues, 'acc': acc})
    print(f"Subject {subject_id} finished: {acc:.2f}%")

# -----------------------------------------------------------------------------
# 3. MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='Data/Preprocessed_2s_25overlap')
    parser.add_argument('--mode', type=str, default='sub_indep', choices=['sub_indep', 'sub_dep'])
    parser.add_argument('--max_parallel', type=int, default=1, help="Keep low for Raw Data (RAM usage)")
    args = parser.parse_args()
    
    label_file = os.path.join(args.dataset_folder, "label.mat")
    
    # 1. Load Everything
    X, y, subjects, sessions = load_raw_data(args.dataset_folder, label_file)
    
    # 2. Share Memory
    print("Constructing Shared Tensors (This may take memory)...")
    X_tensor = torch.tensor(X, dtype=torch.float32).share_memory_()
    y_tensor = torch.tensor(y, dtype=torch.long).share_memory_()
    sub_tensor = torch.tensor(subjects, dtype=torch.long).share_memory_()
    sess_tensor = torch.tensor(sessions, dtype=torch.long).share_memory_()
    base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).share_memory_()

    # 3. Run Setup
    run_id = get_next_run_id("CNNGCN_Raw")
    model_name_str = "CNNGCN_Raw"
    
    print(f"--- STARTING RUN: {model_name_str} | Attempt {run_id} | Mode: {args.mode} ---")
    
    unique_subs = np.unique(subjects)
    processes = []
    
    # 4. Processing Loop
    for i in range(0, len(unique_subs), args.max_parallel):
        chunk = unique_subs[i : i + args.max_parallel]
        print(f"\nProcessing Chunk: {chunk}")
        
        for sub_id in chunk:
            p = mp.Process(target=run_fold, args=(
                sub_id, args, X_tensor, y_tensor, sub_tensor, sess_tensor, base_edge_index, run_id
            ))
            p.start()
            processes.append(p)
            
        for p in processes: p.join()
        processes = []

    # 5. Global Aggregation
    print("\n" + "="*40 + "\nAGGREGATING RESULTS...\n" + "="*40)
    
    run_name = f"Attempt_{run_id}_LOSO_Raw" if args.mode == 'sub_indep' else f"Attempt_{run_id}_SessionHoldout_Raw"
    root_res_dir = os.path.join("Results", model_name_str, run_name)
    
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

    if all_preds:
        global_acc = np.mean(list(subject_accuracies.values()))
        report = classification_report(all_trues, all_preds, target_names=['Neg', 'Neu', 'Pos'])
        
        # Save Final Global Summary
        with open(os.path.join(root_res_dir, "Global_Summary.txt"), "w") as f:
            f.write(f"Architecture: CNNGCN_Raw (Attempt 61)\n")
            f.write(f"Global Mean Accuracy: {global_acc:.2f}%\n")
            f.write("-" * 30 + "\n")
            f.write(report)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()