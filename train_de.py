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

from Models.var_B import GCN_DE_Model
from Models.var_D import Adaptive_DGCNN
from Models.var_ind_graph import GraphSAGE_EEG_Model
from Models.graph_construction import get_knn_adjacency_matrix
from torch_geometric.utils import to_dense_adj
from utils.training_utils import train_model_with_interrupt, evaluate
from utils.focal_loss import FocalLoss

import torch.multiprocessing as mp


LOCS_FILE = "utils/channel_62_pos.locs"
BATCH_SIZE = 1024
EPOCHS = 60
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-3 
PATIENCE = 20
L1_LAMBDA = 1e-4 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_FILE = "run_config.json"

ROLLING_VAR_WINDOW = 3   


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
    parser.add_argument('--mode', type=str, default='sub_indep', choices=['sub_dep', 'sub_indep'],
                        help="Training mode: 'sub_dep' (Session split) or 'sub_indep' (LOSO)")
    parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s', '2s'],
                        help="Feature window size: '1s' or '4s'")
    parser.add_argument('--model_type', type=str, default = 'GraphSAGE', 
                        choices=['GCN', 'DGCNN', 'ADAPTIVE_DGCNN', 'GraphSAGE'],
                        help="Type of GCN model to use")
    parser.add_argument('--max_parallel', type=int, default=3, 
                        help="Maximum number of parallel processes")
    parser.add_argument('--use_overlap_logic', type=bool, default=False,
                        help="Whether to use overlap logic in GCN_DE_Model")
    parser.add_argument('--use_doubling', type=bool, default=False,
                        help="Whether to use feature doubling in GCN_DE_Model")
    parser.add_argument('--use_se', type=bool, default=True, 
                        help="Whether to use SE block in GCN_DE_Model")
    parser.add_argument('--in_features', type=int, default=10, choices=[5, 10],
                        help="Number of input features per node")
    return parser.parse_args()

def compute_rolling_variance(data, window_size=ROLLING_VAR_WINDOW):
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
                
                shape = data.shape
                if shape[0] == 62:
                    if shape[2] == 5:
                        pass 
                    elif shape[1] == 5:
                        data = np.transpose(data, (0, 2, 1)) 
                elif shape[1] == 62:
                    if shape[2] == 5:
                        data = np.transpose(data, (1, 0, 2)) 
                    elif shape[0] == 5:
                        data = np.transpose(data, (1, 2, 0)) 
                elif shape[2] == 62:
                    if shape[1] == 5:
                        data = np.transpose(data, (2, 0, 1)) 
                    elif shape[0] == 5:
                        data = np.transpose(data, (2, 1, 0)) 


                args = get_args() 
                if args.in_features == 10:
                    data_var = compute_rolling_variance(data, window_size=ROLLING_VAR_WINDOW)
                    data_final = np.concatenate([data, data_var], axis=2)
                    data_final = np.transpose(data_final, (1, 0, 2))

                if args.in_features == 5:
                    data_final = np.transpose(data, (1, 0, 2))

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
    
    print("Applying Manual Subject-Specific & Session-Specific Z-Score Normalization...")
    
    group_ids = subjects * 1000 + sessions
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    
    n_groups = len(unique_groups)
    
    group_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    group_sq_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    
    np.add.at(group_sums, group_indices, X)

    group_counts = np.bincount(group_indices)

    group_means = group_sums / group_counts[:, None, None]

    expanded_means = group_means[group_indices]
    X_centered = X - expanded_means

    np.add.at(group_sq_sums, group_indices, X_centered ** 2)
    group_stds = np.sqrt(group_sq_sums / group_counts[:, None, None])
    group_stds[group_stds < 1e-6] = 1.0 

    expanded_stds = group_stds[group_indices]
    X = X_centered / expanded_stds
    print(f"Total Samples: {X.shape[0]}")
    return X, y, sessions, subjects, trials



def run_single_subject_fold(subject_id, args, X_full, y_full, sub_full, 
                            base_edge_index, run_id, model_name):
    """
    Runs training and evaluation for one specific subject in a separate process.
    """
    torch.set_num_threads(2)  


    num_gpus = torch.cuda.device_count()
    local_device = torch.device(f"cuda:{subject_id % num_gpus}" if num_gpus > 0 else "cpu")
    print(f"\n[PROCESS START] Subject {subject_id} assigned to {local_device}")


    test_mask = (sub_full == subject_id)
    X_train, y_train = X_full[~test_mask], y_full[~test_mask]
    X_test, y_test = X_full[test_mask], y_full[test_mask]


    IN_FEATURES = args.in_features  
    
    if args.model_type == 'GCN':
        print("Initializing Static GCN Model...")
        model = GCN_DE_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=128, 
                            num_classes=3, dropout_rate=0.5, num_layers=2, use_doubling=args.use_doubling, use_se=args.use_se).to(local_device)
    elif args.model_type == 'ADAPTIVE_DGCNN':
        static_adj = to_dense_adj(base_edge_index, max_num_nodes=62)[0].to(local_device)
        # This is the new model with var_B's Gatekeepers and var_C's Dynamic Brain
        model = Adaptive_DGCNN(static_adj=static_adj, num_nodes=62, in_features=IN_FEATURES, num_classes=3, use_se=args.use_se,
                                hidden_dim=128, num_layers=2, use_doubling=args.use_doubling).to(local_device)
        print("Using Adaptive DGCNN (var_D) - The 83% Hybrid Architecture")
    elif args.model_type == 'GraphSAGE':
        print("Initializing GraphSAGE Model...")
        model = GraphSAGE_EEG_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=128, 
                                    num_classes=3, num_layers=2, aggregator='lstm', use_se=args.use_se, 
                                    use_doubling=args.use_doubling, dropout_rate=0.5, num_subjects=15).to(local_device)


    lr_val = globals().get('LEARNING_RATE', 0.0005)
    wd_val = globals().get('WEIGHT_DECAY', 1e-3)
    epochs_val = globals().get('EPOCHS', 60)
    batch_size_val = globals().get('BATCH_SIZE', 128)

    gamma_params = [p for n, p in model.named_parameters() if 'static_norm.gamma' in n]
    other_params = [p for n, p in model.named_parameters() if 'static_norm.gamma' not in n]
    
    optimizer = optim.Adam([
        {'params': other_params, 'weight_decay': wd_val},
        {'params': gamma_params, 'weight_decay': 1e-2} 
    ], lr=lr_val)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS)
    alpha_weights = torch.tensor([1.2, 1.1, 0.9]).to(local_device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)


    run_name = f"Attempt_{run_id}_LOSO_Parallel"
    results_dir = os.path.join("Results", model_name, run_name, f"Subject_{subject_id}")
    errors_dir = os.path.join("Errors", model_name, run_name, f"Subject_{subject_id}")
    params_dir = os.path.join("Params", model_name, run_name, f"Subject_{subject_id}")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)


    sub_train = sub_full[~test_mask]
    sub_test = sub_full[test_mask]

    train_loader = DataLoader(TensorDataset(X_train, y_train, sub_train), 
                                batch_size=batch_size_val, 
                                shuffle=True,
                                num_workers=0,     
                                pin_memory=True 
                                )
    test_loader = DataLoader(TensorDataset(X_test, y_test, sub_test), 
                                batch_size=batch_size_val, 
                                shuffle=False,
                                num_workers=0,     
                                pin_memory=True
                                )

    train_model_with_interrupt(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs_val,
        device=local_device,
        results_dir=results_dir,
        params_dir=params_dir,
        errors_dir=errors_dir,
        subject_tag = f"Subject_{subject_id}",
        base_edge_index=base_edge_index.to(local_device),
        evaluate_fn=evaluate,
        in_features=IN_FEATURES
    )


def main():
    args = get_args()
    

    data_folder = os.path.join("Data", f"ExtractedFeatures_{args.window_size}")
    label_file = os.path.join(data_folder, "label.mat")
    
    X, y, sessions, subjects, _ = load_de_data(data_folder, label_file)
    
    run_id = get_next_run_id(args.window_size)
    model_name = f"{args.model_type}_DE_{args.window_size}"

    if args.mode == 'sub_indep':
        print("Running Leave-One-Subject-Out with Parallel Processing...")    
        processes = []
        subject_list = list(range(1, 16))
        X_tensor = torch.tensor(X, dtype=torch.float32).share_memory_()
        y_tensor = torch.tensor(y, dtype=torch.long).share_memory_()
        sub_tensor = torch.tensor(subjects, dtype=torch.long).share_memory_()
        base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).share_memory_()


        for i in range(0, 15, args.max_parallel):
            chunk = subject_list[i : i + args.max_parallel]
            print(f"\n--- Starting Chunk: Subjects {chunk} ---")
            
            for sub_id in chunk:
                p = mp.Process(target=run_single_subject_fold, 
                               args=(sub_id, args, X_tensor, y_tensor, 
                                     sub_tensor, base_edge_index, run_id, 
                                     model_name))
                p.start()
                processes.append(p)
            
            try:
                for p in processes: p.join()
            except KeyboardInterrupt:
                print(f"\n\n{'!'*40}")
                print(f"MAIN PROCESS INTERRUPTED ON CHUNK {chunk}")
                print(f"Waiting for children to save state and exit...")
                print(f"{'!'*40}\n")
                
                for p in processes: 
                    if p.is_alive(): p.join()
                
                print("\n>>> Moving to next subject chunk... (Press Ctrl+C again quickly to stop completely)")
                processes = []
                continue 

            processes = []
        
        print("\n" + "="*40)
        print("AGGREGATING GLOBAL RESULTS FROM DISK...")
        print("="*40 + "\n")
        
        all_preds = []
        all_trues = []
        subject_accuracies = {}
        
        root_res_dir = os.path.join("Results", model_name, f"Attempt_{run_id}_LOSO_Parallel")
        
        for sub_id in subject_list:
            res_file = os.path.join(root_res_dir, f"Subject_{sub_id}", f"final_test_preds_sub{sub_id}.npy")
            
            if os.path.exists(res_file):
                data = np.load(res_file, allow_pickle=True).item()
                preds = data['preds']
                trues = data['true']
                acc = data['acc']
                
                subject_accuracies[sub_id] = acc
                all_preds.extend(preds)
                all_trues.extend(trues)
                print(f"Loaded Subject {sub_id}: {acc:.2f}%")
            else:
                print(f"Warning: Results missing for Subject {sub_id}")
                subject_accuracies[sub_id] = 0.0

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        
        if len(all_preds) == 0:
            print("\n[ERROR] No predictions were aggregated. Skipping report generation.")
            print("Did the training processes finish and save their .npy files?")
        else:
            global_acc = np.mean(list(subject_accuracies.values()))
            std_acc = np.std(list(subject_accuracies.values()))
            
            print(f"\n[LOSO COMPLETE] Mean Acc: {global_acc:.2f}% (+/- {std_acc:.2f}%)")
            
            class_names = ['Negative', 'Neutral', 'Positive']
            
            try:
                cls_report = classification_report(all_trues, all_preds, target_names=class_names)
                conf_matrix = confusion_matrix(all_trues, all_preds)

                print("\nGlobal Classification Report:")
                print(cls_report)
                print("\nGlobal Confusion Matrix:")
                print(conf_matrix)

                with open(os.path.join(root_res_dir, "LOSO_Global_Summary.txt"), "w") as f:
                    f.write(f"Global LOSO Average: {global_acc:.2f}% (+/- {std_acc:.2f}%)\n")
                    f.write("-" * 30 + "\n")
                    f.write("Per Subject Accuracies:\n")
                    for sub, acc in subject_accuracies.items():
                        f.write(f"Subject {sub}: {acc:.2f}%\n")
                    f.write("\n" + "="*30 + "\n")
                    f.write("GLOBAL CLASSIFICATION REPORT:\n")
                    f.write(cls_report)
                    f.write("\n" + "="*30 + "\n")
                    f.write("GLOBAL CONFUSION MATRIX:\n")
                    f.write(str(conf_matrix))
                    
                print(f"Global summary saved to {root_res_dir}")
            except Exception as e:
                print(f"Error generating report: {e}")


    else:
        print("  -> Strategy: Session Holdout (Train on S1+S2, Test on S3)")
        train_mask = (sessions == 1) | (sessions == 2)
        test_mask = (sessions == 3)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
        sub_tensor = torch.tensor(subjects, dtype=torch.long).to(DEVICE)
        base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)

        
        X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
        X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
        
        sub_train = sub_tensor[train_mask]
        sub_test = sub_tensor[test_mask]
        
        train_loader = DataLoader(TensorDataset(X_train, y_train, sub_train), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test, sub_test), 
                                 batch_size=BATCH_SIZE, 
                                 shuffle=False)

        run_name = f"Attempt_{run_id}_Phase2"
        results_dir = os.path.join("Results", model_name, run_name)
        params_dir = os.path.join("Params", model_name, run_name)
        errors_dir = os.path.join("Errors", model_name, run_name)

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)
        os.makedirs(errors_dir, exist_ok=True)


        IN_FEATURES = args.in_features 
        
        if args.model_type == 'GCN':
            print("Initializing Static GCN Model...")
            model = GCN_DE_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=128, 
                                num_classes=3, dropout_rate=0.5, num_layers=2, use_doubling=args.use_doubling, use_se=args.use_se).to(DEVICE)
        elif args.model_type == 'ADAPTIVE_DGCNN':
            static_adj = to_dense_adj(base_edge_index, max_num_nodes=62)[0].to(DEVICE)
            # This is the new model with var_B's Gatekeepers and var_C's Dynamic Brain
            model = Adaptive_DGCNN(static_adj=static_adj, num_nodes=62, in_features=IN_FEATURES, num_classes=3, use_se=args.use_se,
                                   hidden_dim=128, num_layers=2, use_doubling=args.use_doubling).to(DEVICE)
            print("Using Adaptive DGCNN (var_D) - The 83% Hybrid Architecture")
        elif args.model_type == 'GraphSAGE':
            print("Initializing GraphSAGE Model...")
            model = GraphSAGE_EEG_Model(num_nodes=62, in_features=IN_FEATURES, hidden_dim=128, 
                                        num_classes=3, num_layers=2, aggregator='lstm', use_se=args.use_se, 
                                        use_doubling=args.use_doubling, dropout_rate=0.5, num_subjects=15).to(DEVICE)

        
        gamma_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'static_norm.gamma' in name:
                gamma_params.append(param)
            else:
                other_params.append(param)
                
        optimizer = optim.Adam([
            {'params': other_params, 'weight_decay': WEIGHT_DECAY}, 
            {'params': gamma_params, 'weight_decay': 1e-2}          
        ], lr=LEARNING_RATE)
        

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS)
        

        class_weights = torch.tensor([1.2, 0.9, 1.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        train_model_with_interrupt(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=EPOCHS,
            device=DEVICE,
            # patience=PATIENCE,
            results_dir=results_dir,
            params_dir=params_dir,
            errors_dir=errors_dir,
            subject_tag="SessionHoldout",
            base_edge_index=base_edge_index,
            evaluate_fn=evaluate,
            hyperparams=args,
            in_features=IN_FEATURES  
        )        


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()