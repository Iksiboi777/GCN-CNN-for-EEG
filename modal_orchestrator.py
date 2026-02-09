import modal
import os
import numpy as np
import sys
import time

# 1. DEFINE THE ENVIRONMENT
# This builds your Docker-like environment automatically
image = (
    modal.Image.debian_slim(python_version="3.11.5")
    .pip_install("torch", "torch_geometric", "numpy", "scikit-learn", "scipy", "matplotlib", 
                 "mne","mne_icalabel", "seaborn", "filterpy")
    .workdir("/data")
)

app = modal.App("eeg-de-loso-parallel")
volume = modal.Volume.from_name("eeg-data-volume")


@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/data": volume}, # Mounts /dataset/ to /root/dataset/
    timeout=7200,
    max_containers=5
)
def train_subject_remote(subject_id, args_dict):
    import torch
    import pandas as pd
    import sys
    import os
    import threading
    import time
    import subprocess
    from torch.utils.data import TensorDataset, DataLoader
    sys.path.append("/data") # Ensure we can import from the mounted volume
    os.chdir("/data") # Change working directory to the mounted volume

    # --- GPU MONITORING ---
    def log_gpu_stats():
        while True:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                ).strip()
                util, mem, temp = output.split(',')
                print(f"🔥 [GPU Monitor] Util: {util}% | VRAM: {mem} MB | Temp: {temp}C")
            except: pass
            time.sleep(60)

    threading.Thread(target=log_gpu_stats, daemon=True).start()
    
    # --- IMPORTS ---
    from Models.var_B import GCN_DE_Model
    from Models.var_C import DGCNN_Model
    from Models.var_D import Adaptive_DGCNN
    from Models.var_ind_graph import GraphSAGE_EEG_Model
    from Models.graph_construction import get_knn_adjacency_matrix
    from torch_geometric.utils import to_dense_adj
    from utils.training_utils import train_model_with_interrupt, evaluate
    from utils.focal_loss import FocalLoss
    from train_de import load_de_data, compute_rolling_variance
    
    # --- A. PARSE ARGS (Aligns with train_de.py) ---
    class VirtualArgs:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    args = VirtualArgs(args_dict)

    print(f"\n⚡ [Cloud Worker] Starting Subject {subject_id} on {torch.cuda.get_device_name(0)}")
    print(f"   Config: {args.model_type} | Window: {args.window_size} | SE: {args.use_se} | Features: {args.in_features}")
    
    # --- B. LOAD DATA DYNAMICALLY ---
    # Matches get_args() -> window_size logic
    data_folder = f"Data/ExtractedFeatures_{args.window_size}" 
    label_file = os.path.join(data_folder, "label.mat")
    
    try:
        X, y, subjects, sessions, _ = load_de_data(data_folder, label_file)
    except FileNotFoundError:
        # Fallback check for capitalization issues
        if os.path.exists(os.path.join(data_folder, "Label.mat")):
             label_file = os.path.join(data_folder, "Label.mat")
             X, y, subjects, sessions, _ = load_de_data(data_folder, label_file)
        else:
            print(f"CRITICAL ERROR: Data not found at {data_folder}. Check mount paths.")
            return {"subject": subject_id, "acc": 0.0, "status": "FailedData"}
    
    # Filter 1-15 (SEED Standard)
    mask_15 = subjects <= 15
    X, y, subjects, sessions = X[mask_15], y[mask_15], subjects[mask_15], sessions[mask_15]

    # --- C. PREPROCESSING (Feature Logic from train_de.py) ---
    # If in_features is 10, we calculate Rolling Variance
    if args.in_features == 10:
        print("Calculating Rolling Variance (Feature Augmentation)...")
        X = compute_rolling_variance(X, window_size=3) # Hardcoded as ROLLING_VAR_WINDOW=3 in train_de.py

    # --- FIX: REPLACED CRASHING PANDAS LOGIC WITH NUMPY Broadcasting ---
    print("Applying Vectorized Z-Score Normalization (Numpy 3D)...")
    
    # 1. Create unique group IDs for (Subject, Session) pairs
    # subjects and sessions are 1D arrays matching X's first dim
    group_ids = subjects * 1000 + sessions
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    n_groups = len(unique_groups)
    
    # 2. Initialize arrays for stats: Shape (n_groups, 62, 10)
    # X.shape is (N, 62, 10)
    group_sums = np.zeros((n_groups, *X.shape[1:]), dtype=np.float32)
    group_sq_sums = np.zeros((n_groups, *X.shape[1:]), dtype=np.float32)
    
    # 3. Accumulate sums (Vectorized "GroupBy" for 3D data)
    np.add.at(group_sums, group_indices, X)
    
    # 4. Counts
    group_counts = np.bincount(group_indices)
    
    # 5. Means: (n_groups, 62, 10)
    # Reshape counts for broadcasting: (n_groups, 1, 1)
    group_means = group_sums / group_counts[:, None, None]
    
    # 6. Center data
    # group_means[group_indices] expands to (N, 62, 10) to match X
    X_centered = X - group_means[group_indices]
    
    # 7. Stds
    np.add.at(group_sq_sums, group_indices, X_centered ** 2)
    group_stds = np.sqrt(group_sq_sums / group_counts[:, None, None])
    group_stds[group_stds < 1e-8] = 1.0 # Avoid div by zero
    
    # 8. Final Normalize
    X = X_centered / group_stds[group_indices]
    
    print(f"   ✅ Data Loaded & Normalized. Shape: {X.shape}", flush=True)

    # --- D. LOSO SPLIT ---
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    sub_tensor = torch.tensor(subjects, dtype=torch.long)
    
    train_mask = (sub_tensor != subject_id)
    test_mask = (sub_tensor == subject_id)

    X_train, y_train, sub_train = X_tensor[train_mask], y_tensor[train_mask], sub_tensor[train_mask]
    X_test, y_test, sub_test = X_tensor[test_mask], y_tensor[test_mask], sub_tensor[test_mask]


    # --- FIX START: EMPTY SUBJECT CHECK ---
    if len(X_test) == 0:
        print(f"❌ CRITICAL ERROR: No samples found for Subject {subject_id}!")
        print(f"   Available Subjects in Data: {np.unique(subjects)}")
        return {"subject": subject_id, "acc": 0.0, "status": "Failed(EmptySubject)"}
    # --- FIX END ---

    # --- E. DATALOADERS ---
    train_loader = DataLoader(TensorDataset(X_train, y_train, sub_train), 
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test, sub_test), 
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- F. MODEL INIT ---
    device = torch.device("cuda")
    base_edge_index = get_knn_adjacency_matrix("utils/channel_62_pos.locs", k=5).to(device)
    
    # Calculate input dim dynamically (should match args.in_features)
    calculated_in_features = args.in_features
    
    if args.model_type == 'ADAPTIVE_DGCNN':
        static_adj = to_dense_adj(base_edge_index, max_num_nodes=62)[0]
        model = Adaptive_DGCNN(
            num_nodes=62, 
            in_features=calculated_in_features, 
            num_classes=3,
            hidden_dim=128, 
            num_layers=2, 
            use_se=args.use_se,           # <--- Passed from args
            dropout_rate=0.5, 
            use_doubling=args.use_doubling, # <--- Passed from args
            num_subjects=15
        ).to(device)
        model.register_buffer('static_adj', static_adj)

    elif args.model_type == 'GCN':
        model = GCN_DE_Model(
            num_nodes=62, 
            in_features=calculated_in_features, 
            hidden_dim=128,
            num_classes=3, 
            num_layers=2, 
            dropout_rate=0.5, 
            num_subjects=15, 
            use_se=args.use_se,             # <--- Passed from args
            use_overlap_logic=args.use_overlap_logic, # <--- Passed from args
            use_doubling=args.use_doubling
        ).to(device)
        
    # ADDED: GraphSAGE Logic
    elif args.model_type == 'GraphSAGE':
        model = GraphSAGE_EEG_Model(
            num_nodes=62, in_features=calculated_in_features, hidden_dim=128, 
            num_classes=3, num_layers=2, aggregator='max', 
            use_se=args.use_se, use_doubling=args.use_doubling, 
            dropout_rate=0.5, num_subjects=15
        ).to(device)

    
    # --- G. TRAINING SETUP ---
    criterion = FocalLoss(gamma=2.0)
    
    # Split Optimizer Logic (Regularization for Gamma)
    gamma_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'static_norm.gamma' in name or 'adaptive_input.gamma' in name:
            gamma_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': other_params, 'weight_decay': 1e-3},
        {'params': gamma_params, 'weight_decay': 1e-2} 
    ], lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    res_dir = f"/results/{args.model_type}_Attempt_{args.run_id}/Subject_{subject_id}"
    os.makedirs(res_dir, exist_ok=True)
    
    def eval_wrapper(mod, load, adj, crit, dev, inf, return_preds=False, return_embeddings=False):
        return evaluate(mod, load, adj, crit, dev, inf, return_preds, return_embeddings)

    train_model_with_interrupt(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        results_dir=res_dir,
        params_dir=res_dir,
        errors_dir=res_dir,
        subject_tag=f"[Sub-{subject_id}]",
        base_edge_index=base_edge_index,
        evaluate_fn=eval_wrapper,
        in_features=calculated_in_features,
        patience=args.patience
    )
    
    # --- H. RETRIEVE RESULTS ---
    try:
        res_file = os.path.join(res_dir, "training_history.npy")
        history = np.load(res_file, allow_pickle=True).item()
        best_acc = max(history['val_acc'])
        
        preds_file = os.path.join(res_dir, f"final_test_preds_sub{subject_id}.npy")
        if os.path.exists(preds_file):
            preds_data = np.load(preds_file, allow_pickle=True).item()
            return {"subject": subject_id, "acc": best_acc, "preds": preds_data['preds'], "true": preds_data['true'], "status": "Success"}
        else:
             return {"subject": subject_id, "acc": best_acc, "preds": [], "true": [], "status": "Partial (No Preds)"}

    except Exception as e:
        print(f"Error retrieving results: {e}")
        return {"subject": subject_id, "acc": 0.0, "status": "Failed Retrieval"}

# --- 4. ORCHESTRATOR ---
@app.local_entrypoint()
def main():
    import time
    # Check if we can import logic from local train_de.py
    try:
        # Import dynamically from local file to get the exact ID logic
        from train_de import get_next_run_id
    except ImportError:
        def get_next_run_id(ws): return 999 # Fallback if local import fails
        print("Warning: Could not import get_next_run_id from train_de.py, using 999")

    from sklearn.metrics import classification_report, confusion_matrix
    
    # --- CONFIGURATION (Match train_de.py Args here) ---
    config = {
        "model_type": "GraphSAGE", # 'GCN', 'DGCNN', 'ADAPTIVE_DGCNN'
        "window_size": "1s",            # '1s', '4s'
        "in_features": 10,              # 5 or 10
        "use_se": True,                 # True/False
        "use_doubling": False,          # True/False
        "use_overlap_logic": False,     # True/False
        "mode": "sub_indep",            # 'sub_dep' or 'sub_indep'
        "batch_size": 2048,
        "epochs": 100,
        "learning_rate": 0.0005,
        "patience": 20 
    }
    
    subjects = list(range(1, 16))
    # Get Run ID locally so it increments your local file
    run_id = get_next_run_id(config["window_size"])
    config["run_id"] = run_id
    
    print(f"🚀 Launching Cloud LOSO for {config['model_type']} [Run ID: {config['run_id']}]")
    start_time = time.time()
    
    # Run in parallel
    results = list(train_subject_remote.map(subjects, kwargs={"args_dict": config}))
    
    end_time = time.time()
    
    print("\n" + "="*40)
    print("☁️  CLOUD TRAINING COMPLETE")
    print(f"⏱️  Total Time: {end_time - start_time:.2f}s")
    print("="*40)
    
    accuracies = []
    all_preds = []
    all_trues = []
    
    for res in results:
        status = res.get('status', 'Unknown')
        sub = res.get('subject', '?')
        acc = res.get('acc', 0.0)
        
        if status == 'Success':
            print(f"Subject {sub}: {acc:.2f}%")
            accuracies.append(acc)
            all_preds.extend(res['preds'])
            all_trues.extend(res['true'])
        else:
            print(f"Subject {sub}: FAILED ({status})")
        
    print("-" * 30)
    if accuracies:
        print(f"Global LOSO Accuracy: {np.mean(accuracies):.2f}% (+/- {np.std(accuracies):.2f}%)")
        
        if len(all_preds) > 0:
            print("\nGlobal Classification Report:")
            try:
                print(classification_report(all_trues, all_preds, target_names=['Neg', 'Neu', 'Pos']))
                print("\nConfusion Matrix:")
                print(confusion_matrix(all_trues, all_preds))
            except: pass