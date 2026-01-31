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

app = modal.App("eeg-raw-loso-parallel")
volume = modal.Volume.from_name("eeg-data-volume")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": volume}, # Mounts /dataset/ to /root/dataset/
    timeout=7200
)
def cloud_worker(subject_id, mode_str, run_id):
    # Move to project dir so imports work
    sys.path.append("/data")
    os.chdir("/data")

    # --- ADDED IMPORTS ---
    import threading
    import time
    import subprocess
    
    # --- DEFINE MONITOR FUNCTION ---
    def log_gpu_stats():
        while True:
            try:
                # Query nvidia-smi for Utilization and Memory Used
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                ).strip()
                util, mem, temp = output.split(',')
                print(f"🔥 [GPU Monitor] Util: {util}% | VRAM: {mem} MB | Temp: {temp}C")
            except Exception as e:
                print(f"GPU Monitor Error: {e}")
            
            # Wait 60 seconds before next check
            time.sleep(60)

    # --- START MONITOR IN BACKGROUND ---
    # Daemon=True means this thread dies automatically when the main function finishes
    threading.Thread(target=log_gpu_stats, daemon=True).start()
    
    import torch
    import numpy as np
    # Standard Imports: These now work because of the Mount and workdir
    from train import load_raw_data, run_fold, ScaledSharedDataset
    from train_sage import load_de_data, 
    from Models.graph_construction import get_knn_adjacency_matrix
    
    # Replicate your 'args' object logic
    class Args: pass
    args = Args(); args.mode = mode_str; args.dataset_folder = "Data/Preprocessed_SOTA_individual"

    # 1. Load Data (Volume access is 10x faster than local PCIe)
    label_file = os.path.join(args.dataset_folder, "label.mat")
    X, y, subjects, sessions = load_raw_data(args.dataset_folder, label_file)
    
    # 2. Move to GPU (A100 80GB can take the whole dataset)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    sub_tensor = torch.tensor(subjects, dtype=torch.long)
    sess_tensor = torch.tensor(sessions, dtype=torch.long)

    # 3. Adjacency
    base_edge_index = get_knn_adjacency_matrix("utils/channel_62_pos.locs", k=5).cuda()

    # 4. Run
    # run_fold creates folders in /root/project/Results/...
    run_fold(subject_id, args, X_tensor, y_tensor, sub_tensor, sess_tensor, base_edge_index, run_id)

    print(f"--- [Cloud] Sub {subject_id} | Mode: {args.mode} | GPU: A100-80GB ---")

    # 4. Retrieve Result
    # run_fold saves to Results/... inside the container. We read it back.
    run_name = f"Attempt_{run_id}_{'LOSO' if mode_str=='sub_indep' else 'SessionHoldout'}_Raw"
    res_path = f"Results/SOTA/{run_name}/Subject_{subject_id}/final_test_preds_sub{subject_id}.npy"
    
    if os.path.exists(res_path):
        return np.load(res_path, allow_pickle=True).item()
    return None

# ...existing code...
@app.local_entrypoint()
def main(mode: str = "sub_indep"):
    from train import get_next_run_id
    from sklearn.metrics import classification_report
    import numpy as np

    run_id = 908  # Set manually or use get_next_run_id if synced
    
    # --- STEP 1: SAFETY DRY RUN ---
    print(f"\n🧪 STARTING DRY RUN (Subject 1) for Run {run_id}...")
    
    # We map only a single item list: [1]
    dry_run_results = list(cloud_worker.map([1], kwargs={"mode_str": mode, "run_id": run_id}))
    
    if not dry_run_results or dry_run_results[0] is None:
        print("\n❌ DATA/TRAINING CRASHED ON SUBJECT 1. ABORTING REST TO SAVE MONEY.")
        return
    
    # If we get here, Subject 1 finished successfully
    acc = dry_run_results[0]['acc']
    print(f"\n✅ Dry Run Successful! Subject 1 Accuracy: {acc:.2f}%")
    print("🚀 Unleashing the remaining 14 subjects...")

    # --- STEP 2: FULL RUN (Subjects 2-15) ---
    remaining_subs = list(range(2, 16))
    
    # Now we trigger the parallel swarm for the rest
    rest_results = list(cloud_worker.map(remaining_subs, kwargs={"mode_str": mode, "run_id": run_id}))

    # --- STEP 3: AGGREGATION ---
    results = dry_run_results + rest_results
    
    print("\n" + "="*40 + "\nAGGREGATING CLOUD RESULTS...\n" + "="*40)
    
    subject_accuracies, all_preds, all_trues = {}, [], []
    for i, data in enumerate(results):
        if data:
            sub_id = i + 1
            subject_accuracies[sub_id] = data['acc']
            all_preds.extend(data['preds'])
            all_trues.extend(data['true'])
            print(f"Subject {sub_id}: {data['acc']:.2f}%")

    if all_preds:
        global_acc = np.mean(list(subject_accuracies.values()))
        report = classification_report(all_trues, all_preds, target_names=['Neg', 'Neu', 'Pos'])
        print(f"\nGlobal Mean Accuracy: {global_acc:.2f}%\n")
        print(report)
        
        with open(f"Global_Summary_Attempt_{run_id}.txt", "w") as f:
            f.write(f"Global Mean Accuracy: {global_acc:.2f}%\n{report}")