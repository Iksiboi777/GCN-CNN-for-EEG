import modal
import os

# 1. Define the environment and MOUNT your local folders
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "torch_geometric", "numpy", "scikit-learn", "scipy", "matplotlib")
)

app = modal.App("eeg-raw-loso-parallel")
volume = modal.Volume.from_name("eeg-data-volume")

# This is the "Sync" button. It mounts your local scripts into the cloud.
# It assumes your scripts are in the same folder as this launcher.
script_mount = modal.Mount.from_local_dir(".", remote_path="/root/project")

# 2. THE WORKER (Your 'run_fold' logic)
@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": volume},
    mounts=[script_mount], # This makes var_A.py and training_utils.py available
    timeout=7200,
    workdir="/root/project" # Tells Modal to run from your project root
)
def cloud_worker(subject_id, args_mode):
    # NOW YOU CAN IMPORT YOUR SCRIPTS NORMALLY
    import torch
    import numpy as np
    from Models.var_A import Attempt61_CNNGCN
    from utils.training_utils import train_model_with_interrupt
    from train import load_raw_data, get_knn_adjacency_matrix # Adjust as needed
    
    print(f"🚀 Subject {subject_id} starting on A100-80GB...")

    # A. Load Data from the Volume (Fast!)
    X, y, sub, sess = load_raw_data("/data/dataset", "/data/dataset/label.mat")
    
    # B. Get your Adjacency (Integrity preserved)
    base_edge_index = get_knn_adjacency_matrix(locs_file="channel_62_pos.locs")

    # C. Trigger your existing 'run_fold' logic
    # We pass 'args_mode' as a string ('sub_indep' or 'sub_dep')
    class MockArgs: mode = args_mode
    
    # This runs YOUR EXACT run_fold logic
    # Make sure run_fold is imported or defined in your scripts
    from train import run_fold 
    run_fold(subject_id, MockArgs(), X, y, sub, sess, base_edge_index, run_id=61)

    return f"Subject {subject_id} Complete."

# 3. THE PARALLEL COMMAND
@app.local_entrypoint()
def main(mode: str = "sub_indep"):
    print(f"Starting parallel {mode} run on Modal...")
    
    # Trigger all 15 subjects at once
    # This sends each one to its own A100-80GB GPU
    results = list(cloud_worker.map(range(1, 16), kwargs={"args_mode": mode}))
    
    for r in results:
        print(r)