import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.io
from sklearn.metrics import accuracy_score

# --- Configuration ---
# Update these paths to match your actual result folder names for the Session Holdout runs
# Format: "Model Name": "Result_Folder_Name"
RUNS_TO_COMPARE = {
    'GCN': "Attempt_155_Phase2",        # Example: Replace with actual run ID
    'Adaptive_DGCNN': "Attempt_116_Phase2", # Example: Replace with actual run ID
    'GraphSAGE': "Attempt_113_Phase2"       # Example: Replace with actual run ID
}

WINDOW_SIZE = '1s' # or '1s'
BASE_RESULTS_DIR = "Results" 
DATA_FOLDER = f"Data/ExtractedFeatures_{WINDOW_SIZE}"
LABEL_FILE = os.path.join(DATA_FOLDER, "label.mat")

# Plot Styling
sns.set_theme(style="whitegrid")
PALETTE = "Pastel1"

def load_metadata_for_session_holdout(data_folder, label_file):
    """
    Reconstructs Subject IDs for Session 3 (Test Set).
    Adapted from analyze_history.py
    """
    print(f"Loading metadata from {data_folder}...")
    try:
        label_mat = scipy.io.loadmat(label_file)
    except FileNotFoundError:
        print("Error: Label file not found.")
        return None

    # Logic to reconstruct which sample belongs to which subject
    # We only care about Session 3 because that is the Test Set in 'sub_dep' mode
    subject_list = []
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    
    # Group files by subject
    for f in files:
        parts = f.split('_')
        try: subj_id = int(parts[0])
        except: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)
        
    # Iterate in order
    for subj_id in sorted(subject_files.keys()):
        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        
        # In Session Holdout, the test set is usually Session 3 (index 2)
        # We need to find how many samples exist for Session 3 for this subject
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            
            # We only care about Session 3 aka the Test Set
            if session_id != 3:
                continue

            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            
            for trial_i in range(1, 16):
                key = f"de_LDS{trial_i}"
                if key not in mat: continue
                data = mat[key]
                num_samples = data.shape[1] 
                
                # Append subject ID for every sample in this trial
                subject_list.append(np.full(num_samples, subj_id))

    if not subject_list:
        return np.array([])
        
    return np.concatenate(subject_list, axis=0)

def extract_per_subject_accuracy(model_name, run_folder, true_subjects):
    """
    Loads evolution_history.npy and calculates accuracy per subject for the final epoch.
    """
    # Construct path: Results/{Model_Type}_DE_{Window}/{Run_Name}
    full_model_name = f"{model_name}_DE_{WINDOW_SIZE}" if "DE" not in model_name else model_name
    
    # Determine the correct directory path depending on how your folders are named
    # Option A: Strict hierarchy
    path = os.path.join(BASE_RESULTS_DIR, full_model_name, run_folder, "evolution_history.npy")
    
    # Option B: Fallback if user put run_folder directly in Results
    if not os.path.exists(path):
        # Try searching loosely
        print(f"  -> Path {path} not found. Searching...")
        # (Add logic here if your folder structure varies)

    if not os.path.exists(path):
        print(f"  [Error] Could not find history file for {model_name} at {path}")
        return []

    data = np.load(path, allow_pickle=True).item()
    
    # Get predictions from the LAST epoch (best model is usually saved, but history tracks all)
    # Or you can look for 'best_val_acc' index if you tracked it.
    # Here we take the last logged epoch.
    final_preds = np.array(data['preds_history'][-1])
    true_labels = np.array(data['true_labels'])

    if len(final_preds) != len(true_subjects):
        print(f"  [Warning] Mismatch: Preds ({len(final_preds)}) vs Metadata ({len(true_subjects)})")
        # Truncate to match length (safety net)
        min_len = min(len(final_preds), len(true_subjects))
        final_preds = final_preds[:min_len]
        true_labels = true_labels[:min_len]
        true_subjects = true_subjects[:min_len]

    # Calculate Accuracy Per Subject
    subject_accuracies = []
    unique_subs = np.unique(true_subjects)
    
    for sub in unique_subs:
        mask = (true_subjects == sub)
        sub_acc = accuracy_score(true_labels[mask], final_preds[mask]) * 100
        subject_accuracies.append(sub_acc)
        
    print(f"  -> Loaded {model_name}: Mean Acc = {np.mean(subject_accuracies):.2f}%")
    return subject_accuracies

def create_violin_sess_holdout():
    # 1. Load Metadata (Ground Truth Subject IDs for the Test Set)
    true_subjects = load_metadata_for_session_holdout(DATA_FOLDER, LABEL_FILE)
    
    if len(true_subjects) == 0:
        print("Failed to load metadata. Check paths.")
        return

    # 2. Collect Data
    data_dict = {}
    
    print(f"--- Analyzing Session Holdout ({WINDOW_SIZE}) ---")
    for model_key, run_folder in RUNS_TO_COMPARE.items():
        accs = extract_per_subject_accuracy(model_key, run_folder, true_subjects)
        if accs:
            data_dict[model_key] = accs

    if not data_dict:
        print("No data extracted. Exiting.")
        return

    # 3. Plotting
    # Reusing the style from your provided violins.py
    df = pd.DataFrame(data_dict)
    df_melted = df.melt(var_name='Model', value_name='Accuracy (%)')
    
    plt.figure(figsize=(10, 6))
    
    # Violin
    ax = sns.violinplot(
        x='Model', 
        y='Accuracy (%)', 
        data=df_melted, 
        palette=PALETTE, 
        inner="quartile",
        cut=0 # Cut=0 limits the violin to the data range (no extrapolation beyond min/max)
    )
    
    # Strip Plot (Dots)
    sns.stripplot(
        x='Model', 
        y='Accuracy (%)', 
        data=df_melted, 
        color="black", 
        alpha=0.4, 
        jitter=True
    )
    
    plt.title(f'Distribucija točnosti po ispitanicima (Session Holdout, {WINDOW_SIZE})', fontsize=14)
    plt.ylim(30, 100) # Adjusted range for Session Holdout (often lower than LOSO)
    
    outfile = f"violin_SessHold_{WINDOW_SIZE}.png"
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"\nSaved plot to {outfile}")
    # plt.show()

if __name__ == "__main__":
    create_violin_sess_holdout()