import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
# Get the absolute path of the directory where this script resides
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "GCN_DE_4s" 
ATTEMPT_ID = "Attempt_60_LOSO_Parallel" 

# paths relative to PROJECT_ROOT
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data", "ExtractedFeatures_4s")  
LABEL_FILE = os.path.join(DATA_ROOT, "label.mat")

# Subjects to analyze
SUBJECTS_TO_ANALYZE = [1, 4, 6, 7, 10, 12, 13]

PARAMS_DIR = os.path.join(PROJECT_ROOT, "Params", MODEL_NAME, ATTEMPT_ID)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results", MODEL_NAME, ATTEMPT_ID)

# Standard Channel Order to find F3/F4
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

def load_checkpoint_weights(subject_id):
    path = os.path.join(PARAMS_DIR, f"Subject_{subject_id}", "best_model_checkpoint.pth")
    if not os.path.exists(path):
        print(f"  [Warn] No checkpoint found at {path}")
        return None
    return torch.load(path, map_location='cpu')

def load_real_subject_data_and_labels(subject_id):
    """
    Loads raw .mat files for specific subject to calculate REAL FAS values.
    Attempts to align with global labels from label.mat.
    """
    print(f"  -> Loading raw data for Subject {subject_id}...")
    
    if not os.path.exists(DATA_ROOT):
        print(f"  [Error] Data root not found: {DATA_ROOT}")
        return None, None

    # 1. Identify Subject Files
    file_prefix = f"{subject_id}_"
    files = [f for f in sorted(os.listdir(DATA_ROOT)) if f.startswith(file_prefix) and f.endswith('.mat')]
    
    if not files:
        print(f"  [Error] No .mat files found for subject prefix '{file_prefix}'")
        return None, None

    # 2. Extract FAS Features
    f3_idx = CHANNEL_NAMES.index('F3')
    f4_idx = CHANNEL_NAMES.index('F4')
    
    fas_values = []
    
    # Iterate to load data
    for f in files:
        try:
            file_path = os.path.join(DATA_ROOT, f)
            mat = loadmat(file_path)
            
            if 'de_data' in mat:
                data = mat['de_data'] # Shape (62, samples, 5)
            elif 'data' in mat:
                 # Handle cases where key might differ
                 data = mat['data']
            else:
                continue

            # Alpha band is index 2
            # Add epsilon to avoid log(0)
            f3_alpha = data[f3_idx, :, 2] + 1e-9
            f4_alpha = data[f4_idx, :, 2] + 1e-9
            
            # FAS Formula: ln(Right) - ln(Left)
            fas = np.log(f4_alpha) - np.log(f3_alpha)
            fas_values.extend(fas)
        except Exception as e:
            print(f"    Error reading file {f}: {e}")

    fas_values = np.array(fas_values)
    
    # 3. Load Labels from .mat and Slice
    true_labels = None
    if os.path.exists(LABEL_FILE):
        try:
            mat_labels = loadmat(LABEL_FILE)
            # Inspect keys to find label array (usually 'label' or 'labels')
            key = 'label' if 'label' in mat_labels else ('labels' if 'labels' in mat_labels else None)
            
            if key:
                all_labels = mat_labels[key]
                # Flatten if it's 2D (1, N) -> (N,)
                all_labels = all_labels.flatten()
                
                # Handling Label alignment (Assuming concatenated structure)
                # SEED typically repeats the label sequence for 15 subjects
                # Check if label length matches single subject or full dataset
                
                total_samples = len(all_labels)
                num_subs_heuristic = 15
                
                # If labels are huge (e.g. 45,000), it's the full set
                if total_samples > 10000:
                    samples_per_sub = total_samples // num_subs_heuristic
                    start_idx = (subject_id - 1) * samples_per_sub
                    end_idx = subject_id * samples_per_sub
                    true_labels = all_labels[start_idx:end_idx]
                else:
                    # If labels are small (e.g. 3000), it's likely just one subject's labels
                    # meant to be repeated for everyone
                    true_labels = all_labels

                # Trim if lengths differ slightly due to preprocessing artifacts
                if len(fas_values) != len(true_labels):
                    min_len = min(len(fas_values), len(true_labels))
                    fas_values = fas_values[:min_len]
                    true_labels = true_labels[:min_len]
            else:
                print("  [Error] Could not find 'label' or 'labels' key in .mat file")
        except Exception as e:
            print(f"  [Error] Failed to load label.mat: {e}")
            
    else:
         print(f"  [Warn] label.mat not found at {LABEL_FILE}")

    return fas_values, true_labels

# =================================================================
# 1. THE "TRUST" CHECK
# =================================================================
def plot_feature_weights(subject_id):
    ckpt = load_checkpoint_weights(subject_id)
    if ckpt is None: return

    gamma_key =  'adaptive_input.gamma' if 'adaptive_input.gamma' in ckpt else 'static_norm.gamma'
    
    if gamma_key not in ckpt:
        print("  [Info] Model has no readable input weights.")
        return

    gamma = ckpt[gamma_key].cpu().numpy()
    # Average across nodes (1, 62, 11) -> (11,)
    avg_weights = np.mean(np.abs(gamma), axis=(0, 1))
    
    features = ['Del', 'The', 'Alp', 'Bet', 'Gam', 
                'V_Del', 'V_The', 'V_Alp', 'V_Bet', 'V_Gam', 
                'FAS']
    
    plt.figure(figsize=(12, 6))
    colors = ['steelblue']*5 + ['salmon']*5 + ['purple']
    
    if len(avg_weights) != 11:
        print(f"  [Warn] Weight shape {avg_weights.shape} does not match 11 features. Skipping FAS specific check.")
        return

    bars = plt.bar(features, avg_weights, color=colors)
    
    fas_weight = avg_weights[10]
    
    plt.title(f"Subject {subject_id}: Learned Feature Weights ('Trust')")
    plt.ylabel("Weight Magnitude")
    plt.grid(axis='y', alpha=0.3)
    
    plt.text(10, fas_weight, f"{fas_weight:.4f}", ha='center', va='bottom', fontweight='bold')

    result_text = "IGNORED" if fas_weight < 0.01 else ("DOMINANT" if fas_weight > np.max(avg_weights[:10]) else "USED")
    print(f"  -> FAS Weight Status: {result_text} ({fas_weight:.4f})")
    
    save_path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "DIAG_1_Trust_Weights.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# =================================================================
# 2. THE "REALITY" CHECK
# =================================================================
def plot_fas_distribution_overlap(subject_id):
    fas_values, true_labels = load_real_subject_data_and_labels(subject_id)
    
    if fas_values is None or true_labels is None or len(fas_values) == 0:
        print("  [Skip] Cannot plot distribution (Empty Data).")
        return
     
    # Normalize labels to 0, 1, 2
    unique_labels = np.unique(true_labels)
    # print(f"    Unique Labels Found: {unique_labels}")
    
    if -1 in unique_labels:
        true_labels = true_labels + 1
        
    plt.figure(figsize=(10, 6))
    
    # We use a try-except block for plotting to catch "No data for this class" issues cleanly
    has_data = False
    
    # Negative (0)
    if np.any(true_labels == 0):
        try:
            sns.kdeplot(fas_values[true_labels == 0], fill=True, label='Negative', color='red', alpha=0.3, warn_singular=False)
            has_data = True
        except: pass
        
    # Neutral (1)
    if np.any(true_labels == 1):
        try:
            sns.kdeplot(fas_values[true_labels == 1], fill=True, label='Neutral', color='gray', alpha=0.3, warn_singular=False)
            has_data = True
        except: pass

    # Positive (2)
    if np.any(true_labels == 2):
        try:
            sns.kdeplot(fas_values[true_labels == 2], fill=True, label='Positive', color='blue', alpha=0.1, warn_singular=False)
            has_data = True
        except: pass

    if not has_data:
        print("  [Warn] Labels exist but no KDE could be plotted (Singular matrix or constant values?)")
        plt.close()
        return

    plt.title(f"Subject {subject_id}: Ground Truth FAS Distribution ('Reality')")
    plt.xlabel("FAS Value (DE(F4) - DE(F3))")
    plt.axvline(0, color='black', linestyle='--')
    plt.legend()
    
    save_path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "DIAG_2_Reality_Distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print("  -> Saved Distribution Plot")

# =================================================================
# 3. THE "EVOLUTION" CHECK
# =================================================================
def plot_evolution_trajectory(subject_id):
    history_file = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "evolution_history.npy")
    if not os.path.exists(history_file):
        print("  [Skip] No evolution history found.")
        return

    data = np.load(history_file, allow_pickle=True).item()
    
    true_labels = data['true_labels']
    preds_history = data['preds_history']
    
    epochs = [i for i in range(len(preds_history))] 
    
    neg_recall = []
    neu_recall = []
    
    for preds in preds_history:
        min_len = min(len(preds), len(true_labels))
        p = preds[:min_len]
        t = true_labels[:min_len]
        
        acc_neg = accuracy_score(t[t==0], p[t==0]) if np.any(t==0) else 0
        acc_neu = accuracy_score(t[t==1], p[t==1]) if np.any(t==1) else 0
        
        neg_recall.append(acc_neg)
        neu_recall.append(acc_neu)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, neg_recall, 'r-o', label='Negative Recall', linewidth=2)
    plt.plot(epochs, neu_recall, 'gray', linestyle='--', label='Neutral Recall', linewidth=2)
    
    plt.title(f"Subject {subject_id}: Class Separation Trajectory")
    plt.xlabel("Training Snapshot")
    plt.ylabel("Recall (Efficiency)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "DIAG_3_Evolution_Trajectory.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print("  -> Saved Evolution Trajectory")

# =================================================================
# MAIN LOOP
# =================================================================
if __name__ == "__main__":
    print(f"--- DIAGNOSTIC RUN: {MODEL_NAME} / {ATTEMPT_ID} ---")
    print(f"Running from root: {PROJECT_ROOT}")
    
    for sub in SUBJECTS_TO_ANALYZE:
        print(f"\n[{sub}] Processing Subject {sub}...")
        
        # plot_feature_weights(sub)
        plot_fas_distribution_overlap(sub)
        # plot_evolution_trajectory(sub)
            
    print("\nDiagnostics Complete.")