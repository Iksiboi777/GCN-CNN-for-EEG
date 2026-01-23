import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import math
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
MODEL_NAME = "GCN_DE_4s" 
ATTEMPT_ID = "Attempt_56_LOSO_Parallel" 

RESULTS_DIR = f"Results/{MODEL_NAME}/{ATTEMPT_ID}"
PARAMS_DIR = f"Params/{MODEL_NAME}/{ATTEMPT_ID}"

# CORRECTED CHANNEL LIST
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

MNE_MAPPING = {
    'CB1': 'PO9', 
    'CB2': 'PO10'
}

def load_model_weights(subject_id):
    path = os.path.join(PARAMS_DIR, f"Subject_{subject_id}", "best_model_checkpoint.pth")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')

# ==========================================
# 1. CLASS EVOLUTION HEATMAP (Refactored for Grid)
# ==========================================
def plot_evolution_heatmap_ax(subject_id, ax):
    path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "evolution_history.npy")
    
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
        return

    data = np.load(path, allow_pickle=True).item()
    true_labels = data['true_labels']
    preds_history = data['preds_history']
    
    epochs = np.arange(1, len(preds_history) + 1)
    # Downsample x-ticks if too many epochs
    xtick_freq = max(1, len(epochs) // 10)
    
    classes = ['Neg', 'Neu', 'Pos']
    heatmap_matrix = np.zeros((3, len(preds_history)))

    for i, preds_epoch in enumerate(preds_history):
        current_preds = preds_epoch
        current_labels = true_labels
        if len(current_preds) != len(current_labels):
            min_len = min(len(current_preds), len(current_labels))
            current_preds = current_preds[:min_len]
            current_labels = current_labels[:min_len]

        for cls_idx in range(3):
            mask = (current_labels == cls_idx)
            if np.sum(mask) > 0:
                acc = accuracy_score(current_labels[mask], current_preds[mask])
                heatmap_matrix[cls_idx, i] = acc

    sns.heatmap(heatmap_matrix, annot=False, cmap="RdYlGn", ax=ax, cbar=False,
                xticklabels=xtick_freq, yticklabels=classes, vmin=0, vmax=1)
    
    # Set ticks manually to match downsampling
    ax.set_xticks(np.arange(0, len(epochs), xtick_freq))
    ax.set_xticklabels(epochs[::xtick_freq], rotation=45)
    
    ax.set_title(f"Sub-{subject_id} Evol.")
    ax.set_xlabel("Snap")
    ax.set_ylabel("Class")

# ==========================================
# 2. FEATURE IMPORTANCE (Refactored for Grid)
# ==========================================
def plot_band_importance_ax(subject_id, ax):
    ckpt = load_model_weights(subject_id)
    if ckpt is None: 
        ax.text(0.5, 0.5, "No Weights", ha='center', va='center')
        return

    weights = None
    if 'static_norm.gamma' in ckpt:
        weights = ckpt['static_norm.gamma'].abs().numpy()
        if len(weights.shape) == 3: weights = weights[0]
        importance = np.mean(weights, axis=0) 
    elif 'conv1.lin.weight' in ckpt:
        weights = ckpt['conv1.lin.weight'].abs().numpy()
        importance = np.mean(weights, axis=0) 
    else:
        ax.text(0.5, 0.5, "Arch Mismatch", ha='center', va='center')
        return

    count = len(importance)
    if count == 5:
        bands = ['Del', 'The', 'Alp', 'Bet', 'Gam']
        ax.bar(bands, importance, color='steelblue')
    elif count == 10:
        bands = ['Del', 'The', 'Alp', 'Bet', 'Gam']
        x = np.arange(5)
        width = 0.35
        ax.bar(x - width/2, importance[:5], width, label='DE', color='steelblue')
        ax.bar(x + width/2, importance[5:], width, label='Var', color='salmon')
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        if subject_id == 1: # Only legend on the first one to save space
            ax.legend(fontsize='x-small')
    else:
        ax.bar(range(count), importance)

    ax.set_title(f"Sub-{subject_id} Feats")
    ax.tick_params(axis='x', rotation=45, labelsize=8)

# ==========================================
# 3. SPATIAL TOPOMAP (Refactored for Grid)
# ==========================================
def plot_brain_map_ax(subject_id, ax, band_idx=2): 
    ckpt = load_model_weights(subject_id)
    if ckpt is None: 
        ax.axis('off')
        ax.text(0.5, 0.5, "Missing", ha='center')
        return
    
    spatial_weights = None
    if 'static_norm.gamma' in ckpt:
        weights = ckpt['static_norm.gamma'].abs().numpy()
        if len(weights.shape) == 3: weights = weights[0]
        spatial_weights = weights[:, band_idx]
    elif 'agli.gamma' in ckpt:
        weights = ckpt['agli.gamma'].abs().numpy()
        if len(weights.shape) == 3: weights = weights[0]
        spatial_weights = weights[:, band_idx]
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, "No Spatial Layer", ha='center')
        return

    # Setup MNE
    info = mne.create_info(CHANNEL_NAMES, sfreq=100, ch_types='eeg')
    info_mapped = info.copy()
    mne.rename_channels(info_mapped, MNE_MAPPING)
    montage = mne.channels.make_standard_montage('standard_1020')
    
    try:
        info_mapped.set_montage(montage)
    except Exception:
        pass # Ignore montage errors for grid plotting
    
    # Plot on specific axes
    mne.viz.plot_topomap(spatial_weights, info_mapped, axes=ax, show=False, cmap='Reds', contours=0)
    ax.set_title(f"Sub-{subject_id}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    target_subjects = [4, 5, 6, 7, 8, 10, 11, 12, 13] 
    
    # Calculate Grid Size
    num_plots = len(target_subjects)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    print(f"Generating Aggregated Plots for {num_plots} subjects (Grid: {rows}x{cols})...")

    # ----------------------------------------------------
    # 1. Aggregated Evolution Heatmaps
    # ----------------------------------------------------
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    print("-> Stitching Evolution Heatmaps...")
    for i, sub in enumerate(target_subjects):
        plot_evolution_heatmap_ax(sub, axes[i])
    
    # Hide empty subplots
    for j in range(i+1, len(axes)): axes[j].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "AGGREGATED_Evolution_Heatmaps.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved: {save_path}")

    # ----------------------------------------------------
    # 2. Aggregated Feature Importance Bars
    # ----------------------------------------------------
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    print("-> Stitching Feature Importance Bars...")
    for i, sub in enumerate(target_subjects):
        plot_band_importance_ax(sub, axes[i])

    for j in range(i+1, len(axes)): axes[j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "AGGREGATED_Feature_Importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved: {save_path}")

    # ----------------------------------------------------
    # 3. Aggregated Alpha Band (Topomaps)
    # ----------------------------------------------------
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    print("-> Stitching Alpha Band Topomaps...")
    for i, sub in enumerate(target_subjects):
        plot_brain_map_ax(sub, axes[i], band_idx=2) # Alpha

    for j in range(i+1, len(axes)): axes[j].axis('off')

    plt.suptitle("Alpha Band Spatial Attention", y=0.99)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "AGGREGATED_Alpha_Topomaps.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved: {save_path}")

    # ----------------------------------------------------
    # 4. Aggregated Gamma Band (Topomaps)
    # ----------------------------------------------------
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    print("-> Stitching Gamma Band Topomaps...")
    for i, sub in enumerate(target_subjects):
        plot_brain_map_ax(sub, axes[i], band_idx=4) # Gamma

    for j in range(i+1, len(axes)): axes[j].axis('off')

    plt.suptitle("Gamma Band Spatial Attention", y=0.99)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "AGGREGATED_Gamma_Topomaps.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved: {save_path}")
    
    print("done.")