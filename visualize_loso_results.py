import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
MODEL_NAME = "GCN_DE_4s" 
ATTEMPT_ID = "Attempt_54_LOSO_Parallel" 

RESULTS_DIR = f"Results/{MODEL_NAME}/{ATTEMPT_ID}"
PARAMS_DIR = f"Params/{MODEL_NAME}/{ATTEMPT_ID}"

# CORRECTED CHANNEL LIST (From ICA_for_SEED.py)
# This order matches the index 0-61 in your model's weight matrix.
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

# Mapping SEED-specific names to Standard 10-20 for plotting
# Note: Fp1, Fz, etc. are already standard, so we don't need to map them anymore.
MNE_MAPPING = {
    'CB1': 'PO9',  # Approximate location for Cerebellar Left
    'CB2': 'PO10', # Approximate location for Cerebellar Right
}

def load_model_weights(subject_id):
    path = os.path.join(PARAMS_DIR, f"Subject_{subject_id}", "best_model_checkpoint.pth")
    if not os.path.exists(path):
        print(f"[Warn] No checkpoint found for Subject {subject_id}")
        return None
    return torch.load(path, map_location='cpu')

# ==========================================
# 1. CLASS EVOLUTION HEATMAP
# ==========================================
def plot_evolution_heatmap(subject_id):
    path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "evolution_history.npy")
    
    if not os.path.exists(path):
        print(f"[Skip] No evolution history found for Subject {subject_id}")
        return

    print(f"Plotting Evolution Heatmap for Subject {subject_id}...")
    data = np.load(path, allow_pickle=True).item()
    
    true_labels = data['true_labels']
    preds_history = data['preds_history']
    
    epochs = np.arange(1, len(preds_history) + 1)
    classes = ['Negative', 'Neutral', 'Positive']
    heatmap_matrix = np.zeros((3, len(preds_history)))

    for i, preds_epoch in enumerate(preds_history):
        # Safety check for length mismatch
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

    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=epochs, yticklabels=classes, vmin=0, vmax=1)
    plt.title(f"Subject {subject_id}: Validation Accuracy Evolution")
    plt.xlabel("Snapshot Index")
    plt.ylabel("Emotion Class")
    plt.tight_layout()
    # plt.show()

    save_path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "evolution_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")

# ==========================================
# 2. FEATURE IMPORTANCE (Band Analysis)
# ==========================================
def plot_band_importance(subject_id):
    ckpt = load_model_weights(subject_id)
    if ckpt is None: return

    print(f"Analyzing Band Importance for Subject {subject_id}...")
    
    weights = None
    if 'static_norm.gamma' in ckpt:
        weights = ckpt['static_norm.gamma'].abs().numpy()
        if len(weights.shape) == 3: weights = weights[0]
        importance = np.mean(weights, axis=0) # (Feat,)
    elif 'conv1.lin.weight' in ckpt:
        # For standard GCN, approximate importance via input layer weights
        # weight shape (Out, In). We sum abs weights for each input feature.
        weights = ckpt['conv1.lin.weight'].abs().numpy()
        importance = np.mean(weights, axis=0) # (In,)
    else:
        print("  [Info] No explicit input weight layer found (Standard GraphSAGE/GCN hidden).")
        return

    count = len(importance)
    plt.figure(figsize=(10, 5))
    
    if count == 5:
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        plt.bar(bands, importance, color='steelblue')
    elif count == 10:
        # Assuming first 5 are DE, next 5 are Rolling Var or similar
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        x = np.arange(5)
        width = 0.35
        plt.bar(x - width/2, importance[:5], width, label='DE Power', color='steelblue')
        plt.bar(x + width/2, importance[5:], width, label='Feature Type 2', color='salmon')
        plt.xticks(x, bands)
        plt.legend()
    else:
        plt.bar(range(count), importance)

    plt.ylabel("Weight Magnitude")
    plt.title(f"Subject {subject_id}: Feature Importance")
    plt.tight_layout()
    # plt.show()

    save_path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")

# ==========================================
# 3. SPATIAL TOPOMAP (The Brain Map)
# ==========================================
def plot_brain_map(subject_id, band_idx=2): # 2 = Alpha
    ckpt = load_model_weights(subject_id)
    if ckpt is None: return
    
    spatial_weights = None
    
    # 1. Extract Weights
    if 'static_norm.gamma' in ckpt:
        weights = ckpt['static_norm.gamma'].abs().numpy()
        if len(weights.shape) == 3: weights = weights[0]
        spatial_weights = weights[:, band_idx]
    elif 'agli.gamma' in ckpt:
        weights = ckpt['agli.gamma'].abs().numpy()
        if len(weights.shape) == 3: weights = weights[0]
        spatial_weights = weights[:, band_idx]
    else:
        print("  [Skip] Brain Map requires an architectural component with node-specific weights.")
        return

    print(f"Plotting Topomap for Subject {subject_id} (Band Index {band_idx})...")

    # 2. Setup MNE with Correct Names
    info = mne.create_info(CHANNEL_NAMES, sfreq=100, ch_types='eeg')
    
    # 3. Map CB1/CB2 to standard names
    info_mapped = info.copy()
    mne.rename_channels(info_mapped, MNE_MAPPING) # CB1->PO9, CB2->PO10
    
    # 4. Apply Montage
    montage = mne.channels.make_standard_montage('standard_1020')
    try:
        info_mapped.set_montage(montage)
    except ValueError as e:
        print(f"  [Error] Montage mapping failed: {e}")
        return

    # 5. Plot
    plt.figure(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(spatial_weights, info_mapped, show=False, cmap='Reds', contours=0)
    
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    b_name = band_names[band_idx] if band_idx < 5 else f"Feat_{band_idx}"
    
    plt.title(f"Subject {subject_id} Spatial Attention\n({b_name} Band)")
    plt.colorbar(im, label="Importance")
    # plt.show()

    save_path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", f"brain_map_{b_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Analyze a specific problematic subject
    target_subjects = [1, 4, 6, 7, 8, 10, 11, 12, 13] 

    for sub in target_subjects:
        print(f"\n--- ANALYZING SUBJECT {sub} ---")
        plot_evolution_heatmap(sub)
        plot_band_importance(sub)
        plot_brain_map(sub, band_idx=2) # Alpha
        plot_brain_map(sub, band_idx=4) # Gamma


# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import mne
# from sklearn.metrics import accuracy_score

# # --- CONFIGURATION ---
# MODEL_NAME = "GCN_DE_4s"  # Adjust to match your folder
# ATTEMPT_ID = "Attempt_56_LOSO_Parallel" # Adjust to match your folder

# RESULTS_DIR = f"Results/{MODEL_NAME}/{ATTEMPT_ID}"
# PARAMS_DIR = f"Params/{MODEL_NAME}/{ATTEMPT_ID}"

# # SEED Channel Names (Standard 62)
# CHANNEL_NAMES = [
#     'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
#     'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#     'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
#     'PO6', 'PO8', 'CB1', 'CB2', 'O1', 'OZ', 'O2'
# ]

# # Mapping SEED names to Standard 10-20 for MNE Plotting
# MNE_MAPPING = {
#     'FPZ': 'Fpz', 'FCZ': 'FCz', 'CPZ': 'CPz', 'POZ': 'POz', 'OZ': 'Oz', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
#     'CB1': 'PO9', 'CB2': 'PO10' # Approximation for Cerebellar
# }

# def load_model_weights(subject_id):
#     path = os.path.join(PARAMS_DIR, f"Subject_{subject_id}", "best_model_checkpoint.pth")
#     if not os.path.exists(path):
#         print(f"[Warn] No checkpoint found for Subject {subject_id}")
#         return None
#     return torch.load(path, map_location='cpu')

# # ==========================================
# # 1. CLASS EVOLUTION HEATMAP (Spectral Theft Detector)
# # ==========================================
# def plot_evolution_heatmap(subject_id):
#     path = os.path.join(RESULTS_DIR, f"Subject_{subject_id}", "evolution_history.npy")
    
#     if not os.path.exists(path):
#         print(f"[Skip] No evolution history found for Subject {subject_id}")
#         return

#     print(f"Plotting Evolution Heatmap for Subject {subject_id}...")
#     data = np.load(path, allow_pickle=True).item()
    
#     # Check if we have the right keys
#     if 'preds_history' not in data or 'true_labels' not in data:
#         print("  [Error] History file keys mismatch.")
#         return

#     true_labels = data['true_labels']
#     preds_history = data['preds_history']
    
#     # Dynamic Epoch Labeling
#     num_snapshots = len(preds_history)
#     # Assuming snapshots were taken every X epochs, we just label them 1 to N
#     epochs = np.arange(1, num_snapshots + 1)
    
#     classes = ['Negative', 'Neutral', 'Positive']
#     heatmap_matrix = np.zeros((3, num_snapshots))

#     for i, preds_epoch in enumerate(preds_history):
#         # Handle case where history might be different length than labels (rare bug)
#         if len(preds_epoch) != len(true_labels):
#             min_len = min(len(preds_epoch), len(true_labels))
#             preds_epoch = preds_epoch[:min_len]
#             true_labels_curr = true_labels[:min_len]
#         else:
#             true_labels_curr = true_labels

#         for cls_idx in range(3):
#             mask = (true_labels_curr == cls_idx)
#             if np.sum(mask) > 0:
#                 acc = accuracy_score(true_labels_curr[mask], preds_epoch[mask])
#                 heatmap_matrix[cls_idx, i] = acc
#             else:
#                 heatmap_matrix[cls_idx, i] = 0.0

#     plt.figure(figsize=(10, 5))
#     sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
#                 xticklabels=epochs, yticklabels=classes, vmin=0, vmax=1)
#     plt.title(f"Subject {subject_id}: Validation Accuracy Evolution")
#     plt.xlabel("Snapshot Index (Time ->)")
#     plt.ylabel("Emotion Class")
#     plt.tight_layout()
#     # plt.show()

# # ==========================================
# # 2. FEATURE IMPORTANCE (Band Analysis)
# # ==========================================
# def plot_band_importance(subject_id):
#     ckpt = load_model_weights(subject_id)
#     if ckpt is None: return

#     print(f"Analyzing Band Importance for Subject {subject_id}...")
    
#     # Try to find learnable feature weights
#     weights = None
    
#     # Case A: Adaptive DGCNN (static_norm.gamma)
#     if 'static_norm.gamma' in ckpt:
#         weights = ckpt['static_norm.gamma'].abs().numpy() # Shape (1, 62, 10) or (62, 10)
#         if len(weights.shape) == 3: weights = weights[0]
#         # Average across nodes -> (10,)
#         importance = np.mean(weights, axis=0) 
        
#     # Case B: Standard Linear/GCN (Check first linear layer)
#     elif 'conv1.lin.weight' in ckpt:
#         # GCN usually aggregates neighbors. Input importance is harder to see directly here.
#         # We look at the col-norm of the first weight matrix (Out, In)
#         w = ckpt['conv1.lin.weight'].abs().numpy()
#         importance = np.mean(w, axis=0) # (In_Features,)
#     else:
#         print("  [Info] Model architecture does not expose explicit input weights (e.g., pure GCN without Adaptive Layer). Skipping.")
#         return

#     # Check Dimensions (5 vs 10)
#     count = len(importance)
#     if count == 5:
#         bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
#         colors = ['steelblue'] * 5
#     elif count == 10:
#         bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
#         colors = ['steelblue'] * 5 + ['salmon'] * 5
#         importance_reshaped = importance
#         x_labels = bands + [f"Var_{b}" for b in bands]
#     else:
#         x_labels = [str(i) for i in range(count)]
#         colors = 'skyblue'

#     plt.figure(figsize=(10, 5))
#     bars = plt.bar(range(count), importance, color=colors)
    
#     if count == 10:
#         plt.xticks(range(count), x_labels, rotation=45)
#         plt.legend([bars[0], bars[5]], ['DE Power', 'Rolling Var'])
#     else:
#         plt.xticks(range(count), bands)

#     plt.ylabel("Weight Magnitude")
#     plt.title(f"Subject {subject_id}: Feature Band Importance")
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     # plt.show()

# # ==========================================
# # 3. SPATIAL TOPOMAP (The Brain Map)
# # ==========================================
# def plot_brain_map(subject_id, band_idx=2): # 2 = Alpha
#     ckpt = load_model_weights(subject_id)
#     if ckpt is None: return
    
#     # We strictly need node-wise weights here. 
#     # Only available if the model has a component that weights nodes/channels individually.
#     spatial_weights = None
    
#     if 'static_norm.gamma' in ckpt:
#         weights = ckpt['static_norm.gamma'].abs().numpy()
#         if len(weights.shape) == 3: weights = weights[0]
#         # Extract specific band column
#         spatial_weights = weights[:, band_idx]
#     elif 'agli.gamma' in ckpt:
#         weights = ckpt['agli.gamma'].abs().numpy()
#         spatial_weights = weights[:, band_idx]
#     else:
#         print("  [skip] Standard GCN treats all nodes equally mostly; no explicit spatial weight layer found.")
#         return

#     print(f"Plotting Topomap for Subject {subject_id} (Band Index {band_idx})...")

#     # MNE Setup
#     # 1. Create Info
#     info = mne.create_info(CHANNEL_NAMES, sfreq=100, ch_types='eeg')
    
#     # 2. Set Montage (with fixing names)
#     montage = mne.channels.make_standard_montage('standard_1020')
    
#     # 3. Rename channels in info to match 10-20
#     mne.rename_channels(info, MNE_MAPPING)
    
#     # 4. Apply montage
#     # Filter montage to only keep channels we actually have
#     try:
#         info.set_montage(montage)
#     except ValueError:
#         print("  [Warn] Some channels could not be mapped to standard 10-20. Plot might be incomplete.")

#     # Plot
#     plt.figure(figsize=(6, 6))
#     im, _ = mne.viz.plot_topomap(spatial_weights, info, show=False, cmap='Reds', contours=0)
    
#     band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
#     b_name = band_names[band_idx] if band_idx < 5 else f"Feature_{band_idx}"
    
#     plt.title(f"Subject {subject_id} Spatial Attention\n({b_name} Band)")
#     plt.colorbar(im, label="Importance")
#     # plt.show()

# # ==========================================
# # MAIN EXECUTION
# # ==========================================
# if __name__ == "__main__":
#     # Suggested Loop: Analyze specific interesting subjects
#     # Subject 2 (Bad Performer), Subject 6 (Good Performer)
#     target_subjects = [2] 

#     for sub in target_subjects:
#         print(f"\n--- ANALYZING SUBJECT {sub} ---")
#         plot_evolution_heatmap(sub)
#         plot_band_importance(sub)
#         plot_brain_map(sub, band_idx=2) # Alpha
#         plot_brain_map(sub, band_idx=4) # Gamma