import torch
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data, Batch

# Import your model definition
from Models.var_B import GCN_DE_Model
from Models.graph_construction import get_knn_adjacency_matrix
from utils.feature_engineering import get_standard_channel_names

# --- CONFIGURATION ---
SUBJECT_ID = 12  # The problem subject
# UPDATE THIS PATH to the exact .pth file of Attempt 47/45
MODEL_PATH = "Params/GCN_DE_1s/Attempt_45_session_holdout_Phase2/best_model_checkpoint.pth" 
DATA_FOLDER = "Data/ExtractedFeatures_1s"
LOCS_FILE = "utils/channel_62_pos.locs"
SAVE_DIR = f"Analysis/Subject_{SUBJECT_ID}_DeepDive" # NEW: Saving directory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
# ---------------------

def compute_rolling_variance(data, window_size=3):
    pad_width = window_size // 2
    padded = np.pad(data, ((0,0), (pad_width, pad_width), (0,0)), mode='edge')
    vars_list = []
    for i in range(data.shape[1]):
        slice_data = padded[:, i : i + window_size, :]
        vars_list.append(np.var(slice_data, axis=1))
    return np.stack(vars_list, axis=1)

def load_subject_12_data():
    """Simplified loader just for Subject 12"""
    print(f"Loading data only for Subject {SUBJECT_ID}...")
    X_list = []
    y_list = []
    
    # Identify Subject 12 files
    files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(f"{SUBJECT_ID}_") and f.endswith('.mat')]
    label_mat = scipy.io.loadmat(os.path.join(DATA_FOLDER, "label.mat"))
    trial_labels = label_mat['label'][0]
    label_map = {-1: 0, 0: 1, 1: 2} # Neg, Neu, Pos

    for fname in files:
        mat = scipy.io.loadmat(os.path.join(DATA_FOLDER, fname))
        for trial_i in range(1, 16):
            key = f"de_LDS{trial_i}"
            if key not in mat: continue
            data = mat[key]
            
            # Shape correction logic (same as train_de.py)
            shape = data.shape
            if shape[0] == 62:
                if shape[1] == 5: data = np.transpose(data, (0, 2, 1))
            elif shape[1] == 62:
                if shape[2] == 5: data = np.transpose(data, (1, 0, 2))
                elif shape[0] == 5: data = np.transpose(data, (1, 2, 0))
            
            # Add Variance Features
            data_var = compute_rolling_variance(data, window_size=3)
            data_combined = np.concatenate([data, data_var], axis=2) # (62, seq, 10)
            data_combined = np.transpose(data_combined, (1, 0, 2))   # (seq, 62, 10)

            X_list.append(data_combined)
            y_list.append(np.full(data_combined.shape[0], label_map[trial_labels[trial_i-1]]))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Normalize (Simple Z-score for visualization purposes)
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    X = (X - mean) / (std + 1e-6)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def visualize_attention():
    # 1. Load Data
    X_raw, y = load_subject_12_data()
    X = X_raw.clone() 
    print(f"Loaded {X.shape[0]} samples for Subject 12.")

    # 2. Load Model
    print("Loading Model...")
    model = GCN_DE_Model(num_nodes=62, in_features=10).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 3. Graph Construction
    edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)

    # 4. Run Inference in batches to collect weights
    batch_size = 64
    all_att_weights = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(DEVICE)
            curr_bs = batch_X.size(0)
            
            # Prepare Batch indices
            batch_idx = torch.arange(curr_bs, device=DEVICE).repeat_interleave(62)
            
            # Prepare Edge Index stacking
            offsets = (torch.arange(curr_bs, device=DEVICE) * 62).view(-1, 1, 1)
            batch_edge_index = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
            # Flatten X for GCN
            batch_X_flat = batch_X.reshape(-1, 10)
            
            # FORWARD PASS WITH ATTENTION
            _, weights = model(batch_X_flat, batch_edge_index, batch_idx, return_attention=True)
            
            # Weights come out as (BatchSize * 62)
            # Reshape to (BatchSize, 62)
            weights = weights.view(curr_bs, 62).cpu().numpy()
            all_att_weights.append(weights)

    # Concatenate all batches
    full_weights = np.concatenate(all_att_weights, axis=0) # Shape: (TotalSamples, 62)
    
    # 5. Plotting
    channel_names = get_standard_channel_names()
    
    # Figure 1: Average Attention per Channel (Bar Plot)
    avg_weights = np.mean(full_weights, axis=0)
    
    # --- 5. USE avg_variance_per_channel: CORRELATION ANALYSIS ---
    # We calculate the variance of the raw DE features (first 5 features) 
    # across the time dimension for Subject 12.
    raw_de_features = X_raw.numpy()[:, :, :5] # (Samples, Channels, Bands)
    # Variance per channel averaged across all samples and bands
    channel_variances = np.mean(np.var(raw_de_features, axis=0), axis=1) 

    # Plot 1: Bar Chart (Focus)
    plt.figure(figsize=(15, 6))
    plt.bar(range(62), avg_weights, color='teal')
    plt.xticks(range(62), channel_names, rotation=90, fontsize=8)
    plt.title(f"Subject {SUBJECT_ID}: Model Focus (Avg Attention Weight)")
    plt.savefig(os.path.join(SAVE_DIR, "attention_bar_chart.png"))
    plt.close()

    # Plot 2: Heatmap (Temporal Stability)
    plt.figure(figsize=(20, 10))
    sns.heatmap(full_weights[:800, :].T, cmap="magma", yticklabels=channel_names)
    plt.title(f"Subject {SUBJECT_ID}: Attention Evolution (First 800 Samples)")
    plt.savefig(os.path.join(SAVE_DIR, "attention_heatmap.png"))
    plt.close()

    # Plot 3: Attention vs Noise (Variance)
    # This identifies if the model is 'sucked in' by high-variance noise
    plt.figure(figsize=(10, 8))
    sns.regplot(x=channel_variances, y=avg_weights)
    for i, txt in enumerate(channel_names):
        if avg_weights[i] > np.percentile(avg_weights, 90) or channel_variances[i] > np.percentile(channel_variances, 90):
            plt.annotate(txt, (channel_variances[i], avg_weights[i]))
    
    plt.title(f"Correlation: Attention vs. Signal Variance (Subj {SUBJECT_ID})")
    plt.xlabel("Physical Variance (Noise Magnitude)")
    plt.ylabel("Attention Weight (Model Importance)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "attention_vs_variance_correlation.png"))
    plt.close()

    print(f"\nAnalysis complete. Plots saved to: {SAVE_DIR}")
    
    # Text Analysis
    top_weighted_idx = np.argsort(avg_weights)[::-1][:5]
    top_variance_idx = np.argsort(channel_variances)[::-1][:5]
    
    print("\n--- Model's Favorite Channels ---")
    for idx in top_weighted_idx:
        print(f"{channel_names[idx]}: Weight {avg_weights[idx]:.4f}")
        
    print("\n--- Highest Variance (Noisiest) Channels ---")
    for idx in top_variance_idx:
        print(f"{channel_names[idx]}: Var {channel_variances[idx]:.4f}")

if __name__ == "__main__":
    visualize_attention()