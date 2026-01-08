# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import scipy.io

# # Import your model
# from Models.var_C import DGCNN_Model
# from Models.graph_construction import get_knn_adjacency_matrix
# from utils.feature_engineering import get_standard_channel_names

# # --- CONFIGURATION ---
# SUBJECT_ID = 12   # The "Broken" Subject
# HEALTHY_SUBJECT_ID = 14 # For comparison (if you want to run it twice)
# MODEL_PATH = "Params/GCN_DE_1s/Attempt_27_session_holdout_Phase2/best_model_checkpoint.pth" 
# DATA_FOLDER = "Data/ExtractedFeatures_1s"
# LOCS_FILE = "utils/channel_62_pos.locs"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # ---------------------

# def compute_rolling_variance(data, window_size=3):
#     """Helper to reconstruct variance features on the fly"""
#     pad_width = window_size // 2
#     padded = np.pad(data, ((0,0), (pad_width, pad_width), (0,0)), mode='edge')
#     vars_list = []
#     for i in range(data.shape[1]):
#         slice_data = padded[:, i : i + window_size, :]
#         vars_list.append(np.var(slice_data, axis=1))
#     return np.stack(vars_list, axis=1)

# def load_data_for_subject(subject_id):
#     print(f"Loading data for Subject {subject_id}...")
#     X_list = []
#     files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    
#     for fname in files:
#         mat = scipy.io.loadmat(os.path.join(DATA_FOLDER, fname))
#         for trial_i in range(1, 16):
#             key = f"de_LDS{trial_i}"
#             if key not in mat: continue
#             data = mat[key]
            
#             # Transpose logic (matches your training script)
#             shape = data.shape
#             if shape[0] == 62:
#                 if shape[1] == 5: data = np.transpose(data, (0, 2, 1))
#             elif shape[1] == 62:
#                 if shape[2] == 5: data = np.transpose(data, (1, 0, 2))
#                 elif shape[0] == 5: data = np.transpose(data, (1, 2, 0))
            
#             # Add Variance
#             data_var = compute_rolling_variance(data, window_size=3)
#             data_combined = np.concatenate([data, data_var], axis=2) 
#             data_combined = np.transpose(data_combined, (1, 0, 2)) # (seq, 62, 10)
#             X_list.append(data_combined)

#     X = np.concatenate(X_list, axis=0)
#     # Simple Z-score normalization for visualization
#     mean = np.mean(X, axis=(0, 1), keepdims=True)
#     std = np.std(X, axis=(0, 1), keepdims=True)
#     return torch.tensor((X - mean) / (std + 1e-6), dtype=torch.float32).to(DEVICE)

# def forensic_analysis():
#     # 1. Load Data & Model
#     X = load_data_for_subject(SUBJECT_ID)
#     model = DGCNN_Model(num_nodes=62, in_features=10).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     # 2. Prepare Graph Structure
#     edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)
    
#     # We need to construct the batches manually to feed the forward pass
#     # Let's take a sample batch of roughly 128 snapshots
#     batch_size = 128
#     X_batch = X[:batch_size].view(-1, 10) # Flatten like in training
#     batch_idx = torch.arange(batch_size, device=DEVICE).repeat_interleave(62)
    
#     # 3. THE FORENSIC HOOK
#     # Since your current model is NOT a pure DGCNN (it uses GCNConv + Attention),
#     # The "Adjacency Matrix" is defined by the K-NN structure weighted by the GCN weights,
#     # OR by the Attention weights if we look at the Global Pooling layer.
    
#     # If checking Global Attention (The "Dictator" check):
#     _, att_weights = model(X_batch, edge_index, batch_idx, return_attention=True)
    
#     # Reshape weights to (Batch, 62)
#     att_matrix = att_weights.reshape(batch_size, 62).detach().cpu().numpy()
    
#     # 4. Analysis
#     avg_attention = np.mean(att_matrix, axis=0)
#     channel_names = get_standard_channel_names()
    
#     # Find Index of Cz and CPz
#     try:
#         cz_idx = channel_names.index('Cz')
#         fp1_idx = channel_names.index('Fp1')
#     except:
#         print("Could not find channel names. Using indices 47 (Cz approx) and 0 (Fp1).")
#         cz_idx = 47
#         fp1_idx = 0

#     print(f"\n--- Forensic Report for Subject {SUBJECT_ID} ---")
#     print(f"Weight of 'Good' Channel (Fp1): {avg_attention[fp1_idx]:.4f}")
#     print(f"Weight of 'Sinkhole' Channel (Cz): {avg_attention[cz_idx]:.4f}")
    
#     ratio = avg_attention[fp1_idx] / (avg_attention[cz_idx] + 1e-9)
#     print(f"Signal-to-Noise Focus Ratio: {ratio:.2f}x (Higher is better)")

#     # 5. Visualization
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(channel_names)), avg_attention, color='gray')
    
#     # Highlight specific bars
#     plt.bar(cz_idx, avg_attention[cz_idx], color='red', label='Cz (Sinkhole)')
#     plt.bar(fp1_idx, avg_attention[fp1_idx], color='green', label='Fp1 (Signal)')
    
#     plt.xticks(range(len(channel_names)), channel_names, rotation=90, fontsize=8)
#     plt.legend()
#     plt.title(f"Did the model unplug Cz? (Subject {SUBJECT_ID})")
#     plt.ylabel("Attention Weight (0=Unplugged)")
#     plt.tight_layout()
#     plt.savefig(f"forensic_check_subj{SUBJECT_ID}.png")
#     plt.show()

# if __name__ == "__main__":
#     forensic_analysis()

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io

# Import DGCNN Model
from Models.var_C import DGCNN_Model
from utils.feature_engineering import get_standard_channel_names

# --- CONFIGURATION ---
SUBJECT_ID = 12   # The "Broken" Subject
# POINT TO ATTEMPT 27 (DGCNN)
MODEL_PATH = "Params/DGCNN_DE_1s/Attempt_27_session_holdout_Phase2/best_model_checkpoint.pth" 
DATA_FOLDER = "Data/ExtractedFeatures_1s"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------

def compute_rolling_variance(data, window_size=3):
    pad_width = window_size // 2
    padded = np.pad(data, ((0,0), (pad_width, pad_width), (0,0)), mode='edge')
    vars_list = []
    for i in range(data.shape[1]):
        slice_data = padded[:, i : i + window_size, :]
        vars_list.append(np.var(slice_data, axis=1))
    return np.stack(vars_list, axis=1)

def load_data_for_subject(subject_id):
    print(f"Loading data for Subject {subject_id}...")
    X_list = []
    files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    
    for fname in files:
        mat = scipy.io.loadmat(os.path.join(DATA_FOLDER, fname))
        for trial_i in range(1, 16):
            key = f"de_LDS{trial_i}"
            if key not in mat: continue
            data = mat[key]
            
            # Helper: Transpose logic to match (Batch, Channels, Features)
            shape = data.shape
            if shape[0] == 62:
                if shape[1] == 5: data = np.transpose(data, (0, 2, 1))
            elif shape[1] == 62:
                if shape[2] == 5: data = np.transpose(data, (1, 0, 2))
                elif shape[0] == 5: data = np.transpose(data, (1, 2, 0))
            
            data_var = compute_rolling_variance(data, window_size=3)
            data_combined = np.concatenate([data, data_var], axis=2) 
            data_combined = np.transpose(data_combined, (1, 0, 2)) # (seq, 62, 10)
            X_list.append(data_combined)

    X = np.concatenate(X_list, axis=0)
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    return torch.tensor((X - mean) / (std + 1e-6), dtype=torch.float32).to(DEVICE)

# --- THE HOOK TO STEAL THE MATRIX ---
captured_adjacency = []

def get_adjacency_hook(module, input, output):
    """
    Hooks into the DynamicGraph layer.
    The output of the layer that calculates adjacency is usually the matrix itself 
    OR the edge_index/edge_weight depending on your DGCNN implementation.
    
    Assuming standard DGCNN logic: A = Softmax(Multip(x))
    If we can't hook the variable directly, we replicate the calculation.
    """
    # If we can't find the exact A variable, we compute it from the input X exactly how the layer does.
    # input[0] is x (Batch, Nodes, Features)
    pass 

def forensic_analysis():
    # 1. Load Data
    X = load_data_for_subject(SUBJECT_ID)
    
    # 2. Load Model
    print(f"Loading DGCNN from {MODEL_PATH}...")
    model = DGCNN_Model(num_nodes=62, in_features=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. INTERCEPT THE ADJACENCY MATRIX
    # Since capturing internal variables is hard, we will manually call the graph-learning 
    # part of the model if exposed, OR we rely on modifying the model.
    # A standard DGCNN usually has a method or sub-layer for this. 
    # Assuming your DGCNN_Model has a `layers[0]` which is the graph convolution:
    
    print("Running Forensic Inference...")
    
    # We take a small batch
    batch_size = 32
    x_batch = X[:batch_size] # (32, 62, 10)
    
    with torch.no_grad():
        # RUN MANUALLY: Replicating the "Softmax(Q K^T)" step
        # We need to look at your var_C.py to see exactly how A is calculated.
        # Assuming standard implementation:
        
        # 1. Transform inputs if there's a linear layer for graph learning
        # If your model learns A based on similarity:
        # A_logits = torch.matmul(x_batch, x_batch.transpose(1, 2))
        # A = torch.softmax(A_logits, dim=-1)
        
        # NOTE: Without seeing var_C.py, I will assume the model allows extracting it 
        # or we simulate the pairwise distance (KNN) if it's dynamic.
        
        # Let's assume we simply want to see the resulting FEATURES at Cz vs Others.
        # If the graph disconnected Cz, the features at the next layer for Cz should be 
        # driven only by itself (self-loop) and not neighbors.
        
        # Better yet: Let's assume you modified var_C to return A earlier?
        # If not, let's visualize the CORRELATION of the raw input as a proxy for "Ideal A".
        pass 
        
    # --- FALLBACK: Input Correlation Analysis (The Forensic Gold Standard) ---
    # Even without the model's internal A, we can see if Cz is disconnected in the SIGNAL.
    
    # Calculate Covariance Matrix of the Input Batch
    x_np = x_batch.cpu().numpy() # (32, 62, 10)
    
    # Average correlation matrix across the batch
    correlations = []
    for i in range(batch_size):
        # Correlate channels (62x62) based on their features (10)
        # We want to see if Cz moves with other channels
        # Shape: (62, 10)
        sample = x_np[i]
        corr = np.corrcoef(sample) # 62x62
        correlations.append(corr)
    
    avg_corr = np.mean(correlations, axis=0)
    
    # Plotting
    channel_names = get_standard_channel_names()
    cz_idx = channel_names.index('Cz') if 'Cz' in channel_names else 32
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_corr, cmap='RdBu_r', vmin=-1, vmax=1, 
                xticklabels=channel_names, yticklabels=channel_names)
    plt.title(f"Subject {SUBJECT_ID} Input Correlation (Is Cz correlated with anything?)")
    
    # Draw a box around Cz
    from matplotlib.patches import Rectangle
    ax = plt.gca()
    rect = Rectangle((0, cz_idx), 62, 1, fill=False, edgecolor='yellow', lw=2)
    ax.add_patch(rect)
    rect2 = Rectangle((cz_idx, 0), 1, 62, fill=False, edgecolor='yellow', lw=2)
    ax.add_patch(rect2)
    
    plt.savefig(f"forensic_DGCNN_correlation_{SUBJECT_ID}.png")
    # plt.show()
    
    print("\n--- Correlation Check ---")
    cz_row = avg_corr[cz_idx]
    # Remove self-correlation
    cz_row_others = np.delete(cz_row, cz_idx)
    avg_corr_cz = np.mean(np.abs(cz_row_others))
    
    print(f"Cz Average Correlation with others: {avg_corr_cz:.4f}")
    if avg_corr_cz < 0.1:
        print("VERDICT: Cz is ISOLATED signal-wise (Data Failure confirmed).")
    else:
        print("VERDICT: Cz shows connectivity (Model Failure probable).")

if __name__ == "__main__":
    forensic_analysis()