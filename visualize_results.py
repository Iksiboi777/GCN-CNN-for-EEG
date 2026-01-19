import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy.io
from torch_geometric.data import Data, Batch
from scipy.spatial import distance_matrix

# Import the model and utilities
from Models.var_ind_graph import GraphSAGE_EEG_Model
from utils.feature_engineering import get_standard_channel_names

# --- Configuration ---
LOCS_FILE = "utils/channel_62_pos.locs" 
OUTPUT_DIR = "Sage_Viz"
SEARCH_ROOT = "Params" 
SEARCH_PATTERN = "**/best_model_checkpoint.pth" 
DATA_DIR = os.path.join("Data", "ExtractedFeatures_1s") # Pointing to your specific folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_knn_graph(coords, k=5):
    """
    Reconstructs an approximate graph structure based on electrode distance.
    This is necessary for GraphSAGE to pass messages.
    """
    dist_mat = distance_matrix(coords, coords)
    edge_index = []
    
    # Iterate over each node to find k nearest neighbors
    for i in range(len(coords)):
        # Argsort gets indices of sorted distances. 
        # [1:k+1] skips the node itself (dist=0)
        neighbors = np.argsort(dist_mat[i])[1:k+1]
        for n in neighbors:
            edge_index.append([i, n]) # Source -> Target
            edge_index.append([n, i]) # Target -> Source (Undirected)
            
    edges = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edges.to(DEVICE)

def get_real_seed_sample(coords, required_dim=10):
    """
    Loads a .mat file from ExtractedFeatures_4s to get a real brain state vector.
    """
    # 1. Find a .mat file
    mat_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not mat_files:
        print(f"  [Warn] No .mat files found in {DATA_DIR}. Using random noise.")
        return None
    
    # 2. Load the first file (e.g., Subject 1, Session 1)
    target_file = mat_files[0]
    try:
        mat_data = scipy.io.loadmat(target_file)
    except Exception as e:
        print(f"  [Warn] Failed to load {target_file}: {e}")
        return None

    # 3. Find a data key (e.g., 'de_LDS1', 'de_movingAve1')
    data_key = None
    for key in mat_data.keys():
        if key.startswith('de_') or key.startswith('psd_'):
            data_key = key
            break
            
    if data_key is None:
        print(f"  [Warn] No 'de_' or 'psd_' keys found in {os.path.basename(target_file)}")
        return None

    # 4. Extract Data: Shape is usually [Channels, Samples, Bands] or [Channels, Bands, Samples]
    # Standard SEED extracted features: (62, Time, 5)
    raw_cube = mat_data[data_key]
    
    # Basic Transpose check: we want (62, ...)
    if raw_cube.shape[0] != 62 and raw_cube.shape[1] == 62:
        raw_cube = np.swapaxes(raw_cube, 0, 1)

    # Pick a random time window (e.g., index 10)
    time_idx = min(10, raw_cube.shape[1] - 1)
    x_sample = raw_cube[:, time_idx, :] # Shape: (62, 5)
    
    # 5. Handle Dimension Mismatch (5 vs 10)
    # If model needs 10 but we have 5 (Standard DE w/o PSD)
    if x_sample.shape[1] == 5 and required_dim == 10:
        # print("  [Info] Data has 5 features. Model needs 10. Concatenating [DE, DE] for visualization.")
        x_sample = np.concatenate([x_sample, x_sample], axis=1) # Naive fix
    elif x_sample.shape[1] != required_dim:
        print(f"  [Warn] Feature dimension mismatch: Model({required_dim}) vs Data({x_sample.shape[1]}).")
        # Proceeding anyway usually crashes, so let's try to pad or cut
        if x_sample.shape[1] > required_dim:
            x_sample = x_sample[:, :required_dim]
        else:
            pad = np.zeros((62, required_dim - x_sample.shape[1]))
            x_sample = np.concatenate([x_sample, pad], axis=1)

    # 6. Build Torch Object
    x_tensor = torch.tensor(x_sample, dtype=torch.float).to(DEVICE)
    edge_index = build_knn_graph(coords)
    batch = torch.zeros(len(coords), dtype=torch.long, device=DEVICE)
    
    return Batch(x=x_tensor, edge_index=edge_index, batch=batch)

def compute_spatial_saliency(model, data):
    """
    Computes loss gradient w.r.t input nodes.
    """
    model.eval()
    data.x.requires_grad = True
    
    try:
        # Forward
        if hasattr(data, 'batch'):
            out = model(data.x, data.edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index, None)
            
        # Target: Max class
        score, _ = torch.max(out, 1)
        score.backward()
        
        # Saliency: Sum of absolute gradients across features per node
        saliency = data.x.grad.abs().sum(dim=1).cpu().numpy()
        return saliency
        
    except Exception as e:
        print(f"  [Error] Saliency computation failed: {e}")
        return None

def visualize_sage_influence(model, coords, channel_names, save_path, label, required_in_ch):
    """Generates Spatial Saliency (Scalp) and Feature Weights (Bar)."""
    
    # --- 1. SPATIAL IMPORTANCE (Real Data Saliency) ---
    data = get_real_seed_sample(coords, required_dim=required_in_ch)
    
    # Fallback to noise if data loading failed
    if data is None:
        # print("  [Info] Using random noise for structural check.")
        x = torch.randn((len(coords), required_in_ch), requires_grad=True, device=DEVICE)
        edge_index = build_knn_graph(coords)
        batch = torch.zeros(len(coords), dtype=torch.long, device=DEVICE)
        data = Batch(x=x, edge_index=edge_index, batch=batch)

    saliency = compute_spatial_saliency(model, data)
    
    if saliency is not None:
        # Normalize for heatmap
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], c=saliency, cmap='jet', s=600, edgecolors='k')
        
        for i, name in enumerate(channel_names):
            plt.annotate(name, (coords[i, 0], coords[i, 1]), ha='center', va='center', color='white', fontsize=8, weight='bold')
        
        plt.title(f"GraphSAGE Saliency Map\n(High Value = High Impact on Emotion Prediction)\n{label}")
        plt.colorbar(label='Gradient Magnitude')
        plt.axis('off') # Remove axis box for cleaner look
        plt.savefig(save_path)
        plt.close()
        print(f"  -> Saliency Map saved to {save_path}")

    # --- 2. FEATURE IMPORTANCE (Model Weights) ---
    try:
        weight = model.sage1.lin_l.weight.detach().cpu().numpy() # [Out_Dim, In_Features]
        feature_importance = np.linalg.norm(weight, axis=0) # [In_Features]
        
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(feature_importance)), feature_importance, color='skyblue', edgecolor='black')
        plt.title(f"Feature Band Importance\n{label}")
        plt.xlabel("Input Dimension Index")
        plt.ylabel("Weight Norm")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", "_features.png"))
        plt.close()
        # print(f"  -> Feature weights saved.")
    except:
        pass

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(LOCS_FILE):
        print(f"Error: Locations file not found at {LOCS_FILE}")
        return

    # Load locations
    # Format: [Number, X, Y, Label] -> We need cols 1,2
    coords = np.loadtxt(LOCS_FILE, usecols=(1, 2)) 
    channel_names = get_standard_channel_names()

    full_search_path = os.path.join(SEARCH_ROOT, SEARCH_PATTERN)
    model_files = glob.glob(full_search_path, recursive=True)
    
    if not model_files:
        print(f"No models found matching: {full_search_path}")
        return

    print(f"Found {len(model_files)} models. Generating visualizations...")

    for path in model_files:
        path = os.path.normpath(path)
        parts = path.split(os.sep)
        
        # Heuristic Labeling
        if len(parts) >= 4:
            model_name = parts[-3] 
            attempt_name = parts[-2]
            label = f"{model_name}_{attempt_name}"
        else:
            label = os.path.basename(os.path.dirname(path))
        
        print(f"Processing: {label}")
        
        # --- Check Model Config ---
        # Heuristic: Try to determine if it's 1s or 4s or specific DE model
        # Defaulting to 10 features as per user context, but try/except handles mismatches
        in_feats = 10 
        
        try:
            model = GraphSAGE_EEG_Model(in_features=in_feats, hidden_dim=64, aggregator='max').to(DEVICE)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            
            save_filename = os.path.join(OUTPUT_DIR, f"Saliency_{label}.png")
            visualize_sage_influence(model, coords, channel_names, save_filename, label, in_feats)

        except RuntimeError as rt_err:
             print(f"  [Error] Dimension mismatch (likely in_features != 10). Details: {rt_err}")
        except Exception as e:
            print(f"  [Error] Failed processing {path}: {e}")

if __name__ == "__main__":
    main()