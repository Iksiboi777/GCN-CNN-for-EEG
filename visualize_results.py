# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import json

# def plot_session_drift_comparison(results_root, subject_id):
#     """
#     Plots the Confusion Matrices of the 3 session holdouts side-by-side.
#     Allows us to see which session has the most 'geometric drift'.
#     """
#     sessions = [1, 2, 3]
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     fig.suptitle(f"Subject {subject_id}: Cross-Session Generalization Drift", fontsize=16)

#     class_names = ['Negative', 'Neutral', 'Positive']

#     for i, test_sess in enumerate(sessions):
#         # Path to the specific permutation results
#         p_tag = f"TestSess_{test_sess}"
#         matrix_path = os.path.join(results_root, p_tag, "predictions.npy")
        
#         if not os.path.exists(matrix_path):
#             print(f"Warning: Matrix for Test Session {test_sess} not found.")
#             axes[i].text(0.5, 0.5, f"Missing Session {test_sess}", ha='center')
#             continue

#         cm = np.load(matrix_path)
#         # Normalize by row (true labels) to show recall percentage
#         cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#         sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', ax=axes[i],
#                     xticklabels=class_names, yticklabels=class_names, cbar=False)
        
#         axes[i].set_title(f"Held-out: Session {test_sess}")
#         axes[i].set_xlabel("Predicted")
#         axes[i].set_ylabel("True")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     save_path = f"Subject_{subject_id}_Session_Drift_Report.png"
#     plt.savefig(save_path)
#     print(f"Drift report saved to {save_path}")
#     plt.show()

# # Example Usage (Run this after train_sage_advanced.py completes):
# plot_session_drift_comparison("Errors/GraphSAGE_DE_1s/Attempt_81_Phase2", subject_id=1)


import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Import the model and utilities from your project
from Models.var_ind_graph import GraphSAGE_EEG_Model
from utils.feature_engineering import get_standard_channel_names

# --- Configuration ---
LOCS_FILE = "utils/channel_62_pos.locs" 
OUTPUT_DIR = "Sage_Viz"
SEARCH_ROOT = "Params" # Look inside this folder
SEARCH_PATTERN = "**/best_model_checkpoint.pth" # Find checkpoints
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_sage_influence(model, coords, channel_names, save_path, label):
    """Visualizes the 'importance' of electrodes after training."""
    model.eval()
    with torch.no_grad():
        try:
            # GraphSAGE's first layer weights: (out_channels, in_channels * 2) usually due to concatenation of self+neighbor
            # We want to see which input nodes (electrodes) have high weight norms.
            # adjusting based on standard PyG implementation or your custom one
            weight = model.sage1.lin_l.weight 
            
            # Calculate importance: L2 norm across output dimensions
            importance = torch.norm(weight, dim=0).cpu().numpy()
            
            # If the weight dimension is larger than channels (e.g. if one-hot encoding or spectral features were inputs per node),
            # we need to be careful. Assuming 1-to-1 mapping roughly or taking the first 62 if raw features.
            # However, usually GraphSAGE takes (N, In_Feats). The weight matrix is (Out_Feats, In_Feats).
            # So 'importance' here shows which *Input Feature* (e.g. Delta band, Theta band) is important, 
            # NOT necessarily which *Channel* is important, unless the graph structure itself is being weighted (GAT).
            
            # NOTE for User: strictly speaking, GraphSAGE weights apply to *features* (bands), not nodes (channels), 
            # because the same weight matrix is shared across all nodes. 
            # To visualize *Node Importance*, we typically look at attention weights (GAT) or 
            # Gradient-based saliency maps. 
            # BUT, since requested, let's try to map this. If features vary per channel, this might just show feature importance.
            
            # If we really want channel importance for SAGE, we often look at the magnitude of activations per channel 
            # over a dataset pass. Since we can't do that without data, we will attempt to interpret 
            # the weights if they are spatially specific (which they usually aren't in shared-weight GNNs).
            #
            # If your model is "var_ind_graph" (Variable Independent), maybe it doesn't share weights?
            # Assuming standard GraphSAGE:
            print(f"  Shape of weights: {weight.shape}")

            # Fallback A: Just plot a dummy heat map if we can't map 1-to-1 to channels
            # because SAGE weights are (Hidden, Input_Features), not (Hidden, Num_Channels).
            # To fix this properly, we need data to run a forward pass and measure activations.
            # However, let's proceed with the plot logic just in case your input dim equals channel count.
            
            if len(importance) != 62:
               print(f"  [Info] Weight dim ({len(importance)}) != Channel count (62). This plot shows Feature Importance, not Spatial.")
               # We can still plot the top features if we want, but let's just create a generic plot
               # indicating this limitation to avoid misleading graphs.
               plt.figure()
               plt.bar(range(len(importance)), importance)
               plt.title(f"Feature Importance (Not Spatial)\n{label}")
               plt.xlabel("Input Feature Index")
               plt.ylabel("Weight Norm")
               plt.savefig(save_path.replace(".png", "_feature_weights.png"))
               plt.close()
               return

            # Normalize
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(coords[:, 0], coords[:, 1], c=importance, cmap='viridis', s=500, edgecolors='k')
            
            for i, name in enumerate(channel_names):
                plt.annotate(name, (coords[i, 0], coords[i, 1]), ha='center', va='center', color='white', fontsize=8)
            
            plt.title(f"Spatial Importance? (Check Dimensions)\n{label}")
            plt.colorbar(label='Normalized Importance')
            plt.savefig(save_path)
            plt.close()
            print(f"Visualization saved to {save_path}")

        except AttributeError:
             print(f"Error: Could not access model.sage1.lin_l.weight for {label}")
        except Exception as e:
            print(f"An error occurred visualizing {label}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    if not os.path.exists(LOCS_FILE):
        print(f"Error: Locations file not found at {LOCS_FILE}")
        return

    coords = np.loadtxt(LOCS_FILE, usecols=(1, 2)) 
    channel_names = get_standard_channel_names()

    # Construct the full search path
    # glob will look like: Params/**/best_model_checkpoint.pth
    full_search_path = os.path.join(SEARCH_ROOT, SEARCH_PATTERN)
    model_files = glob.glob(full_search_path, recursive=True)
    
    if not model_files:
        print(f"No models found matching: {full_search_path}")
        return

    print(f"Found {len(model_files)} models in {SEARCH_ROOT}...")

    for path in model_files:
        # path example: Params\GraphSAGE_DE_4s\Attempt_27_Phase2\best_model_checkpoint.pth
        
        # Get the parts of the path
        parts = os.path.normpath(path).split(os.sep)
        # Assuming structure: Params / ModelName / AttemptName / file
        if len(parts) >= 4:
            model_name = parts[-3] # GraphSAGE_DE_4s
            attempt_name = parts[-2] # Attempt_27_Phase2
            label = f"{model_name}_{attempt_name}"
        else:
            label = os.path.basename(os.path.dirname(path))
        
        print(f"Processing: {label}")
        
        # Initialize model 
        model = GraphSAGE_EEG_Model(
            in_features=10,  # 10 features (5 bands * 2 for DE? or just 5?) check your config!
            hidden_dim=64, 
            aggregator='max' 
        ).to(DEVICE)

        try:
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            
            save_filename = os.path.join(OUTPUT_DIR, f"viz_{label}.png")
            visualize_sage_influence(model, coords, channel_names, save_filename, label)

        except RuntimeError as rt_err:
             print(f"  [Error] Dimension mismatch loading {label}. Check 'in_features'. Error: {rt_err}")
        except Exception as e:
            print(f"  [Error] Failed processing {path}: {e}")

if __name__ == "__main__":
    main()