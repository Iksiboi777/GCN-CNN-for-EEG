# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# from Models.var_ind_graph import GraphSAGE_EEG_Model
# from utils.feature_engineering import get_standard_channel_names

# # --- Config ---
# SEARCH_ROOT = "Params"
# SEARCH_PATTERN = "**/best_model_checkpoint.pth"
# OUTPUT_DIR = "Sage_Viz"
# LOCS_FILE = "utils/channel_62_pos.locs"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def diagnose_model(model_path, save_prefix):
#     print(f"Diagnosing: {model_path}")
    
#     # 1. Load Model
#     # Note: Adjust 'in_features' if your model differs (e.g. 5 vs 10)
#     # Using try-except block to handle shape mismatches
#     try:
#         model = GraphSAGE_EEG_Model(in_features=10, hidden_dim=64, aggregator='max').to(DEVICE)
#         state_dict = torch.load(model_path, map_location=DEVICE)
#         model.load_state_dict(state_dict)
#     except RuntimeError:
#         try:
#              # Fallback for 5 features
#             model = GraphSAGE_EEG_Model(in_features=5, hidden_dim=64, aggregator='max').to(DEVICE)
#             state_dict = torch.load(model_path, map_location=DEVICE)
#             model.load_state_dict(state_dict)
#         except Exception as e:
#             print(f"Skipping {model_path}: {e}")
#             return

#     model.eval()

#     # Data containers
#     sensor_importance = None
#     band_importance = None
#     diag_success = False

#     with torch.no_grad():
#         # --- Check for AGLI (Adaptive Graph Learning) ---
#         if hasattr(model, 'agli') and hasattr(model.agli, 'gamma'):
#             # gamma shape: (1, 62, 10) or similar. 
#             gamma = model.agli.gamma.squeeze(0).cpu().numpy() 
#             # Mean across features to get sensor health
#             if len(gamma.shape) > 1:
#                 sensor_importance = np.mean(np.abs(gamma), axis=1)
#             else:
#                 sensor_importance = np.abs(gamma)
#             diag_success = True
        
#         # --- Check for SE Block (Squeeze-Excitation) ---
#         if hasattr(model, 'se_block'):
#             # Look at FC weights in SE block
#             # Usually: model.se_block.fc contains Linear layers
#             # We try to get weights from the first or last linear layer
#             for layer in model.se_block.modules():
#                 if isinstance(layer, torch.nn.Linear):
#                     w = layer.weight.detach().cpu().numpy()
#                     # Just taking mean importance across output neurons for input features
#                     band_importance = np.mean(np.abs(w), axis=0)
#                     diag_success = True
#                     break
        
#         # --- Fallback: Standard GraphSAGE Weighs ---
#         if not diag_success and hasattr(model, 'sage1'):
#             print("  Custom blocks (AGLI/SE) not found. Plotting standard SAGE weights.")
#             w = model.sage1.lin_l.weight.detach().cpu().numpy() # [Out, In]
#             band_importance = np.mean(np.abs(w), axis=0)
#             # Cannot infer sensor importance from standard SAGE weights easily without data

#     # --- Visualization ---
#     if not os.path.exists(LOCS_FILE):
#         print("Locs file not found.")
#         return

#     coords = np.loadtxt(LOCS_FILE, usecols=(1, 2))
#     names = get_standard_channel_names()
    
#     plt.figure(figsize=(12, 6))

#     # Plot 1: Sensor Map (if AGLI exists)
#     if sensor_importance is not None:
#         if len(sensor_importance) == len(coords):
#             plt.subplot(1, 2, 1)
#             # Normalize
#             imp_norm = (sensor_importance - sensor_importance.min()) / (sensor_importance.max() - sensor_importance.min() + 1e-8)
#             sc = plt.scatter(coords[:, 0], coords[:, 1], c=imp_norm, cmap='viridis', s=200, edgecolors='k')
#             plt.colorbar(sc, label="Importance (Norm)")
#             for i, txt in enumerate(names):
#                 plt.annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=7, ha='center', color='white')
#             plt.title(f"Sensor Importance (AGLI)\n{save_prefix}")
#         else:
#             print(f"  Dimension mismatch: Data {len(sensor_importance)} vs Locs {len(coords)}")

#     # Plot 2: Band Importance
#     if band_importance is not None:
#         plt.subplot(1, 2, 2)
#         n_bands = len(band_importance)
#         # Create labels based on size
#         if n_bands == 10:
#             labels = ['D', 'T', 'A', 'B', 'G', 'vD', 'vT', 'vA', 'vB', 'vG']
#         elif n_bands == 5:
#              labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
#         else:
#             labels = [str(i) for i in range(n_bands)]
            
#         plt.bar(labels, band_importance, color='skyblue', edgecolor='black')
#         plt.ylabel("Weight Magnitude")
#         plt.title(f"Feature Importance\n{save_prefix}")

#     if sensor_importance is not None or band_importance is not None:
#         save_path = os.path.join(OUTPUT_DIR, f"Diagnosis_{save_prefix}.png")
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close()
#         print(f"  Saved to {save_path}")
#     else:
#         print("  Nothing to visualize for this architecture.")

# def main():
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     # Recursive search for models
#     full_search_path = os.path.join(SEARCH_ROOT, SEARCH_PATTERN)
#     model_files = glob.glob(full_search_path, recursive=True)
    
#     if not model_files:
#         print(f"No models found in {full_search_path}")
#         return

#     print(f"Found {len(model_files)} models.")

#     for path in model_files:
#         # Create a readable label from path
#         parts = os.path.normpath(path).split(os.sep)
#         if len(parts) >= 3:
#             # e.g., GraphSAGE_DE_4s_Attempt_27
#             label = f"{parts[-3]}_{parts[-2]}"
#         else:
#             label = "Unknown_Model"
            
#         diagnose_model(path, label)

# if __name__ == "__main__":
#     main()



import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def plot_session_drift_comparison(results_root, subject_id):
    """
    Plots the Confusion Matrices of the 3 session holdouts side-by-side.
    Allows us to see which session has the most 'geometric drift'.
    """
    sessions = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Subject {subject_id}: Cross-Session Generalization Drift", fontsize=16)

    class_names = ['Negative', 'Neutral', 'Positive']

    for i, test_sess in enumerate(sessions):
        # Path to the specific permutation results
        p_tag = f"TestSess_{test_sess}"
        matrix_path = os.path.join(results_root, p_tag, "predictions.npy")
        
        if not os.path.exists(matrix_path):
            print(f"Warning: Matrix for Test Session {test_sess} not found.")
            axes[i].text(0.5, 0.5, f"Missing Session {test_sess}", ha='center')
            continue

        cm = np.load(matrix_path)
        # Normalize by row (true labels) to show recall percentage
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        
        axes[i].set_title(f"Held-out: Session {test_sess}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"Subject_{subject_id}_Session_Drift_Report.png"
    plt.savefig(save_path)
    print(f"Drift report saved to {save_path}")
    plt.show()

# Example Usage (Run this after train_sage_advanced.py completes):
plot_session_drift_comparison("Errors/GraphSAGE_Advanced_1s/Attempt_5_Phase2", subject_id=1)