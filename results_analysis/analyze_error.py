import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from Models.var_A import CNNGCNModel
from Models.graph_construction import get_knn_adjacency_matrix

# Config
DATA_FOLDER = "Data/Raw_Data_For_CNN"
LOCS_FILE = "channel_62_pos.locs"
MODEL_PATH = os.path.join(DATA_FOLDER, "best_model_sub_dep_2.pth") # Adjust if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Loading Data...")
    X = np.load(os.path.join(DATA_FOLDER, "X_raw.npy"))
    y = np.load(os.path.join(DATA_FOLDER, "y_labels.npy"))
    sessions = np.load(os.path.join(DATA_FOLDER, "sessions.npy"))
    subjects = np.load(os.path.join(DATA_FOLDER, "subjects.npy"))

    # Recreate Test Split (Session 3)
    test_mask = (sessions == 3)
    X_test = torch.tensor(X[test_mask], dtype=torch.float32)
    y_test = torch.tensor(y[test_mask], dtype=torch.long)
    
    # Keep track of metadata for the test set
    test_subjects = subjects[test_mask]
    test_sessions = sessions[test_mask]

    print(f"Test Samples: {len(X_test)}")

    # Load Model
    print("Loading Model...")
    edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)
    model = CNNGCNModel(num_nodes=62, time_steps=400).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Run Inference
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(DEVICE)
            batch_idx = torch.arange(batch_X.size(0), device=DEVICE).repeat_interleave(62)
            outputs = model(batch_X, edge_index, batch_idx)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    all_preds = np.array(all_preds)
    y_test_np = y_test.numpy()

    # Identify Errors
    errors = (all_preds != y_test_np)
    error_indices = np.where(errors)[0]
    
    print(f"Total Errors: {len(error_indices)} / {len(y_test)}")
    print(f"Accuracy: {(1 - len(error_indices)/len(y_test))*100:.2f}%")

    # --- Visualization 1: Errors per Subject ---
    error_subjects = test_subjects[error_indices]
    plt.figure(figsize=(10, 5))
    plt.hist(error_subjects, bins=np.arange(1, 17)-0.5, rwidth=0.8, color='red', alpha=0.7)
    plt.title("Number of Errors per Subject (Session 3)")
    plt.xlabel("Subject ID")
    plt.ylabel("Count of Errors")
    plt.xticks(range(1, 16))
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # --- Visualization 2: Error Distribution over Time (First Subject) ---
    # Let's look at Subject 1's errors over time to see if they cluster
    sub1_mask = (test_subjects == 1)
    sub1_errors = errors[sub1_mask]
    
    plt.figure(figsize=(12, 2))
    plt.plot(sub1_errors, color='black', linewidth=0.5)
    plt.title("Error Locations for Subject 1 over Time (Session 3)")
    plt.xlabel("Sample Index")
    plt.ylabel("Error (1=Wrong)")
    plt.yticks([0, 1], labels=['Correct', 'Wrong'])
    plt.show()

if __name__ == "__main__":
    main()