import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, add_self_loops

# --- 1. Split Data into Train and Test Sets ---
# This is CRITICAL. We build the graph *only* from training data.
print("Splitting features into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_features, 
    y_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_labels # Good practice for balanced splits
)

print(f"X_train shape: {X_train.shape}") # (19872, 14, 5)
print(f"X_test shape: {X_test.shape}")   # (4968, 14, 5)

# --- 2. Calculate the Pearson Correlation (PCC) Graph ---
# We compute the correlation between the 14 channels.
print("Calculating Pearson Correlation (PCC) matrix from training data...")

# We have (Samples, Channels, Features). We need to correlate channels.
# We'll reshape to (Samples * Features, Channels)
# (19872, 14, 5) -> (19872, 5, 14) -> (19872 * 5, 14)
num_samples, num_channels, num_features = X_train.shape
reshaped_features = X_train.transpose(0, 2, 1).reshape(-1, num_channels)
print(f"Reshaped for PCC: {reshaped_features.shape}") # (99360, 14)

# np.corrcoef computes correlation between rows, so we transpose
# Input: (14, 99360) -> Output: (14, 14) PCC matrix
A_pcc = np.corrcoef(reshaped_features.T)

# Take the absolute value. Both strong positive and negative 
# correlations are strong connections.
A_weighted = np.abs(A_pcc)
print(f"Weighted Adjacency Matrix (A) shape: {A_weighted.shape}")


# --- 3. Convert to PyG Graph Format (edge_index & edge_weight) ---
print("Converting dense matrix to sparse PyG format...")

# Convert to tensor
A_tensor = torch.tensor(A_weighted, dtype=torch.float)

# Add self-loops (GCNs need this)
# This adds 1.0 to the diagonal (the self-connection weight)
edge_index, edge_weight = add_self_loops(
    dense_to_sparse(A_tensor)[0], 
    dense_to_sparse(A_tensor)[1],
    fill_value=1.0, # The weight for the self-loop
    num_nodes=num_channels
)

print(f"Edge_index shape: {edge_index.shape}") # (2, 196) -> 14*14
print(f"Edge_weight shape: {edge_weight.shape}") # (196,)


# --- 4. Create the final PyG Data Lists ---
# Now we build the Data objects using our new features AND new graph
print("Building final data_list with DE features and weighted graph...")

def create_data_list(X, y, edge_index, edge_weight):
    data_list = []
    for i in range(len(X)):
        data = Data(
            x=torch.tensor(X[i], dtype=torch.float), # Shape: (14, 5)
            y=torch.tensor([y[i]], dtype=torch.long),
            edge_index=edge_index,   # Shared graph structure
            edge_weight=edge_weight  # Shared graph weights
        )
        data_list.append(data)
    return data_list

# Create lists for both training and testing
train_data_list = create_data_list(X_train, y_train, edge_index, edge_weight)
test_data_list = create_data_list(X_test, y_test, edge_index, edge_weight)

print(f"Created {len(train_data_list)} training samples.")
print(f"Created {len(test_data_list)} test samples.")
print(f"Example Data object: {train_data_list[0]}")
print(f"Sample 'x' shape: {train_data_list[0].x.shape}")
print(f"Sample 'edge_weight' exists: {train_data_list[0].edge_weight is not None}")