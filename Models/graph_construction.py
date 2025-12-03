import numpy as np
import torch
import mne
from torch_geometric.utils import add_self_loops

# The exact order of channels used in your preprocessing
STANDARD_CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

def get_knn_adjacency_matrix(locs_file_path, k=5):
    """
    Creates an edge_index tensor based on k-Nearest Neighbors (kNN).
    
    Args:
        locs_file_path (str): Path to the .locs file.
        k (int): Number of nearest neighbors to connect (default 5).
        
    Returns:
        edge_index (Tensor): Shape (2, Num_Edges)
    """
    # 1. Load positions
    montage = mne.channels.read_custom_montage(locs_file_path)
    pos_dict = montage.get_positions()['ch_pos']
    
    # 2. Extract positions in the correct order
    positions = []
    valid_names = []
    for name in STANDARD_CH_NAMES:
        if name in pos_dict:
            positions.append(pos_dict[name])
            valid_names.append(name)
        else:
            print(f"Warning: Channel {name} not found in montage!")
            
    positions = np.array(positions) # Shape: (62, 3)
    num_nodes = len(positions)
    
    # 3. Calculate Pairwise Distances
    dists = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dists[i, j] = np.linalg.norm(positions[i] - positions[j])
            
    # 4. Build kNN Edges
    edges = []
    for i in range(num_nodes):
        # Sort by distance (ascending)
        # Index 0 is the node itself (dist=0), so we take indices 1 to k+1
        sorted_indices = np.argsort(dists[i])
        nearest_neighbors = sorted_indices[1 : k + 1]
        
        for neighbor in nearest_neighbors:
            # Add edge: Neighbor -> Node (Source -> Target)
            edges.append([neighbor, i])
            
    # Convert to PyTorch Tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 5. Add Self-Loops (Crucial step from your notes)
    # This ensures the node considers its own features in the next layer
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    return edge_index

if __name__ == "__main__":
    # Test the graph construction
    try:
        edge_index = get_knn_adjacency_matrix("channel_62_pos.locs", k=5)
        print(f"Successfully created kNN Graph (k=5).")
        print(f"Total Edges (including self-loops): {edge_index.shape[1]}")
        print(f"Expected Edges: 62 * 5 (neighbors) + 62 (self-loops) = {62*5 + 62}")
    except Exception as e:
        print(f"Error: {e}")