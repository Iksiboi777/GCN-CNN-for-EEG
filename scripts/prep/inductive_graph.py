# import torch
# import numpy as np
# from torch_geometric.data import Data
# from torch_geometric.loader import NeighborLoader
# from scipy.spatial.distance import pdist, squareform

# def build_inductive_graph(features, coordinates, labels, k=5):
#     """
#     Redefines the EEG data as an Inductive Graph structure.
    
#     Args:
#         features: (Samples, 62, 10) - Your DE + Variance features.
#         coordinates: (62, 3) or (62, 2) - Electrode positions.
#         labels: (Samples,) - Emotion labels.
#         k: Number of neighbors for GraphSAGE to aggregate from.
#     """
#     num_samples = features.shape[0]
#     num_nodes = features.shape[1]
    
#     # 1. Create Adjacency (The "Template" for the Aggregator)
#     # We calculate distances once because the sensors don't move.
#     dist_matrix = squareform(pdist(coordinates))
    
#     # For each node, find the K-nearest neighbors
#     edge_index_list = []
#     for i in range(num_nodes):
#         # Sort distances and get indices of k-nearest (excluding self)
#         nn_indices = np.argsort(dist_matrix[i])[1:k+1]
#         for nn in nn_indices:
#             edge_index_list.append([nn, i]) # Direction: Neighbor -> Target
            
#     edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
#     # 2. Convert each trial into a PyG Data object
#     dataset = []
#     for i in range(num_samples):
#         # Node features for this specific trial (62, 10)
#         x = features[i].detach().clone().float()
#         y = labels[i].detach().clone().long()
        
#         # In GraphSAGE, each Data object represents the WHOLE graph 
#         # but the Loader will handle the inductive sampling.
#         data = Data(x=x, edge_index=edge_index, y=y)
#         dataset.append(data)
        
#     return dataset, edge_index

# # --- THE INDUCTIVE ENGINE: THE NEIGHBOR LOADER ---
# def get_sage_loader(dataset, batch_size=32, shuffle=True):
#     """
#     This is the core of GraphSAGE. Unlike standard DataLoader, 
#     this samples neighborhoods on the fly.
#     """
#     # Note: We aggregate over the list of graphs
#     # num_neighbors=[10, 5] means:
#     # Sample 10 neighbors at Depth 1, and 5 neighbors for each of those at Depth 2.
#     loader = NeighborLoader(
#         dataset[0], # Using the first graph as a structural template
#         num_neighbors=[10, 5], 
#         batch_size=batch_size,
#         shuffle=shuffle
#     )
#     return loader

# print("Graph Redefinition Logic Ready.")


import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def get_base_edge_index(locs_path, k=5):
    """
    Creates the fixed spatial template for 62 electrodes.
    This is what 'training_utils.py' uses to build the batch graph.
    """
    # Load coordinates (assumes x, y are in cols 1, 2)
    coords = np.loadtxt(locs_path, usecols=(1, 2))
    dist_matrix = squareform(pdist(coords))
    
    edge_index_list = []
    for i in range(62):
        # Find k-nearest neighbors (excluding self)
        nn_indices = np.argsort(dist_matrix[i])[1:k+1]
        for nn in nn_indices:
            # Information flows from Neighbor -> Target (SAGE style)
            edge_index_list.append([nn, i])
            
    return torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(), coords

def visualize_sage_influence(model, coords, channel_names):
    """Visualizes the 'importance' of electrodes after training."""
    model.eval()
    with torch.no_grad():
        # Using the norms of the weights in the first aggregator
        importance = torch.norm(model.sage1.lin_l.weight, dim=0).cpu().numpy()
        # Scale for visualization
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=importance[:62], cmap='hot', s=500, edgecolors='k')
    for i, name in enumerate(channel_names):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), ha='center', va='center', color='white')
    plt.title("GraphSAGE Aggregation Importance Map")
    plt.savefig("sage_importance_map.png")
    # plt.show()