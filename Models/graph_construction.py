import numpy as np
import torch
import mne
from torch_geometric.utils import add_self_loops

STANDARD_CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

def get_knn_adjacency_matrix(locs_file_path=None, k=5):
    """
    Creates an edge_index tensor based on k-Nearest Neighbors (kNN).
    CHECKLIST #13: Uses Standard 10-20 Montage if .locs is missing.
    """
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        pos_dict = montage.get_positions()['ch_pos']
    except Exception as e:
        print(f"Fallback to custom locs: {e}")
        montage = mne.channels.read_custom_montage(locs_file_path)
        pos_dict = montage.get_positions()['ch_pos']
    
    positions = []
    for name in STANDARD_CH_NAMES:
        lookup = name if name not in ['CB1', 'CB2'] else ('PO9' if name == 'CB1' else 'PO10')
        if lookup in pos_dict:
            positions.append(pos_dict[lookup])
        else:
            positions.append([0, 0, 0])
            print(f"Warning: Channel {name} location synthesized.")
            
    positions = np.array(positions) 
    num_nodes = len(positions)
    
    dists = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dists[i, j] = np.linalg.norm(positions[i] - positions[j])
            
    edges = []
    for i in range(num_nodes):
        sorted_indices = np.argsort(dists[i])
        nearest_neighbors = sorted_indices[1 : k + 1]
        
        for neighbor in nearest_neighbors:
            edges.append([neighbor, i])
            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    return edge_index