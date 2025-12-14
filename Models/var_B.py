import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN_DE_Model(nn.Module):
    """
    Dynamic GCN Model for Differential Entropy (DE) Features.
    Input: (Batch * Nodes, 5) -> 5 Frequency Bands per Node
    Output: (Batch, Num_Classes)
    """
    def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5, num_layers=3):
        super(GCN_DE_Model, self).__init__()
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout_rate = dropout_rate
        
        # Input Layer (Layer 1)
        self.layers.append(GCNConv(in_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden Layers (Layer 2 to N)
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch_index, return_embedding=False):
        # x shape: (Batch * Nodes, 5)
        
        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
            # Apply dropout to all layers except the last GCN layer
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        
        # Global Pooling (aggregates nodes for each graph in the batch)
        # Output: (Batch_Size, Hidden_Dim)
        embedding = global_mean_pool(x, batch_index) 
        
        # Final Classification
        out = self.fc(embedding)
        
        if return_embedding:
            return out, embedding
        return out