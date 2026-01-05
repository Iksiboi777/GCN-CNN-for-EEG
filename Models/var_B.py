import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class AdaptiveGraphInputLayer(nn.Module):
    """
    Learnable Input Normalization Layer.
    Applies a learnable affine transformation per node (channel) and per feature (band).
    Formula: y = x * gamma + beta
    
    This allows the model to learn to:
    1. Suppress consistently noisy channels (gamma -> 0)
    2. Re-scale channels with different impedances
    3. Shift distributions to a common baseline
    """
    def __init__(self, num_nodes=62, in_features=5):
        super(AdaptiveGraphInputLayer, self).__init__()
        # Shape: (1, 62, 5) to broadcast over batch
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
        
        # Optional: Add a Batch Norm to stabilize the initial distribution
        # We use LayerNorm over the feature dimension to keep nodes independent initially
        self.initial_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # x shape: (Batch*Nodes, Features) or (Batch, Nodes, Features)
        # GCN input is usually (Batch*Nodes, Features)
        
        # 1. Reshape to (Batch, Nodes, Features) for channel-wise scaling
        # We assume the input x is flattened as (Batch * 62, 5)
        batch_size = x.size(0) // 62
        x_reshaped = x.view(batch_size, 62, -1)
        
        # 2. Apply Learnable Affine Transformation
        # x_reshaped: (B, 62, 5) * (1, 62, 5) + (1, 62, 5)
        x_adapted = x_reshaped * self.gamma + self.beta
        
        # 3. Apply standard normalization (optional, helps convergence)
        x_adapted = self.initial_norm(x_adapted)
        
        # 4. Flatten back to (Batch*Nodes, Features) for PyG
        return x_adapted.view(-1, x.size(-1))

class GCN_DE_Model(nn.Module):
    """
    Dynamic GCN Model for Differential Entropy (DE) Features.
    Input: (Batch * Nodes, 5) -> 5 Frequency Bands per Node
    Output: (Batch, Num_Classes)
    """
    def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5, num_layers=3):
        super(GCN_DE_Model, self).__init__()
        
        # --- NEW: Insert Adaptive Layer ---
        self.input_norm = AdaptiveGraphInputLayer(num_nodes, in_features)
        # ----------------------------------
        
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
        
        # --- NEW: Apply Adaptive Normalization ---
        x = self.input_norm(x)
        # -----------------------------------------
        
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





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool

# class GCN_DE_Model(nn.Module):
#     """
#     Static GCN Model with Feature Attention.
#     1. Feature Attention: Reweights inputs (Mean vs Variance) per graph.
#     2. Static Graph: Uses fixed adjacency matrix (KNN).
#     """
#     def __init__(self, num_nodes=62, in_features=10, hidden_dim=64, num_classes=3, dropout_rate=0.5, num_layers=3):
#         super(GCN_DE_Model, self).__init__()
        
#         # --- 1. Feature Attention Layer (SE-Block style) ---
#         # We compress features to 16, then expand back to in_features
#         self.feature_fc1 = nn.Linear(in_features, 16)
#         self.feature_fc2 = nn.Linear(16, in_features)
        
#         # --- 2. GCN Layers ---
#         self.layers = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.dropout_rate = dropout_rate
        
#         # Input Layer (Layer 1)
#         self.layers.append(GCNConv(in_features, hidden_dim))
#         self.bns.append(nn.BatchNorm1d(hidden_dim))
        
#         # Hidden Layers (Layer 2 to N)
#         for _ in range(num_layers - 1):
#             self.layers.append(GCNConv(hidden_dim, hidden_dim))
#             self.bns.append(nn.BatchNorm1d(hidden_dim))
            
#         # Classifier
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, return_embedding=False):
#         # x shape: (Batch * Nodes, in_features)
        
#         # --- Feature Attention ---
#         # 1. Aggregate node features to get graph-level context: (Batch, in_features)
#         global_features = global_mean_pool(x, batch_index)
        
#         # 2. Learn importance weights
#         w = F.relu(self.feature_fc1(global_features))
#         w = torch.sigmoid(self.feature_fc2(w)) # Weights in [0, 1]
        
#         # 3. Apply weights to nodes
#         # w[batch_index] broadcasts the (Batch, Feat) weights to (Batch*Nodes, Feat)
#         # so every node in Graph i gets multiplied by weights of Graph i
#         x = x * w[batch_index]
        
#         # --- GCN Forward Pass ---
#         for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.relu(x)
            
#             # Apply dropout to all layers except the last GCN layer
#             if i < len(self.layers) - 1:
#                 x = self.dropout(x)
        
#         # Global Pooling
#         embedding = global_mean_pool(x, batch_index) 
        
#         # Final Classification
#         out = self.fc(embedding)
#         if return_embedding:
#             return out, embedding
        
#         return out