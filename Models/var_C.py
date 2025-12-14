# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import DenseGCNConv

# class DGCNN_Model(nn.Module):
#     """
#     Dynamic Graph CNN (DGCNN)
#     - Learns the Adjacency Matrix (A) instead of using a fixed one.
#     - Uses DenseGCNConv because the graph is fully connected (or dynamically connected) 
#       and the structure changes or is learned as a dense matrix.
#     """
#     def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5):
#         super(DGCNN_Model, self).__init__()
        
#         self.num_nodes = num_nodes
        
#         # --- Learnable Adjacency Matrix ---
#         # We initialize it randomly. The model will learn the best structure.
#         # Shape: (Num_Nodes, Num_Nodes)
#         self.A = nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=True)
        
#         # Layers
#         # DenseGCNConv expects: (Batch, Nodes, Features) and (Batch, Nodes, Nodes)
#         self.conv1 = DenseGCNConv(in_features, hidden_dim)
#         self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
#         # Batch Norms (1D because we treat nodes as a sequence or just feature vectors)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
        
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, return_embedding=False):
#         """
#         Args:
#             x: Input features (Batch * Nodes, Features) - Flattened from DataLoader
#             edge_index: Ignored (we use self.A)
#             batch_index: Ignored (we reshape manually)
#         """
#         # 1. Reshape x back to (Batch, Nodes, Features)
#         batch_size = x.size(0) // self.num_nodes
#         x = x.view(batch_size, self.num_nodes, -1)
        
#         # 2. Prepare Adjacency Matrix
#         # We use the learned A for all samples in the batch.
#         # We can apply an activation function to A to ensure constraints (e.g., ReLU for non-negative)
#         # or Softmax to make it a probability distribution.
#         # Here we use raw weights + ReLU to keep it simple and non-negative.
#         adj = F.relu(self.A) 
        
#         # Expand A to match batch size: (Batch, Nodes, Nodes)
#         adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        
#         # 3. Layer 1
#         x = self.conv1(x, adj)
#         # BatchNorm expects (Batch, Channels, Length). We have (Batch, Nodes, Hidden).
#         # We treat 'Hidden' as Channels.
#         x = x.permute(0, 2, 1) # -> (Batch, Hidden, Nodes)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1) # -> (Batch, Nodes, Hidden)
#         x = self.dropout(x)
        
#         # 4. Layer 2
#         x = self.conv2(x, adj)
#         x = x.permute(0, 2, 1)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1)
#         x = self.dropout(x)
        
#         # 5. Global Pooling
#         # Average over all nodes to get graph representation
#         # x: (Batch, Nodes, Hidden) -> (Batch, Hidden)
#         embedding = torch.mean(x, dim=1)
        
#         # 6. Classification
#         out = self.fc(embedding)
        
#         if return_embedding:
#             return out, embedding
#         return out





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import DenseGCNConv

# class DGCNN_Model(nn.Module):
#     """
#     True Dynamic Graph CNN (Input-Dependent)
#     - The Adjacency Matrix (A) is computed dynamically for EACH sample.
#     - We use a simple similarity metric (dot product) to determine connections.
#     """
#     def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5):
#         super(DGCNN_Model, self).__init__()
        
#         self.num_nodes = num_nodes
#         self.hidden_dim = hidden_dim
        
#         # Transformation to learn "better" features for graph construction
#         self.weight_q = nn.Linear(in_features, hidden_dim)
#         self.weight_k = nn.Linear(in_features, hidden_dim)
        
#         # Layers
#         self.conv1 = DenseGCNConv(in_features, hidden_dim)
#         self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
        
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, return_embedding=False):
#         # 1. Reshape x: (Batch * Nodes, Features) -> (Batch, Nodes, Features)
#         batch_size = x.size(0) // self.num_nodes
#         x = x.view(batch_size, self.num_nodes, -1)
        
#         # 2. Dynamic Graph Construction (Self-Attention style)
#         # Q: (Batch, Nodes, Hidden)
#         Q = self.weight_q(x)
#         K = self.weight_k(x)
        
#         # Attention / Similarity: (Batch, Nodes, Nodes)
#         # We compute dot product between every pair of nodes
#         # A_ij = Q_i * K_j
#         A = torch.bmm(Q, K.transpose(1, 2))
        
#         # Normalize (Softmax makes it a probability distribution per node)
#         # This creates a directed graph where rows sum to 1
#         A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)
        
#         # 3. Layer 1
#         x = self.conv1(x, A)
#         x = x.permute(0, 2, 1) # (Batch, Hidden, Nodes)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1)
#         x = self.dropout(x)
        
#         # 4. Layer 2
#         # We can reuse A or recompute it. Reusing is standard for DGCNN to save compute.
#         x = self.conv2(x, A)
#         x = x.permute(0, 2, 1)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1)
#         x = self.dropout(x)
        
#         # 5. Global Pooling
#         embedding = torch.mean(x, dim=1)
        
#         # 6. Classification
#         out = self.fc(embedding)
        
#         if return_embedding:
#             return out, embedding
#         return out
    




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class DGCNN_Model(nn.Module):
    """
    DGCNN with Input-Dependent Band Attention.
    1. Band Attention: Reweights the 5 frequency bands dynamically per sample.
    2. Dynamic Graph: Learns the graph structure per sample.
    """
    def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5):
        super(DGCNN_Model, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # --- 1. Band Attention Layer (SE-Block style) ---
        # Input: (Batch, Nodes, 5) -> Global Pool -> (Batch, 5) -> MLP -> (Batch, 5) weights
        self.band_fc1 = nn.Linear(in_features, 16)
        self.band_fc2 = nn.Linear(16, in_features)
        
        # --- 2. Dynamic Graph Layers ---
        self.weight_q = nn.Linear(in_features, hidden_dim)
        self.weight_k = nn.Linear(in_features, hidden_dim)
        
        self.conv1 = DenseGCNConv(in_features, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch_index, return_embedding=False):
        # x: (Batch * Nodes, 5)
        
        # 1. Reshape: (Batch, Nodes, 5)
        batch_size = x.size(0) // self.num_nodes
        x = x.view(batch_size, self.num_nodes, -1)
        
        # --- Band Attention ---
        # Global Average Pooling over Nodes: (Batch, 5)
        # We want to know which bands are active in the *whole brain* right now
        global_bands = torch.mean(x, dim=1) 
        
        # Excitation: Learn weights
        w = F.relu(self.band_fc1(global_bands))
        w = torch.sigmoid(self.band_fc2(w)) # Weights in [0, 1]
        
        # Apply weights: (Batch, Nodes, 5) * (Batch, 1, 5)
        x = x * w.unsqueeze(1)
        
        # --- Dynamic Graph Construction ---
        Q = self.weight_q(x)
        K = self.weight_k(x)
        A = torch.bmm(Q, K.transpose(1, 2))
        A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)
        
        # --- GCN Layers ---
        x = self.conv1(x, A)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        
        x = self.conv2(x, A)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        
        embedding = torch.mean(x, dim=1)
        out = self.fc(embedding)
        
        if return_embedding:
            return out, embedding
        return out