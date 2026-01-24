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
    




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

# class AdaptiveGraphInputLayer(nn.Module):
#     """
#     Learnable Input Normalization Layer.
#     Applies a learnable affine transformation per node (channel) and per feature (band).
#     Formula: y = x * gamma + beta
    
#     This allows the model to learn to:
#     1. Suppress consistently noisy channels (gamma -> 0)
#     2. Re-scale channels with different impedances
#     3. Shift distributions to a common baseline
#     """
#     def __init__(self, num_nodes=62, in_features=5):
#         super(AdaptiveGraphInputLayer, self).__init__()
#         # Shape: (1, 62, 5) to broadcast over batch
#         self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
#         self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
        
#         # Optional: Add a Batch Norm to stabilize the initial distribution
#         # We use LayerNorm over the feature dimension to keep nodes independent initially
#         self.initial_norm = nn.LayerNorm(in_features)

#     def forward(self, x):
#         # x shape: (Batch*Nodes, Features) or (Batch, Nodes, Features)
#         # GCN input is usually (Batch*Nodes, Features)
        
#         # 1. Reshape to (Batch, Nodes, Features) for channel-wise scaling
#         # We assume the input x is flattened as (Batch * 62, 5)
#         batch_size = x.size(0) // 62
#         x_reshaped = x.view(batch_size, 62, -1)
        
#         # 2. Apply Learnable Affine Transformation
#         # x_reshaped: (B, 62, 5) * (1, 62, 5) + (1, 62, 5)
#         x_adapted = x_reshaped * self.gamma + self.beta
        
#         # 3. Apply standard normalization (optional, helps convergence)
#         x_adapted = self.initial_norm(x_adapted)
        
#         # 4. Flatten back to (Batch*Nodes, Features) for PyG
#         return x_adapted.view(-1, x.size(-1))

# class DGCNN_Model(nn.Module):
#     def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5):
#         super(DGCNN_Model, self).__init__()
        
#         # --- NEW: Insert Adaptive Layer ---
#         self.input_norm = AdaptiveGraphInputLayer(num_nodes, in_features)
#         # ----------------------------------

#         self.conv1 = GCNConv(in_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
#         # Learnable Adjacency Matrix (Edge Weight Prediction)
#         self.edge_weight_learner = nn.Sequential(
#             nn.Linear(in_features * 2, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
        
#         self.fc1 = nn.Linear(hidden_dim * 2, 64)
#         self.fc2 = nn.Linear(64, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch, return_embedding=False):
#         # x shape: (Batch * Num_Nodes, In_Features)
        
#         # --- NEW: Apply Adaptive Normalization ---
#         x = self.input_norm(x)
#         # -----------------------------------------

#         # 1. Dynamic Edge Weight Learning
#         row, col = edge_index
#         x_i = x[row]
#         x_j = x[col]
#         edge_feat = torch.cat([x_i, x_j], dim=1)
#         edge_weight = self.edge_weight_learner(edge_feat).squeeze()

#         # 2. GCN Layers
#         x = self.conv1(x, edge_index, edge_weight)
#         x = F.relu(x)
#         x = self.dropout(x)
        
#         x = self.conv2(x, edge_index, edge_weight)
#         x = F.relu(x)
        
#         # 3. Global Pooling
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x_cat = torch.cat([x_mean, x_max], dim=1)
        
#         # 4. Classifier
#         embedding = self.fc1(x_cat)
#         x_out = F.relu(embedding)
#         x_out = self.dropout(x_out)
#         out = self.fc2(x_out)
        
#         if return_embedding:
#             return out, embedding
#         return out





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import DenseGCNConv

# class DGCNN_Model(nn.Module):
#     """
#     DGCNN with Input-Dependent Feature Attention.
#     1. Feature Attention: Reweights the 10 input features (5 Mean + 5 Var) dynamically.
#     2. Dynamic Graph: Learns the graph structure per sample.
#     """
#     def __init__(self, num_nodes=62, in_features=10, hidden_dim=64, num_classes=3, dropout_rate=0.5):
#         super(DGCNN_Model, self).__init__()
        
#         self.num_nodes = num_nodes
#         self.hidden_dim = hidden_dim
        
#         # --- 1. Feature Attention Layer (SE-Block style) ---
#         # Input: (Batch, Nodes, in_features) -> Global Pool -> (Batch, in_features)
#         # We want to reweight the features dynamically per sample
#         self.feature_fc1 = nn.Linear(in_features, 16)
#         self.feature_fc2 = nn.Linear(16, in_features)
        
#         # --- 2. Dynamic Graph Layers ---
#         # Transforms input to Hidden Dim for Graph Construction
#         self.weight_q = nn.Linear(in_features, hidden_dim)
#         self.weight_k = nn.Linear(in_features, hidden_dim)
        
#         # GCN Layers
#         self.conv1 = DenseGCNConv(in_features, hidden_dim)
#         self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
#         # Batch Norms (1D because we treat nodes as a sequence)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
        
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, return_embedding=False):
#         # x: (Batch * Nodes, in_features)
        
#         # 1. Reshape: (Batch, Nodes, in_features)
#         batch_size = x.size(0) // self.num_nodes
#         x = x.view(batch_size, self.num_nodes, -1)
        
#         # --- Feature Attention ---
#         # Global Average Pooling over Nodes: (Batch, in_features)
#         # We want to know which features (Bands/Variance) are active in the *whole brain*
#         global_features = torch.mean(x, dim=1) 
        
#         # Excitation: Learn weights
#         w = F.relu(self.feature_fc1(global_features))
#         w = torch.sigmoid(self.feature_fc2(w)) # Weights in [0, 1]
        
#         # Apply weights: (Batch, Nodes, in_features) * (Batch, 1, in_features)
#         x = x * w.unsqueeze(1)
        
#         # --- Dynamic Graph Construction ---
#         # Q, K: (Batch, Nodes, Hidden)
#         Q = self.weight_q(x)
#         K = self.weight_k(x)
        
#         # Attention / Similarity: (Batch, Nodes, Nodes)
#         A = torch.bmm(Q, K.transpose(1, 2))
#         # Softmax to create a probability distribution for edges
#         A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)
        
#         # --- GCN Layers ---
#         x = self.conv1(x, A)
#         x = x.permute(0, 2, 1) # (Batch, Hidden, Nodes) for BatchNorm
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1) # Back to (Batch, Nodes, Hidden)
#         x = self.dropout(x)
        
#         x = self.conv2(x, A)
#         x = x.permute(0, 2, 1)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1)
#         x = self.dropout(x)
        
#         # Global Pooling
#         embedding = torch.mean(x, dim=1)
        
#         # Classification
#         out = self.fc(embedding)

#         if return_embedding:
#             return out, embedding
#         return out
    





# ADAPTIVE SUBJECT BIAS MODEL

# ...existing imports...
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class DGCNN_Model(nn.Module):
    """
    DGCNN with Input-Dependent Feature Attention.
    """
    def __init__(self, num_nodes=62, in_features=10, hidden_dim=64, 
                 num_classes=3, dropout_rate=0.5, num_subjects=15): # Added num_subjects
        super(DGCNN_Model, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # --- 1. Feature Attention Layer (SE-Block style) ---
        self.feature_fc1 = nn.Linear(in_features, 16)
        self.feature_fc2 = nn.Linear(16, in_features)
        
        # --- 2. Dynamic Graph Layers ---
        self.weight_q = nn.Linear(in_features, hidden_dim)
        self.weight_k = nn.Linear(in_features, hidden_dim)
        
        # GCN Layers
        self.conv1 = DenseGCNConv(in_features, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
        # Batch Norms
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # --- NEW: SUBJECT BIAS ---
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0)
        # -------------------------
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch_index, subject_ids=None, return_embedding=False): # Added subject_ids
        # x: (Batch * Nodes, in_features)
        
        # 1. Reshape: (Batch, Nodes, in_features)
        batch_size = x.size(0) // self.num_nodes
        x = x.view(batch_size, self.num_nodes, -1)
        
        # --- Feature Attention ---
        global_features = torch.mean(x, dim=1) 
        w = F.relu(self.feature_fc1(global_features))
        w = torch.sigmoid(self.feature_fc2(w)) 
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
        
        # Global Pooling
        embedding = torch.mean(x, dim=1)
        
        # Classification
        logits = self.fc(embedding)

        # --- NEW: APPLY BIAS ---
        if subject_ids is not None:
             logits = logits + self.subject_bias(subject_ids)
        # -----------------------

        if return_embedding:
            return logits, embedding
        return logits