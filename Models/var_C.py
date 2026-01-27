# # ADAPTIVE SUBJECT BIAS MODEL

# # ...existing imports...
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import DenseGCNConv

# class DGCNN_Model(nn.Module):
#     """
#     DGCNN with Input-Dependent Feature Attention.
#     """
#     def __init__(self, num_nodes=62, in_features=10, hidden_dim=64, 
#                  num_classes=3, dropout_rate=0.5, num_subjects=15): # Added num_subjects
#         super(DGCNN_Model, self).__init__()
        
#         self.num_nodes = num_nodes
#         self.hidden_dim = hidden_dim
        
#         # --- 1. Feature Attention Layer (SE-Block style) ---
#         self.feature_fc1 = nn.Linear(in_features, 16)
#         self.feature_fc2 = nn.Linear(16, in_features)
        
#         # --- 2. Dynamic Graph Layers ---
#         self.weight_q = nn.Linear(in_features, hidden_dim)
#         self.weight_k = nn.Linear(in_features, hidden_dim)
        
#         # GCN Layers
#         self.conv1 = DenseGCNConv(in_features, hidden_dim)
#         self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        
#         # Batch Norms
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
        
#         self.fc = nn.Linear(hidden_dim, num_classes)
        
#         # --- NEW: SUBJECT BIAS ---
#         self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
#         self.subject_bias.weight.data.fill_(0.0)
#         # -------------------------
        
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, subject_ids=None, return_embedding=False): # Added subject_ids
#         # x: (Batch * Nodes, in_features)
        
#         # 1. Reshape: (Batch, Nodes, in_features)
#         batch_size = x.size(0) // self.num_nodes
#         x = x.view(batch_size, self.num_nodes, -1)
        
#         # --- Feature Attention ---
#         global_features = torch.mean(x, dim=1) 
#         w = F.relu(self.feature_fc1(global_features))
#         w = torch.sigmoid(self.feature_fc2(w)) 
#         x = x * w.unsqueeze(1)
        
#         # --- Dynamic Graph Construction ---
#         Q = self.weight_q(x)
#         K = self.weight_k(x)
#         A = torch.bmm(Q, K.transpose(1, 2))
#         A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)
        
#         # --- GCN Layers ---
#         x = self.conv1(x, A)
#         x = x.permute(0, 2, 1) 
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = x.permute(0, 2, 1) 
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
#         logits = self.fc(embedding)

#         # --- NEW: APPLY BIAS ---
#         if subject_ids is not None:
#              logits = logits + self.subject_bias(subject_ids)
#         # -----------------------

#         if return_embedding:
#             return logits, embedding
#         return logits




# ATTEMPT 61
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from .phase2_layers import SubjectSpecificBatchNorm1d

class DGCNN_Model(nn.Module):
    def __init__(self, num_nodes=62, in_features=11, hidden_dim=64, 
                 num_classes=3, num_subjects=15, num_layers=2):
        super(DGCNN_Model, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        self.input_ln = nn.LayerNorm(in_features) 
        
        # Prioritized Feature Attention
        self.feature_gate = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, in_features),
            nn.Sigmoid()
        )
        
        # Dynamic Learners
        self.weight_q = nn.Linear(in_features, hidden_dim)
        self.weight_k = nn.Linear(in_features, hidden_dim)
        
        # Loop for Dense GCN
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_dim
            self.convs.append(DenseGCNConv(in_dim, hidden_dim))
            self.norms.append(SubjectSpecificBatchNorm1d(hidden_dim, num_subjects))
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0)

    def forward(self, x, edge_index, batch_index, subject_ids_graph, return_embedding=False):
        # NOTE: Helper reshapes flattened x back to Dense
        batch_size = x.size(0) // self.num_nodes
        x = x.view(batch_size, self.num_nodes, -1) # (B, N, F)
        
        x = self.input_ln(x)
        
        # Feature Attention Step
        global_vec = x.mean(dim=1) 
        weights = self.feature_gate(global_vec).unsqueeze(1) 
        x = x * weights
        
        # Learn Adjacency
        Q = self.weight_q(x)
        K = self.weight_k(x)
        A = torch.bmm(Q, K.transpose(1, 2))
        A = F.softmax(A / 8.0, dim=-1) 
        
        # Stacked Loop
        for i in range(self.num_layers):
            x = self.convs[i](x, A)
            
            # SSBN Wrap
            x = x.transpose(1, 2)
            x = self.norms[i](x, subject_ids_graph)
            x = x.transpose(1, 2)
            
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=0.5, training=self.training)
        
        # Simple Mean Pool
        embedding = x.mean(dim=1)
        
        logits = self.fc(embedding)
        logits = logits + self.subject_bias(subject_ids_graph)
        
        if return_embedding:
            return logits, embedding
        return logits