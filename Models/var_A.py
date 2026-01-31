import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import ChebConv, global_max_pool, GCNConv, global_mean_pool

# --- 1. THE ADAPTIVE GRAPH LEARNING (SPARSE COMPATIBLE) ---
class SparseAGLI(nn.Module):
    """
    Computes adaptive edge weights for an existing sparse graph structure (edge_index).
    """
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, in_features))
        init.xavier_uniform_(self.W)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index):
        # x: (TotalNodes, F)
        # edge_index: (2, TotalEdges)
        
        row, col = edge_index
        
        # 1. Transform Source Nodes
        x_query = torch.matmul(x, self.W) # (TotalNodes, F)
        
        # 2. Compute Attention Score for each existing edge
        # Gather features for source (row) and target (col)
        # q: (TotalEdges, F), k: (TotalEdges, F)
        q = x_query[row] 
        k = x[col]       
        
        # Dot product attention
        edge_attn = (q * k).sum(dim=-1) # (TotalEdges,)
        
        # Normalize (Sigmoid for stability 0-1)
        edge_attn = torch.sigmoid(edge_attn)
        
        # 3. Add to static weight (Assumed 1.0)
        # New Weight = 1 + alpha * learned_attention
        return 1.0 + self.alpha * edge_attn

# --- 2. THE FEATURE ATTENTION (SEBlock) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B*N, C)
        avg_pool = torch.mean(x, dim=0, keepdim=True) 
        weight = self.fc(avg_pool)
        return x * weight

# --- 3. THE MASTER HYBRID MODEL (Attempt 61) ---
class Attempt61_CNNGCN(nn.Module):
    def __init__(self, num_nodes=62, time_steps=400, num_classes=3, num_subjects=15):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Temporal Stream
        self.conv_short = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(1, 32, kernel_size=31, padding=15)
        self.bn_temp = nn.BatchNorm1d(64)
        
        # Feature Compression
        self.compressor = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=5),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # --- FIX 1: Add Dropout Layers ---
        self.dropout_input = nn.Dropout(0.2) # Drop raw features
        self.dropout_hidden = nn.Dropout(0.5) # Drop internal representation
        
        self.agli = SparseAGLI(128)
        self.se = SEBlock(128)
        
        # Spatial Stream
        self.gnn1 = ChebConv(128, 128, K=2) # Reduced K from 3 to 2 to reduce complexity
        self.gnn2 = ChebConv(128, 128, K=2)
        
        # Subject Calibration
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, batch_index, subject_ids):
        # 1. Ensure correct shape (B*N, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 2. Temporal Inception
        x_s = self.conv_short(x)
        x_l = self.conv_long(x)
        x = torch.cat([x_s, x_l], dim=1)
        x = F.gelu(self.bn_temp(x))
        
        # --- APPLY DROPOUT 1 ---
        x = self.dropout_input(x)
        
        # 3. Node Embedding
        x = self.compressor(x).squeeze(-1) 
        
        # 4. Sparse Dynamic Graph Learning
        edge_weight = self.agli(x, edge_index)
        
        # 5. Feature Refinement
        x = self.se(x) 
        
        # 6. GNN layers
        x = F.gelu(self.gnn1(x, edge_index, edge_weight))
        
        # --- APPLY DROPOUT 2 (Between GNN layers is critical) ---
        x = self.dropout_hidden(x)
        
        x = F.gelu(self.gnn2(x, edge_index, edge_weight))
        
        # 7. Global Pool & Classify
        graph_emb = global_max_pool(x, batch_index)
        
        # --- APPLY DROPOUT 3 (Before Final Classifier) ---
        graph_emb = self.dropout_hidden(graph_emb)
        
        logits = self.fc(graph_emb)
        
        # Apply the Subject-Specific Offset
        bias = self.subject_bias(subject_ids)
        
        return logits + bias

# # --- OLD CLASS FOR BACKWARD COMPATIBILITY ---
# class Conv1dBlock(nn.Module):
#     """
#     Per-Channel CNN (Shared Weights).
#     Input: (Batch * Nodes, 1, Time_Steps)
#     Output: (Batch * Nodes, Embedding_Dim)
#     """
#     def __init__(self, in_channels=1, out_channels=64):
#         super(Conv1dBlock, self).__init__()
        
#         # Layer 1: Conv1d(1, 64, k=15, s=1, p=7) - Wider first layer
#         self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=1, padding=7)
#         self.bn1 = nn.BatchNorm1d(64)
        
#         # Layer 2: Conv1d(64, 128, k=5, s=1, p=2)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm1d(128)
        
#         self.pool = nn.MaxPool1d(2)
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(0.5)

#         # Final Linear Projection
#         self.fc = nn.Linear(128, out_channels)
#         self.out_bn = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         if self.training:
#             noise = torch.randn_like(x) * 0.01
#             x = x + noise
            
#         x = F.relu(self.bn1(self.conv1(x))) 
#         x = self.pool(x)
#         x = F.relu(self.bn2(self.conv2(x))) 
#         x = self.global_pool(x).squeeze(-1) 
#         x = self.dropout(x)
#         x = self.fc(x) 
#         x = self.out_bn(x)
#         return x

# class GCNBlock(nn.Module):
#     def __init__(self, in_features=64, hidden_dim=64, num_classes=3):
#         super(GCNBlock, self).__init__()
#         self.gcn1 = GCNConv(in_features, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.gcn3 = GCNConv(hidden_dim, hidden_dim)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x, edge_index, batch_index):
#         x = self.gcn1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.gcn2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.gcn3(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = global_mean_pool(x, batch_index) 
#         x = self.fc(x)
#         return x

# class CNNGCNModel(nn.Module):
#     def __init__(self, num_nodes=62, time_steps=400):
#         super(CNNGCNModel, self).__init__()
#         self.num_nodes = num_nodes
#         self.cnn = Conv1dBlock(out_channels=128)
#         self.gcn = GCNBlock(in_features=128, hidden_dim=128)

#     def forward(self, x, edge_index, batch_index):
#         batch_size = x.size(0)
#         x = x.view(batch_size * self.num_nodes, 1, -1)
#         node_embeddings = self.cnn(x) 
#         out = self.gcn(node_embeddings, edge_index, batch_index)
#         return out