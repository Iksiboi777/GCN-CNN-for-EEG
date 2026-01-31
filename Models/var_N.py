import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    """
    Stage III: Temporal Block (CNN Encoder)
    Physics-Aware: Hand-tuned for 200Hz SEED Data.
    """
    def __init__(self, in_channels=1):
        super(TemporalBlock, self).__init__()
        
        # Layer 1: Capture Delta/Theta (0.64s window)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=128, stride=2, padding=64)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Layer 2: Capture Alpha/Beta (0.32s window)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=64, stride=2, padding=32)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Layer 3: Capture Gamma (0.16s window)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=32, stride=2, padding=16)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.4) 
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        x = self.dropout(F.gelu(self.bn2(self.conv2(x))))
        x = self.dropout(F.gelu(self.bn3(self.conv3(x))))
        
        x = self.pool(x)
        return x.squeeze(-1) # (Batch*62, 256)

class SelfAdaptiveAdj(nn.Module):
    """
    Stage IV: Spatial Block (Adjacency Learning)
    """
    def __init__(self, dim, k=10):
        super(SelfAdaptiveAdj, self).__init__()
        self.k = k
        self.dim = dim
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (Batch, 62, Features)
        Q = self.W_q(x) 
        K = self.W_k(x) 
        
        # Raw Attention Scores
        attn = torch.matmul(Q, K.transpose(1, 2)) * self.scale
        
        # --- Top-K Sparsity Constraint (Fixed) ---
        # 1. Find the top K indices per row
        topk_vals, topk_inds = torch.topk(attn, self.k, dim=-1)
        
        # 2. Create a mask filled with -infinity
        mask = torch.full_like(attn, float('-inf'))
        
        # 3. Scatter zeros into the Top-K positions
        # (Anything added to 0 stays the same; Anything added to -inf becomes -inf)
        mask.scatter_(-1, topk_inds, 0)
        
        # 4. Apply mask
        attn_masked = attn + mask
        
        # 5. Softmax turns -inf into 0.0, preserving probability distribution
        A = F.softmax(attn_masked, dim=-1)
        
        return A

class SpatialBlock(nn.Module):
    """
    Stage IV: GCN Layers
    """
    def __init__(self, in_features, hidden_features):
        super(SpatialBlock, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        
        # Correctly sized BatchNorm for the feature dimension
        self.bn = nn.BatchNorm1d(hidden_features) 
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, A):
        x = self.fc(x)
        x = torch.matmul(A, x) 
        
        x = x.permute(0, 2, 1) # BN over features
        x = self.bn(x) 
        x = F.gelu(x)
        x = x.permute(0, 2, 1) 
        
        x = self.dropout(x)
        return x

class SOTA_CNNGCN(nn.Module):
    def __init__(self, num_nodes=62, time_steps=400, num_classes=3, sparsity_k=10):
        super(SOTA_CNNGCN, self).__init__()
        self.adaptive_input = True
        
        # 1. Temporal Encoder
        self.temporal = TemporalBlock()
        self.temporal_dim = 256 # Matches output of TemporalBlock
        
        # 2. Graph Learner
        self.adj_learner = SelfAdaptiveAdj(self.temporal_dim, k=sparsity_k)
        
        # 3. Spatial GCNs
        self.gcn1 = SpatialBlock(self.temporal_dim, 256)
        self.gcn2 = SpatialBlock(256, 128)
        
        # Residual Connections
        self.res_proj = nn.Linear(256, 128) 
        
        # 4. Classifier
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, _=None, __=None, ___=None): 
        B, N, T = x.shape
        
        # --- Stage III: Temporal Block ---
        x_reshaped = x.view(B * N, 1, T)
        t_feat = self.temporal(x_reshaped)      # (B*N, 256)
        x_graph = t_feat.view(B, N, -1)         # (B, 62, 256)
        
        # --- Stage IV: Spatial Block ---
        A = self.adj_learner(x_graph)
        
        out = self.gcn1(x_graph, A)             # 256 -> 256
        
        identity = self.res_proj(x_graph)
        out = self.gcn2(out, A)                 # 256 -> 128
        out = out + identity 
        
        # --- Stage V: Classification ---
        out = out.mean(dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        
        return logits