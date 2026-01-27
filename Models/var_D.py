# # ADAPTIVE SUBJECT BIAS MODEL


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import DenseGCNConv, GlobalAttention
# from phase2_layers import SubjectSpecificBatchNorm1d

# class AdaptiveGraphInputLayer(nn.Module):
#     """ The 'Pre-Amp'. Learned scaling factors for nodes. """
#     def __init__(self, num_nodes=62, in_features=5):
#         super(AdaptiveGraphInputLayer, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
#         self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
#         self.initial_norm = nn.LayerNorm(in_features)

#     def forward(self, x):
#         x = self.initial_norm(x)
#         return x * self.gamma + self.beta

# class SEBlock(nn.Module):
#     """ Squeeze-and-Excitation to weight frequency bands. """
#     def __init__(self, channel, reduction=4):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, max(1, channel // reduction), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(1, channel // reduction), channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, n, f = x.size()
#         y = x.transpose(1, 2)
#         y = self.avg_pool(y).view(b, f)
#         y = self.fc(y).view(b, 1, f)
#         return x * y.expand_as(x)

# class Adaptive_DGCNN(nn.Module):
#     def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, 
#                  num_classes=3, dropout_rate=0.5, num_layers=3, num_subjects=15):
#         super(Adaptive_DGCNN, self).__init__()
        
#         # 1. PHYSIOLOGICAL PRIORS
#         self.adaptive_input = AdaptiveGraphInputLayer(num_nodes, in_features)
#         self.se_block = SEBlock(in_features)

#         # 2. DYNAMIC GRAPH LEARNERS
#         self.weight_q = nn.Linear(in_features, hidden_dim)
#         self.weight_k = nn.Linear(in_features, hidden_dim)
        
#         # 3. DENSE CONVOLUTIONS (Stacked)
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
        
#         # Layers 2..N
#         for _ in range(num_layers):
#             self.convs.append(DenseGCNConv(hidden_dim, hidden_dim))
#             self.bns.append(SubjectSpecificBatchNorm1d(num_nodes, num_subjects))

#         # 4. SUBJECT-SPECIFIC BIAS (The "Adjustment Term")
#         # Learned bias for each of the 15 subjects to shift probabilities 
#         # (e.g., bias Subject 12 towards Neutral if they are prone to artifacts)
#         self.subject_bias = nn.Embedding(num_subjects + 1, num_classes) # +1 for safe indexing
#         self.subject_bias.weight.data.fill_(0.0) # Start with no bias

#         # 5. POOLING & CLASSIFICATION
#         self.att_gate = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#         self.pool = GlobalAttention(gate_nn=self.att_gate)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, subject_ids=None):
#         # x: (Batch, Nodes, Features)
        
#         # --- PHASE 1: SENSOR CALIBRATION ---
#         x = self.adaptive_input(x)
#         x = self.se_block(x)
        
#         # --- PHASE 2: LEARN TOPOLOGY ---
#         Q = self.weight_q(x)
#         K = self.weight_k(x)
#         A = torch.bmm(Q, K.transpose(1, 2))
#         A = F.softmax(A / (x.shape[-1]**0.5), dim=-1)

#         # --- PHASE 3: GRAPH CONVOLUTIONS ---
#         for i in range(self.num_layers):
#             x = self.convs[i](x, A)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             if i < self.num_layers - 1:
#                 x = self.dropout(x)

#         # --- PHASE 4: POOLING ---
#         b, n, h = x.shape
#         x_flat = x.reshape(-1, h)
#         batch_idx = torch.arange(b, device=x.device).repeat_interleave(n)
#         embedding = self.pool(x_flat, batch_idx)
        
#         logits = self.fc(embedding)

#         # --- PHASE 5: ADAPTIVE BIAS ---
#         if subject_ids is not None:
#             # Shift the logits based on who the subject is
#             bias = self.subject_bias(subject_ids)
#             logits = logits + bias

#         return logits


# ATTEMPT 61 MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from .phase2_layers import SubjectSpecificBatchNorm1d

class Adaptive_DGCNN(nn.Module):
    def __init__(self, num_nodes=62, in_features=11, hidden_dim=64, 
                 num_classes=3, num_subjects=15, num_layers=2):
        super(Adaptive_DGCNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # Standard input norm for dense (simplifies initial distribution)
        self.input_ln = nn.LayerNorm(in_features) 

        # AGLI
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
        
        # Dense SE Block
        self.se_fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )

        # Dynamic Graph Learners
        self.weight_q = nn.Linear(in_features, hidden_dim)
        self.weight_k = nn.Linear(in_features, hidden_dim)
        
        # Loop for Dense GCN Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_dim
            self.convs.append(DenseGCNConv(in_dim, hidden_dim))
            # SSBN expects (N, C, L), so we use hidden_dim as 'features'
            self.norms.append(SubjectSpecificBatchNorm1d(hidden_dim, num_subjects))
        
        # Global Attention Pool Gate
        self.gate = nn.Linear(hidden_dim, 1)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0)

    def forward(self, x, subject_ids_graph, return_embedding=False):
        # x: (Batch, Nodes, Features)
        
        x = self.input_ln(x)
        x = x * self.gamma + self.beta
        
        # SE Block
        global_feat = x.mean(dim=1) 
        importance = self.se_fc(global_feat).unsqueeze(1) 
        x = x * importance
        
        # LEARN ADJACENCY ONCE (Based on calibrated input)
        Q = self.weight_q(x)
        K = self.weight_k(x)
        A = torch.bmm(Q, K.transpose(1, 2))
        A = F.softmax(A / (x.shape[-1]**0.5), dim=-1) # (Batch, Nodes, Nodes)
        
        # Convolution Loop
        for i in range(self.num_layers):
            x = self.convs[i](x, A) # Output: (B, Nodes, Hidden)
            
            # --- SSBN TRICK for Dense ---
            # SSBN wants (Batch, Features, Nodes) to normalize Features regardless of Node pos
            x = x.transpose(1, 2)  # (B, Hidden, Nodes)
            x = self.norms[i](x, subject_ids_graph) # Normalize using GRAPH IDs (One ID per sample in batch)
            x = x.transpose(1, 2)  # Back to (B, Nodes, Hidden)
            # ---------------------------
            
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=0.5, training=self.training)
        
        # Pooling
        atten_scores = torch.tanh(self.gate(x)) # (B, N, 1)
        atten_weights = F.softmax(atten_scores, dim=1)
        embedding = (x * atten_weights).sum(dim=1) # (B, H)
        
        logits = self.fc(embedding)
        logits = logits + self.subject_bias(subject_ids_graph)
        
        if return_embedding:
            return logits, embedding
        return logits