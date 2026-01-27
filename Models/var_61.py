import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SSBN(nn.Module):
    # CHECKLIST #16: Subject-Specific Batch Normalization
    def __init__(self, num_features, num_subjects=15):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(num_subjects + 1)])

    def forward(self, x, subject_ids):
        # x: (Batch, Nodes, Features)
        out = torch.zeros_like(x)
        for s_id in torch.unique(subject_ids):
            mask = (subject_ids == s_id)
            b, n, f = x[mask].shape
            out[mask] = self.bns[int(s_id)](x[mask].view(-1, f)).view(b, n, f)
        return out

class SEBlock(nn.Module):
    # CHECKLIST #18: Squeeze-and-Excitation (Feature Weighting)
    def __init__(self, num_nodes, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_nodes, num_nodes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_nodes // reduction, num_nodes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, Nodes, Features)
        avg_pool = torch.mean(x, dim=2) # Pool features
        weight = self.fc(avg_pool).unsqueeze(2)
        return x * weight

class GatedFASHead(nn.Module):
    # CHECKLIST #14: Gated FAS Head (Asymmetry Veto)
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 3), nn.Sigmoid())

    def forward(self, logits, fas_val):
        return logits * self.gate(fas_val)

class Attempt61_Network(nn.Module):
    def __init__(self, in_channels=11, hidden_dim=128, num_subjects=15):
        super().__init__()
        self.ssbn = SSBN(in_channels, num_subjects) # CHECKLIST #16
        self.se_block = SEBlock(num_nodes=62)       # CHECKLIST #18
        
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.subject_bias = nn.Embedding(num_subjects + 1, 3) # CHECKLIST #17
        self.gated_head = GatedFASHead(hidden_dim)           # CHECKLIST #14
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index, subject_ids):
        # 1. Subject-Specific Normalization
        x = self.ssbn(x, subject_ids)
        
        # 2. Extract FAS (11th feature) for the Gated Head
        fas_val = x[:, :, 10].mean(dim=1, keepdim=True)
        
        # 3. Channel Attention
        x = self.se_block(x)
        
        # 4. Graph Convolution (Flatten batch for PyG)
        batch_size = x.size(0)
        x_flat = x.view(-1, x.size(2))
        h = F.relu(self.conv1(x_flat, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        # 5. Global Mean Pooling
        h_graph = h.view(batch_size, 62, -1).mean(dim=1)
        
        # 6. Final Logic (Logits + Bias + FAS Gate)
        logits = self.fc(h_graph)
        logits = self.gated_head(logits, fas_val)
        bias = self.subject_bias(subject_ids) # CHECKLIST #17
        
        return logits + bias