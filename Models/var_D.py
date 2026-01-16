import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, GlobalAttention

class AdaptiveGraphInputLayer(nn.Module):
    """ The 'Pre-Amp' from var_B. Fixes Sinkholes and Screamers. """
    def __init__(self, num_nodes=62, in_features=5):
        super(AdaptiveGraphInputLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
        self.initial_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # x shape: (Batch, Nodes, Features)
        x = self.initial_norm(x)
        return x * self.gamma + self.beta

class SEBlock(nn.Module):
    """ The 'Band-Selector' from var_B. Chooses which freq bands to trust. """
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, Nodes, Features)
        b, n, f = x.size()
        y = x.transpose(1, 2) # (Batch, Features, Nodes)
        y = self.avg_pool(y).view(b, f)
        y = self.fc(y).view(b, 1, f)
        return x * y.expand_as(x)

class Adaptive_DGCNN(nn.Module):
    def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, 
                 num_classes=3, dropout_rate=0.5, num_layers=3):
        super(Adaptive_DGCNN, self).__init__()
        
        # 1. THE GATEKEEPERS (From var_B)
        self.adaptive_input = AdaptiveGraphInputLayer(num_nodes, in_features)
        self.se_block = SEBlock(in_features)

        # 2. DYNAMIC GRAPH CONSTRUCTION (Original var_C logic)
        self.weight_q = nn.Linear(in_features, hidden_dim)
        self.weight_k = nn.Linear(in_features, hidden_dim)
        
        # 3. DENSE CONVOLUTIONS
        self.conv1 = DenseGCNConv(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_nodes)

        # 4. THE DICTATOR POOLING (From var_B)
        # Using GlobalAttention instead of MeanPool
        self.att_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.att_gate)
        
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Nodes, Features)
        
        # --- PHASE 1: SENSOR CALIBRATION ---
        x = self.adaptive_input(x) # Scale the sinkholes
        x = self.se_block(x)      # Weight the bands
        
        # --- PHASE 2: LEARN THE TOPOLOGY ---
        Q = self.weight_q(x)
        K = self.weight_k(x)
        A = torch.bmm(Q, K.transpose(1, 2))
        A = F.softmax(A / (x.shape[-1]**0.5), dim=-1) # Dynamic Adjacency

        # --- PHASE 3: SPATIAL PROCESSING ---
        x = self.conv1(x, A)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, A)
        x = self.bn2(x)
        x = F.relu(x)

        # --- PHASE 4: THE DICTATOR POOL ---
        # Note: GlobalAttention expects BatchIndex for flattened tensors.
        # For Dense input (Batch, Nodes, Hidden), we can simulate it.
        b, n, h = x.shape
        x_flat = x.reshape(-1, h)
        batch_idx = torch.arange(b, device=x.device).repeat_interleave(n)
        
        embedding = self.pool(x_flat, batch_idx)
        return self.fc(embedding)





# # ONLY AGLI


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import DenseGCNConv, GlobalAttention

# class AdaptiveGraphInputLayer(nn.Module):
#     """ The 'Pre-Amp' from var_B. Fixes Sinkholes and Screamers. """
#     def __init__(self, num_nodes=62, in_features=5):
#         super(AdaptiveGraphInputLayer, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
#         self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
#         self.initial_norm = nn.LayerNorm(in_features)

#     def forward(self, x):
#         # x shape: (Batch, Nodes, Features)
#         x = self.initial_norm(x)
#         return x * self.gamma + self.beta

# # SEBlock removed to isolate AdaptiveGraphInputLayer impact

# class Adaptive_DGCNN(nn.Module):
#     def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, 
#                  num_classes=3, dropout_rate=0.5, num_layers=3):
#         super(Adaptive_DGCNN, self).__init__()
        
#         # 1. THE GATEKEEPER (Only Adaptive Input now)
#         self.adaptive_input = AdaptiveGraphInputLayer(num_nodes, in_features)
#         # SEBlock removed

#         # 2. DYNAMIC GRAPH CONSTRUCTION (Original var_C logic)
#         self.weight_q = nn.Linear(in_features, hidden_dim)
#         self.weight_k = nn.Linear(in_features, hidden_dim)
        
#         # 3. DENSE CONVOLUTIONS
#         self.conv1 = DenseGCNConv(in_features, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(num_nodes)
        
#         self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(num_nodes)

#         # 4. THE DICTATOR POOLING (From var_B)
#         # Using GlobalAttention instead of MeanPool
#         self.att_gate = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#         self.pool = GlobalAttention(gate_nn=self.att_gate)
        
#         self.fc = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         # x: (Batch, Nodes, Features)
        
#         # --- PHASE 1: SENSOR CALIBRATION ---
#         x = self.adaptive_input(x) # Scale the sinkholes
#         # No SEBlock
        
#         # --- PHASE 2: LEARN THE TOPOLOGY ---
#         Q = self.weight_q(x)
#         K = self.weight_k(x)
#         A = torch.bmm(Q, K.transpose(1, 2))
#         A = F.softmax(A / (x.shape[-1]**0.5), dim=-1) # Dynamic Adjacency

#         # --- PHASE 3: SPATIAL PROCESSING ---
#         x = self.conv1(x, A)
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.conv2(x, A)
#         x = self.bn2(x)
#         x = F.relu(x)

#         # --- PHASE 4: THE DICTATOR POOL ---
#         # Note: GlobalAttention expects BatchIndex for flattened tensors.
#         # For Dense input (Batch, Nodes, Hidden), we can simulate it.
#         b, n, h = x.shape
#         x_flat = x.reshape(-1, h)
#         batch_idx = torch.arange(b, device=x.device).repeat_interleave(n)
        
#         embedding = self.pool(x_flat, batch_idx)
#         return self.fc(embedding)