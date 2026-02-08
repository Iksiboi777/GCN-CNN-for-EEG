import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class AdaptiveGraphInputLayer(nn.Module):
    """ 
    The 'Pre-Amp'. Learned scaling factors for nodes. 
    Normalizes input first to ensure stable gradients for gamma/beta.
    """
    def __init__(self, num_nodes=62, in_features=5):
        super(AdaptiveGraphInputLayer, self).__init__()
        # Shape: (1, 62, 5) to broadcast over batch
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))
        self.initial_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # x: (Batch, Nodes, Features)
        x = self.initial_norm(x)
        return x * self.gamma + self.beta

class SEBlock(nn.Module):
    """ 
    Squeeze-and-Excitation to weight frequency bands (Dense).
    Identifies which features (bands) are globally important for the current trial.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        # Reduction logic ensures we don't scale down to 0 dimensions
        reduced_dim = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, Nodes, Features)
        
        # 1. Squeeze: Global Average Pooling over Nodes -> (Batch, Features)
        global_feat = x.mean(dim=1) 
        
        # 2. Excitation: Learn weights -> (Batch, 1, Features)
        importance = self.fc(global_feat).unsqueeze(1) 
        
        # 3. Scale original input
        return x * importance

class Adaptive_DGCNN(nn.Module):
    def __init__(self, num_nodes=62, in_features=10, hidden_dim=128, 
                 num_classes=3, num_subjects=15, num_layers=2, 
                 use_se=True, use_doubling=False, dropout_rate=0.5):
        super(Adaptive_DGCNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.use_se = use_se
        self.use_doubling = use_doubling
        
        # 1. PHYSIOLOGICAL PRIORS (AGLI)
        self.adaptive_input = AdaptiveGraphInputLayer(num_nodes, in_features)
        
        # 2. SE BLOCK (Optional)
        if self.use_se:
            self.se_block = SEBlock(in_features, reduction=2)

        # 3. DYNAMIC GRAPH LEARNERS
        # We learn one structure Q/K based on initial features
        self.weight_q = nn.Linear(in_features, hidden_dim)
        self.weight_k = nn.Linear(in_features, hidden_dim)
        
        # 4. DENSE CONVOLUTIONS (Stacked Loop)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_input_dim = in_features
        current_hidden_dim = hidden_dim

        for i in range(num_layers):
            # i=0 input is raw features, else previous hidden
            
            self.convs.append(DenseGCNConv(current_input_dim, current_hidden_dim))
            # BatchNorm1d expects (Batch, Channels, Length). 
            # In our case 'Channels' = Hidden Dim.
            self.norms.append(nn.BatchNorm1d(current_hidden_dim))
            
            # Preparation for NEXT layer
            current_input_dim = current_hidden_dim
            if self.use_doubling:
                current_hidden_dim = current_hidden_dim * 2
            # else: current_hidden_dim stays fixed
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final embedding size is the output of the last layer
        final_embedding_dim = current_input_dim

        # 5. POOLING & CLASSIFICATION
        # Attention Pooling Mechanism
        self.gate = nn.Linear(final_embedding_dim, 1)
        
        self.fc = nn.Linear(final_embedding_dim, num_classes)
        
        # 6. SUBJECT BIAS
        # +1 to handle cases where subject_id might not be passed or OOB
        # Acts as a "baseline removal" tool
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0)

    def forward(self, x, subject_ids=None, return_embedding=False):
        """
        x: (Batch, Nodes, Features) - Dense Input
        subject_ids: (Batch,) - Optional
        """
        # --- PHASE 1: SENSOR CALIBRATION ---
        x = self.adaptive_input(x)
        
        if self.use_se:
            x = self.se_block(x)
        
        # --- PHASE 2: LEARN TOPOLOGY ---
        # Calculate Adjacency Matrix A dynamically
        Q = self.weight_q(x)
        K = self.weight_k(x)
        # (B, N, H) x (B, H, N) -> (B, N, N)
        A = torch.bmm(Q, K.transpose(1, 2))
        A = F.softmax(A / (x.shape[-1]**0.5), dim=-1)

        # --- PHASE 3: GRAPH CONVOLUTIONS ---
        for i in range(self.num_layers):
            x = self.convs[i](x, A) # (Batch, Nodes, Hidden)
            
            # BatchNorm expects (N, C, L). We have (B, N, H).
            # Treating 'Hidden' as 'Channels' (C) and 'Nodes' as 'Length' (L)
            x_perm = x.transpose(1, 2) # (Batch, Hidden, Nodes)
            x_perm = self.norms[i](x_perm)
            x = x_perm.transpose(1, 2) # Back to (Batch, Nodes, Hidden)
            
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        # --- PHASE 4: POOLING ---
        # Attention Pooling
        atten_scores = torch.tanh(self.gate(x)) # (B, N, 1)
        atten_weights = F.softmax(atten_scores, dim=1)
        
        # Weighted sum of nodes
        embedding = (x * atten_weights).sum(dim=1) # (B, H)
        
        logits = self.fc(embedding)

        # --- PHASE 5: ADAPTIVE BIAS ---
        if subject_ids is not None:
            logits = logits + self.subject_bias(subject_ids)
        
        if return_embedding:
            return logits, embedding
        return logits