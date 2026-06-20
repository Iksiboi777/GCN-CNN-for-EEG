import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention, AttentionalAggregation
from torch_geometric.nn import global_mean_pool

class AdaptiveGraphInputLayer(nn.Module):
    """
    Learnable Input Normalization Layer.
    Applies a learnable affine transformation per node (channel) and per feature (band).
    CRITICAL CHANGE: Removed LayerNorm to allow gamma to fully silence noisy channels.
    """
    def __init__(self, num_nodes=62, in_features=5):
        super(AdaptiveGraphInputLayer, self).__init__()
        # Initialize gamma at 0.5 (conservative start) and beta at 0
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features) * 0.5)
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))

    def forward(self, x):
        # x shape: (Batch*Nodes, Features)
        batch_size = x.size(0) // 62
        x_reshaped = x.view(batch_size, 62, -1)
        
        # Apply Learnable Affine Transformation
        # Because we removed LayerNorm, if gamma goes to 0, the output effectively becomes 0.
        x_adapted = x_reshaped * self.gamma + self.beta
        
        return x_adapted.view(-1, x.size(-1))

class SEBlock(nn.Module):
    """
    Dynamic Squeeze-and-Excitation Block using Attention Pooling.
    """
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        
        # Use GlobalAttention instead of MeanPooling for the "Squeeze" step
        # This allows the SE block to ignore noise when calculating the global context
        gate_nn = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        self.att_pool = GlobalAttention(gate_nn)
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x, batch_index):
        # 1. Smarter Squeeze: Attention Pooling
        # x: (Batch*Nodes, Features) -> global_feat: (Batch, Features)
        global_feat = self.att_pool(x, batch_index)
        
        # 2. Excitation
        w = F.relu(self.fc1(global_feat))
        w = torch.sigmoid(self.fc2(w))
        
        # 3. Scale
        return x * w[batch_index]




# ADAPTIVE SUBJECT BIAS MODEL

class GCN_DE_Model(nn.Module):
    def __init__(self, num_nodes=62, in_features=10, hidden_dim=128, 
                 num_classes=3, dropout_rate=0.5, num_layers=2, num_subjects=15,
                 use_overlap_logic=False, use_doubling=False, use_se=False): # Added num_subjects
        super(GCN_DE_Model, self).__init__()
        self.use_overlap_logic = use_overlap_logic
        self.use_se = use_se
        # 1. Static Adaptation (Silencing Bad Sensors)
        self.static_norm = AdaptiveGraphInputLayer(num_nodes, in_features)
        
        # 2. THE SE BLOCK (Optional Band Attention)
        if self.use_se:
            self.se_block = SEBlock(in_features, reduction=2)
        
        # 3. GCN Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        
        current_input_dim = in_features
        current_hidden_dim = hidden_dim

        for i in range(num_layers):
            # Create layer
            self.layers.append(GCNConv(current_input_dim, current_hidden_dim))
            
            # Apply Normalization logic
            if self.use_overlap_logic:
                # Better for Overlapping/Correlated windows
                self.norms.append(nn.LayerNorm(current_hidden_dim))
            else:
                # Better for Static/Independent 4s windows
                self.norms.append(nn.BatchNorm1d(current_hidden_dim))
            
            # Preparation for the NEXT layer:
            current_input_dim = current_hidden_dim 
            
            # Logic: If doubling, multiply hidden dim for the NEXT layer's output
            if use_doubling:
                current_hidden_dim = current_hidden_dim * 2
            # else: current_hidden_dim stays the same (Standard Fixed Width)
        
        # The dimension of the final embedding is whatever the last layer output
        final_embedding_dim = current_input_dim
            
        # 4. Final Pooling: Also Attention-based
        gate_nn_final = nn.Sequential(nn.Linear(final_embedding_dim, 1), nn.Sigmoid())
        self.final_pool = AttentionalAggregation(gate_nn_final)
        # self.final_pool = GlobalAttention(gate_nn_final)
        
        self.fc = nn.Linear(final_embedding_dim, num_classes)
        
        # --- NEW: SUBJECT BIAS ---
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0) # Init to zero
        # -------------------------

        self.dropout = nn.Dropout(dropout_rate)

        self._set_weights()

    def _set_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, edge_index, batch_index, subject_ids=None, return_embedding=False, return_attention=False):
        # 1. Preprocessing
        x = self.static_norm(x)
        if self.use_se:
            x = self.se_block(x, batch_index)
        
        # 2. Convolution
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x = conv(x, edge_index)
            # Apply Normalization based on the detected logic
            if self.use_overlap_logic:
                # LayerNorm (N*62, Hidden)
                x = norm(x)
                x = F.gelu(x)
            else:
                # BatchNorm (N*62, Hidden)
                x = norm(x)
                x = F.relu(x)

            if i < len(self.layers) - 1:
                x = self.dropout(x)
        
        # 3. Aggregation (Replaced MeanPool with AttPool)
        if return_attention:
            attention_logits = self.final_pool.gate_nn(x)
            attention_weights = attention_logits.squeeze() 
            embedding = self.final_pool(x, batch_index)
            logits = self.fc(embedding)
            
            # --- NEW: APPLY BIAS ---
            if subject_ids is not None:
                logits = logits + self.subject_bias(subject_ids)
            # -----------------------

            return logits, attention_weights
            
        embedding = self.final_pool(x, batch_index)
        
        # 4. Classification
        logits = self.fc(embedding)

        # --- NEW: APPLY BIAS ---
        if subject_ids is not None:
            logits = logits + self.subject_bias(subject_ids)
        # -----------------------
        
        if return_embedding:
            return logits, embedding
        return logits