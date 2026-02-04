# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool

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

# class GCN_DE_Model(nn.Module):
#     """
#     Dynamic GCN Model for Differential Entropy (DE) Features.
#     Input: (Batch * Nodes, 5) -> 5 Frequency Bands per Node
#     Output: (Batch, Num_Classes)
#     """
#     def __init__(self, num_nodes=62, in_features=5, hidden_dim=64, num_classes=3, dropout_rate=0.5, num_layers=3):
#         super(GCN_DE_Model, self).__init__()
        
#         # --- NEW: Insert Adaptive Layer ---
#         self.input_norm = AdaptiveGraphInputLayer(num_nodes, in_features)
#         # ----------------------------------
        
#         self.layers = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.dropout_rate = dropout_rate
        
#         # Input Layer (Layer 1)
#         self.layers.append(GCNConv(in_features, hidden_dim))
#         self.bns.append(nn.BatchNorm1d(hidden_dim))
        
#         # Hidden Layers (Layer 2 to N)
#         for _ in range(num_layers - 1):
#             self.layers.append(GCNConv(hidden_dim, hidden_dim))
#             self.bns.append(nn.BatchNorm1d(hidden_dim))
            
#         # Classifier
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, return_embedding=False):
#         # x shape: (Batch * Nodes, 5)
        
#         # --- NEW: Apply Adaptive Normalization ---
#         x = self.input_norm(x)
#         # -----------------------------------------
        
#         for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.relu(x)
            
#             # Apply dropout to all layers except the last GCN layer
#             if i < len(self.layers) - 1:
#                 x = self.dropout(x)
        
#         # Global Pooling (aggregates nodes for each graph in the batch)
#         # Output: (Batch_Size, Hidden_Dim)
#         embedding = global_mean_pool(x, batch_index) 
        
#         # Final Classification
#         out = self.fc(embedding)
        
#         if return_embedding:
#             return out, embedding
#         return out




# from colorama import init
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention, AttentionalAggregation

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

# class GCN_DE_Model(nn.Module):
#     def __init__(self, num_nodes=62, in_features=10, hidden_dim=64, 
#                  num_classes=3, dropout_rate=0.5, num_layers=3):
#         super(GCN_DE_Model, self).__init__()
        
#         # 1. Static Adaptation (Silencing Bad Sensors)
#         self.static_norm = AdaptiveGraphInputLayer(num_nodes, in_features)
        
#         # 2. Dynamic Adaptation (Focusing on Good Bands)
#         self.se_block = SEBlock(in_features, reduction=2)
        
#         # 3. GCN Layers
#         self.layers = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.dropout_rate = dropout_rate
        
#         self.layers.append(GCNConv(in_features, hidden_dim))
#         self.bns.append(nn.BatchNorm1d(hidden_dim))
        
#         for _ in range(num_layers - 1):
#             self.layers.append(GCNConv(hidden_dim, hidden_dim))
#             self.bns.append(nn.BatchNorm1d(hidden_dim))
            
#         # 4. Final Pooling: Also Attention-based
#         # We use a separate attention gate for the final classification
#         gate_nn_final = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
#         self.final_pool = GlobalAttention(gate_nn_final)
        
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, edge_index, batch_index, return_embedding=False, return_attention=False):
#         # 1. Preprocessing
#         x = self.static_norm(x)
#         x = self.se_block(x, batch_index)
        
#         # 2. Convolution
#         for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.relu(x)
#             if i < len(self.layers) - 1:
#                 x = self.dropout(x)
        
#         # 3. Aggregation (Replaced MeanPool with AttPool)

#         # --- NEW BLOCK: EXTRACT ATTENTION WEIGHTS ---
#         if return_attention:
#             # We manually pass x through the gating network associated with the final pool
#             # x shape: (TotalNodesInBatch, HiddenDim)
#             # gate_nn output: (TotalNodesInBatch, 1) -> The "Importance" score (0 to 1)
#             attention_logits = self.final_pool.gate_nn(x)
#             attention_weights = attention_logits.squeeze() # Shape: (TotalNodesInBatch)
            
#             # Continue with normal pooling
#             embedding = self.final_pool(x, batch_index)
#             out = self.fc(embedding)
#             return out, attention_weights
#         embedding = self.final_pool(x, batch_index)
        
#         # 4. Classification
#         out = self.fc(embedding)
        
#         if return_embedding:
#             return out, embedding
#         return out


from torch_geometric.nn import global_mean_pool
from utils.phase2_layers import SubjectSpecificBatchNorm1d


class SEBlock_Global(nn.Module):
    """ Global SE Block: Pools the whole graph to decide feature importance """
    def __init__(self, in_channels, reduction=4):
        super(SEBlock_Global, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, batch_index):
        # Global pooling to get (Batch, Features)
        # We effectively ask: "What is the dominant frequency in this trial?"
        global_avg = global_mean_pool(x, batch_index) 
        scale = self.fc(global_avg)
        # Broadcast back to nodes
        return x * scale[batch_index]



# class GCN_DE_Model(nn.Module):
#     def __init__(self, num_nodes=62, in_features=11, hidden_dim=64, 
#                  num_classes=3, num_subjects=15, num_layers=2):
#         super(GCN_DE_Model, self).__init__()
        
#         self.num_nodes = num_nodes
#         self.num_layers = num_layers
        
#         # 1. Main Backbone Components
#         self.ssbn_input = SubjectSpecificBatchNorm1d(in_features, num_subjects) 
#         self.agli = AdaptiveGraphInputLayer(num_nodes, in_features)
#         self.se_block = SEBlock_Global(in_features)
        
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         for i in range(num_layers):
#             in_dim = in_features if i == 0 else hidden_dim
#             self.convs.append(GCNConv(in_dim, hidden_dim))
#             self.norms.append(SubjectSpecificBatchNorm1d(hidden_dim, num_subjects))
        
#         gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
#         self.pool = GlobalAttention(gate_nn)
        
#         # 2. FAS Specialized Head (The "Veto" Mechanism)
#         # It takes the mean FAS value across the brain (Feature index 10)
#         self.fas_classifier = nn.Sequential(
#             nn.Linear(1, 8),
#             nn.ReLU(),
#             nn.Linear(8, num_classes)
#         )
#         # Use a learnable gate to decide how much to trust FAS vs GCN
#         self.fusion_gate = nn.Parameter(torch.tensor(0.5)) 

#         self.backbone_classifier = nn.Linear(hidden_dim, num_classes)
#         self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
#         self.subject_bias.weight.data.fill_(0.0)

#     def forward(self, x, edge_index, batch_index, subject_ids_full, return_embedding=False):
#         # A. EXTRACT RAW FAS (Feature 10) before mixing
#         # x is (Batch*Nodes, 11). FAS is index 10.
#         # We average FAS over every graph (subject state)
#         raw_fas = x[:, 10].unsqueeze(1) # (TotalNodes, 1)
#         graph_fas = global_mean_pool(raw_fas, batch_index) # (Batch, 1)
#         fas_logits = self.fas_classifier(graph_fas)

#         # B. MAIN BACKBONE FLOW
#         x = self.ssbn_input(x, subject_ids_full)
#         x = self.agli(x)
#         x = self.se_block(x, batch_index)
        
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             x = self.norms[i](x, subject_ids_full)
#             x = F.relu(x)
#             if i < self.num_layers - 1:
#                 x = F.dropout(x, p=0.5, training=self.training)
        
#         embedding = self.pool(x, batch_index)
#         gcn_logits = self.backbone_classifier(embedding)
        
#         # C. HYBRID FUSION (Gated Sum)
#         subject_ids_graph = subject_ids_full[::self.num_nodes]
#         bias = self.subject_bias(subject_ids_graph)
        
#         # Final Logits = GCN + (Gate * FAS) + SubjectBias
#         # Limits FAS influence to prevent it from overriding everything
#         gate = torch.sigmoid(self.fusion_gate) 
#         final_logits = gcn_logits + (gate * fas_logits) + bias
        
#         if return_embedding:
#             return final_logits, embedding
#         return final_logits



# ADAPTIVE SUBJECT BIAS MODEL

class GCN_DE_Model(nn.Module):
    def __init__(self, num_nodes=62, in_features=10, hidden_dim=128, 
                 num_classes=3, dropout_rate=0.5, num_layers=2, num_subjects=15,
                 use_overlap_logic=False, use_doubling=False): # Added num_subjects
        super(GCN_DE_Model, self).__init__()
        self.use_overlap_logic = use_overlap_logic
        # 1. Static Adaptation (Silencing Bad Sensors)
        self.static_norm = AdaptiveGraphInputLayer(num_nodes, in_features)
        
        # 2. Dynamic Adaptation (Focusing on Good Bands)
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