# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, AttentionalAggregation

# class AdaptiveGraphInputLayer(nn.Module):
#     """
#     Calibrates individual electrodes before neighborhood aggregation.
#     Learns to 'mute' or 'boost' sensors based on subject-specific noise.
#     """
#     def __init__(self, num_nodes=62, in_features=10):
#         super(AdaptiveGraphInputLayer, self).__init__()
#         # Learnable scaling (gamma) and shift (beta) for every electrode-feature pair
#         self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
#         self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))

#     def forward(self, x):
#         # x expected shape: (Batch, Nodes, Features)
#         return x * self.gamma + self.beta

# class SEBlock(nn.Module):
#     """
#     Squeeze-and-Excitation Block for Feature Attention.
#     Determines which frequency bands/variance features are most discriminative.
#     """
#     def __init__(self, channels, reduction=4):
#         super(SEBlock, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: (NodesInBatch, Features)
#         # We treat each node as a sample to determine feature importance
#         y = self.fc(x) 
#         return x * y

# class GraphSAGE_EEG_Model(nn.Module):
#     def __init__(self, num_nodes=62, in_features=10, hidden_dim=64, num_classes=3, 
#                  aggregator='max', use_se=True, dropout_rate=0.5):
#         super(GraphSAGE_EEG_Model, self).__init__()
        
#         self.use_se = use_se
#         self.num_nodes = num_nodes

#         # 1. THE ADAPTIVE INPUT LAYER (The Sensor Pre-Amp)
#         self.agli = AdaptiveGraphInputLayer(num_nodes, in_features)
        
#         # 2. THE SE BLOCK (Optional Band Attention)
#         if self.use_se:
#             self.se_block = SEBlock(in_features)

#         # 3. NORMALIZATION & SAGE LAYERS
#         self.input_norm = nn.LayerNorm(in_features)
        
#         # First Layer: Local Aggregation
#         self.sage1 = SAGEConv(in_features, hidden_dim, aggr=aggregator)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
        
#         # Second Layer: Higher-order neighborhood refinement
#         self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)

#         # 4. GLOBAL READOUT (Graph-level attention)
#         gate_nn = nn.Linear(hidden_dim, 1)
#         self.global_pool = AttentionalAggregation(gate_nn)
        
#         # 5. CLASSIFIER
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )

#     def forward(self, x, edge_index, batch, return_embedding=False):
#         """
#         Input x shape: (Batch*62, Features) or (Batch, 62, Features) 
#         depending on how your training_utils delivers it.
#         """
#         # --- PHASE 1: SENSOR CALIBRATION (AGLI) ---
#         # If x is flattened (NodesInBatch, Features), reshape to (Batch, 62, Features)
#         if x.dim() == 2:
#             num_graphs = x.size(0) // self.num_nodes
#             x = x.view(num_graphs, self.num_nodes, -1)
            
#         x = self.agli(x)
        
#         # Flatten back for PyG SAGEConv layers: (Batch*62, Features)
#         x = x.view(-1, x.size(-1))

#         # --- PHASE 2: BAND ATTENTION (SE) ---
#         if self.use_se:
#             x = self.se_block(x)

#         # --- PHASE 3: NEIGHBORHOOD AGGREGATION ---
#         x = self.input_norm(x)
        
#         x = self.sage1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.3, training=self.training)
        
#         x = self.sage2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
        
#         # --- PHASE 4: GLOBAL BRAIN STATE POOLING ---
#         graph_embedding = self.global_pool(x, batch)
        
#         logits = self.classifier(graph_embedding)
        
#         if return_embedding:
#             return logits, graph_embedding
#         return logits
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, AttentionalAggregation

class AdaptiveGraphInputLayer(nn.Module):
    """
    Calibrates individual electrodes before neighborhood aggregation.
    Learns to 'mute' or 'boost' sensors based on subject-specific noise.
    """
    def __init__(self, num_nodes=62, in_features=10):
        super(AdaptiveGraphInputLayer, self).__init__()
        # Learnable scaling (gamma) and shift (beta) for every electrode-feature pair
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))

    def forward(self, x):
        # x expected shape: (Batch, Nodes, Features)
        return x * self.gamma + self.beta

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Feature Attention.
    Determines which frequency bands/variance features are most discriminative.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (NodesInBatch, Features)
        # We treat each node as a sample to determine feature importance
        y = self.fc(x) 
        return x * y

class GraphSAGE_EEG_Model(nn.Module):
    def __init__(self, num_nodes=62, in_features=10, hidden_dim=128, num_classes=3, num_layers=2,
                 aggregator='max', use_se=True, use_doubling=False, dropout_rate=0.5):
        super(GraphSAGE_EEG_Model, self).__init__()
        
        self.use_se = use_se
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # 1. THE ADAPTIVE INPUT LAYER (The Sensor Pre-Amp)
        self.agli = AdaptiveGraphInputLayer(num_nodes, in_features)
        
        # 2. THE SE BLOCK (Optional Band Attention)
        if self.use_se:
            self.se_block = SEBlock(in_features)

        # 3. NORMALIZATION & SAGE LAYERS
        self.input_norm = nn.LayerNorm(in_features)
        
        # Loop for Dense GCN Layers
        self.sage = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_input_dim = in_features
        current_hidden_dim = hidden_dim

        for i in range(self.num_layers):
            # Create layer
            self.sage.append(SAGEConv(current_input_dim, current_hidden_dim, aggr=aggregator))
            self.norms.append(nn.BatchNorm1d(current_hidden_dim))
            
            # Preparation for the NEXT layer:
            # The input of next layer = output of this layer
            current_input_dim = current_hidden_dim 
            
            # Logic: If doubling, multiply hidden dim for the NEXT layer's output
            if use_doubling:
                current_hidden_dim = current_hidden_dim * 2
            # else: current_hidden_dim stays the same (Standard Fixed Width)
        
        # The dimension of the final embedding is whatever the last layer output
        final_embedding_dim = current_input_dim

        # 4. GLOBAL READOUT (Graph-level attention)
        gate_nn = nn.Linear(final_embedding_dim, 1)
        self.global_pool = AttentionalAggregation(gate_nn)
        
        # 5. CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(final_embedding_dim, final_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_embedding_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch, return_embedding=False):
        """
        Input x shape: (Batch*62, Features) or (Batch, 62, Features) 
        depending on how your training_utils delivers it.
        """
        # --- PHASE 1: SENSOR CALIBRATION (AGLI) ---
        # If x is flattened (NodesInBatch, Features), reshape to (Batch, 62, Features)
        if x.dim() == 2:
            num_graphs = x.size(0) // self.num_nodes
            x = x.view(num_graphs, self.num_nodes, -1)
            
        x = self.agli(x)
        
        # Flatten back for PyG SAGEConv layers: (Batch*62, Features)
        x = x.view(-1, x.size(-1))

        # --- PHASE 2: BAND ATTENTION (SE) ---
        if self.use_se:
            x = self.se_block(x)

        # --- PHASE 3: NEIGHBORHOOD AGGREGATION ---
        x = self.input_norm(x)
        
        for i in range(self.num_layers):
            x = self.sage[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # --- PHASE 4: GLOBAL BRAIN STATE POOLING ---
        graph_embedding = self.global_pool(x, batch)
        
        logits = self.classifier(graph_embedding)
        
        if return_embedding:
            return logits, graph_embedding
        return logits