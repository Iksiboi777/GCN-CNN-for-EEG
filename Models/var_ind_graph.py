import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, AttentionalAggregation
from phase2_layers import SubjectSpecificBatchNorm1d 

class AdaptiveGraphInputLayer(nn.Module):
    def __init__(self, num_nodes=62, in_features=11):
        super(AdaptiveGraphInputLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))

    def forward(self, x):
        return x * self.gamma + self.beta

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x) 
        return x * y

class GraphSAGE_EEG_Model(nn.Module):
    def __init__(self, num_nodes=62, in_features=11, hidden_dim=64, num_classes=3, 
                 aggregator='mean', num_subjects=15, num_layers=2):
        super(GraphSAGE_EEG_Model, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # 1. PHASE 2: SSBN Input & Calibration
        self.ssbn_input = SubjectSpecificBatchNorm1d(in_features, num_subjects)
        self.agli = AdaptiveGraphInputLayer(num_nodes, in_features)
        self.se_block = SEBlock(in_features)

        # 2. Dynamic GNN Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_dim
            # SAGEConv
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))
            # Subject-Specific BN for hidden state
            self.norms.append(SubjectSpecificBatchNorm1d(hidden_dim, num_subjects))

        # 3. Global Readout
        gate_nn = nn.Linear(hidden_dim, 1)
        self.global_pool = AttentionalAggregation(gate_nn)
        
        # 4. Classifier & Bias
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0)

    def forward(self, x, edge_index, batch_index, subject_ids_full, return_embedding=False):
        # x: (Batch*62, Features)
        
        # --- Pre-GNN Calibration ---
        x = self.ssbn_input(x, subject_ids_full)
        
        # Reshape for AGLI properties (Global scale)
        num_graphs = x.size(0) // self.num_nodes
        x = x.view(num_graphs, self.num_nodes, -1)
        x = self.agli(x)
        x = x.view(-1, x.size(-1)) 
        
        x = self.se_block(x)

        # --- Stacked Graph Convolutions ---
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, subject_ids_full)
            x = F.relu(x)
            if i < self.num_layers - 1: # Dropout on all but last? Or all? Usually all.
                x = F.dropout(x, p=0.4, training=self.training)
        
        # --- Pooling & Readout ---
        embedding = self.global_pool(x, batch_index)
        logits = self.classifier(embedding)
        
        # Add Subject Bias (using graph-level IDs)
        subject_ids_graph = subject_ids_full[::self.num_nodes]
        logits = logits + self.subject_bias(subject_ids_graph)
        
        if return_embedding:
            return logits, embedding
        return logits

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