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
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, in_features))
        self.beta = nn.Parameter(torch.zeros(1, num_nodes, in_features))

    def forward(self, x):
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
        y = self.fc(x) 
        return x * y

class GraphSAGE_EEG_Model(nn.Module):
    def __init__(self, num_nodes=62, in_features=10, hidden_dim=128, num_classes=3, num_layers=2,
                 aggregator='max', use_se=True, use_doubling=False, dropout_rate=0.5, num_subjects=15):
        super(GraphSAGE_EEG_Model, self).__init__()
        
        self.use_se = use_se
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        self.agli = AdaptiveGraphInputLayer(num_nodes, in_features)
        if self.use_se:
            self.se_block = SEBlock(in_features)
        self.input_norm = nn.LayerNorm(in_features)
        
        self.sage = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_input_dim = in_features
        current_hidden_dim = hidden_dim

        for _ in range(self.num_layers):
            self.sage.append(SAGEConv(current_input_dim, current_hidden_dim, aggr=aggregator))
            self.norms.append(nn.BatchNorm1d(current_hidden_dim))
        
            current_input_dim = current_hidden_dim 
            if use_doubling:
                current_hidden_dim = current_hidden_dim * 2
    
        final_embedding_dim = current_input_dim
        gate_nn = nn.Linear(final_embedding_dim, 1)
        self.global_pool = AttentionalAggregation(gate_nn)
        
        self.classifier = nn.Sequential(
            nn.Linear(final_embedding_dim, final_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_embedding_dim // 2, num_classes)
        )
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0) 


    def forward(self, x, edge_index, batch, subject_ids=None, return_embedding=False):
        """
        Input x shape: (Batch*62, Features) or (Batch, 62, Features) 
        depending on how your training_utils delivers it.
        """
        if x.dim() == 2:
            num_graphs = x.size(0) // self.num_nodes
            x = x.view(num_graphs, self.num_nodes, -1)
            
        x = self.agli(x)
        x = x.view(-1, x.size(-1))
        if self.use_se:
            x = self.se_block(x)
        x = self.input_norm(x)
        
        for i in range(self.num_layers):
            x = self.sage[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        graph_embedding = self.global_pool(x, batch)        
        logits = self.classifier(graph_embedding)

        if subject_ids is not None:
            logits = logits + self.subject_bias(subject_ids)
        
        if return_embedding:
            return logits, graph_embedding
        return logits