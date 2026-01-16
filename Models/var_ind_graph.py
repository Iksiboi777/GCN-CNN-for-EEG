import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, AttentionalAggregation

class GraphSAGE_EEG_Model(nn.Module):
    """
    Inductive GraphSAGE for EEG Emotion Recognition.
    Learns robust neighborhood aggregators that generalize across subjects.
    """
    def __init__(self, in_features=10, hidden_dim=64, num_classes=3, aggregator='max', dropout_rate=0.5):
        super(GraphSAGE_EEG_Model, self).__init__()
        
        # 1. Learnable Input Scaling (Your 'Adaptive' logic)
        # Helps normalize subject-specific impedance variations before aggregation
        self.input_norm = nn.LayerNorm(in_features)
        
        # 2. GraphSAGE Layers
        # 'aggr' can be 'mean', 'pool' (Max-Pooling), or 'lstm'
        self.sage1 = SAGEConv(in_features, hidden_dim, aggr=aggregator)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 3. Global Aggregator (Readout)
        # We use AttentionalAggregation to let the model decide 
        # which electrodes are most "truthful" for the final prediction.
        gate_nn = nn.Linear(hidden_dim, 1)
        self.global_pool = AttentionalAggregation(gate_nn)
        
        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch, return_embedding=False):
        """
        x: (TotalNodesInBatch, InFeatures)
        edge_index: (2, TotalEdgesInBatch)
        batch: (TotalNodesInBatch) - map each node to its graph in the batch
        """
        # Step 1: Pre-process features
        x = self.input_norm(x)
        
        # Step 2: Layer 1 Aggregation (Neighbor -> Target)
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Step 3: Layer 2 Aggregation (Refine embeddings)
        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Step 4: Global Pooling (Node Embeddings -> Graph Embedding)
        # This converts 62 node vectors into 1 single brain-state vector
        graph_embedding = self.global_pool(x, batch)
        
        # Step 5: Classification
        logits = self.classifier(graph_embedding)
        
        if return_embedding:
            return logits, graph_embedding
        return logits