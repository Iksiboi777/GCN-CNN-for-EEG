import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class Conv1dBlock(nn.Module):
    """
    Per-Channel CNN (Shared Weights).
    Input: (Batch * Nodes, 1, Time_Steps)
    Output: (Batch * Nodes, Embedding_Dim)
    """
    def __init__(self, in_channels=1, out_channels=64):
        super(Conv1dBlock, self).__init__()
        
        # Layer 1: Conv1d(1, 32, k=7, s=2, p=3)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Layer 2: Conv1d(32, 64, k=5, s=2, p=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Final Linear Projection
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        # x shape: (Batch * Nodes, 1, 400)
        
        x = F.relu(self.bn1(self.conv1(x))) # -> (Batch*Nodes, 32, 200)
        x = F.relu(self.bn2(self.conv2(x))) # -> (Batch*Nodes, 64, 100)
        
        # Global Average Pooling over time
        x = x.mean(dim=2) # -> (Batch*Nodes, 64)
        
        # Linear Projection
        x = self.fc(x) # -> (Batch*Nodes, 64)
        return x

class GCNBlock(nn.Module):
    """
    GCN Part.
    Input: Node Embeddings (Batch * Nodes, Feature_Dim) + Edge Index
    Output: Graph Classification Logits (Batch, Num_Classes)
    """
    def __init__(self, in_features=64, hidden_dim=64, num_classes=3):
        super(GCNBlock, self).__init__()
        
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch_index):
        # x: (Total_Nodes, 64)
        # edge_index: (2, Total_Edges)
        # batch_index: (Total_Nodes,) - tells which node belongs to which graph in the batch
        
        # GCN Layer 1
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN Layer 2
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global Pooling (Aggregates node features to 1 graph vector)
        x = global_mean_pool(x, batch_index) # -> (Batch_Size, 64)
        
        # Classifier
        x = self.fc(x) # -> (Batch_Size, 3)
        return x

class CNNGCNModel(nn.Module):
    def __init__(self, num_nodes=62, time_steps=400):
        super(CNNGCNModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.cnn = Conv1dBlock()
        self.gcn = GCNBlock()

    def forward(self, x, edge_index, batch_index):
        """
        x: (Batch_Size, Num_Nodes, Time_Steps) -> Raw EEG
        edge_index: (2, Num_Edges) -> Graph Connectivity
        batch_index: (Batch_Size * Num_Nodes,) -> Batch vector for PyG
        """
        batch_size = x.size(0)
        
        # 1. Reshape for CNN: Treat every node as an independent sample
        # (Batch, Nodes, Time) -> (Batch * Nodes, 1, Time)
        x = x.view(batch_size * self.num_nodes, 1, -1)
        
        # 2. Extract Temporal Features
        node_embeddings = self.cnn(x) # -> (Batch * Nodes, 64)
        
        # 3. Apply GCN
        out = self.gcn(node_embeddings, edge_index, batch_index)
        
        return out