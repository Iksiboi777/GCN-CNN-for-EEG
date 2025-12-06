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
        
        # Layer 1: Conv1d(1, 64, k=15, s=1, p=7) - Wider first layer
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Layer 2: Conv1d(64, 128, k=5, s=1, p=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Removed Layer 3 to simplify model
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # RESTORED DROPOUT
        self.dropout = nn.Dropout(0.5)

        # Final Linear Projection
        self.fc = nn.Linear(128, out_channels)
        self.out_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Add Gaussian Noise during training to prevent overfitting
        if self.training:
            noise = torch.randn_like(x) * 0.01 # 1% Noise
            x = x + noise
            
        # Layer 1
        x = F.relu(self.bn1(self.conv1(x))) 
        x = self.pool(x)

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x))) 
        x = self.global_pool(x).squeeze(-1) # Global pool after 2nd layer

        x = self.dropout(x) # RESTORED
        
        # Linear Projection
        x = self.fc(x) 
        x = self.out_bn(x)

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
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # RESTORED DROPOUT
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch_index):
        # GCN Layer 1
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x) # RESTORED
        
        # GCN Layer 2
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x) # RESTORED
        
        # GCN Layer 3
        x = self.gcn3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Pooling
        x = global_mean_pool(x, batch_index) 
        
        # Classifier
        x = self.fc(x)
        return x

class CNNGCNModel(nn.Module):
    def __init__(self, num_nodes=62, time_steps=400):
        super(CNNGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.cnn = Conv1dBlock(out_channels=128)
        self.gcn = GCNBlock(in_features=128, hidden_dim=128)

    def forward(self, x, edge_index, batch_index):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_nodes, 1, -1)
        
        node_embeddings = self.cnn(x) 
        out = self.gcn(node_embeddings, edge_index, batch_index)
        
        return out