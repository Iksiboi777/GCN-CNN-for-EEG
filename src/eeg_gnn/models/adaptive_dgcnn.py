import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class AdaptiveGraphInputLayer(nn.Module):
    """
    'Pre-Amp' jedinica. Normalizira ulaze i primjenjuje naučene faktore skaliranja.
    Služi za stabilizaciju gradijenata prije same konvolucije.
    """
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
    """
    Squeeze-and-Excitation za rekalibraciju frekvencijskih pojaseva.
    Pomaže modelu da se fokusira na relevantne značajke prije učenja topologije.
    """
    def __init__(self, channels, reduction=2):
        super(SEBlock, self).__init__()
        reduced_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        global_feat = x.mean(dim=1) 
        importance = self.fc(global_feat).unsqueeze(1) 
        return x * importance



class Adaptive_DGCNN(nn.Module):
    def __init__(self, static_adj, num_nodes=62, in_features=10, hidden_dim=128, 
                 num_classes=3, num_subjects=15, num_layers=2, 
                 use_se=True, dropout_rate=0.5, use_doubling=False):
        super(Adaptive_DGCNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.use_se = use_se
        self.use_doubling = use_doubling
        self.register_buffer('static_adj', static_adj)
        
        self.learnable_alpha = nn.Parameter(torch.tensor(0.5))
        self.adaptive_input = AdaptiveGraphInputLayer(num_nodes, in_features)
        if self.use_se:
            self.se_block = SEBlock(in_features, reduction=2)

        self.weight_q = nn.Linear(in_features, hidden_dim // 2)
        self.weight_k = nn.Linear(in_features, hidden_dim // 2)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        curr_in = in_features
        for _ in range(num_layers):
            self.convs.append(DenseGCNConv(curr_in, hidden_dim))
            self.norms.append(nn.BatchNorm1d(num_nodes))
            curr_in = hidden_dim
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gate = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.subject_bias = nn.Embedding(num_subjects + 1, num_classes)
        self.subject_bias.weight.data.fill_(0.0)


    def normalize_adjacency(self, A):
        """ Simetrična normalizacija matrice susjedstva za stabilnost GCN-a """
        row_sum = A.sum(dim=-1)
        d_inv_sqrt = torch.pow(row_sum + 1e-6, -0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x, subject_ids=None):
        x = self.adaptive_input(x)
        if self.use_se:
            x = self.se_block(x)
        Q = self.weight_q(x)
        K = self.weight_k(x)
        
        A_dyn = torch.bmm(Q, K.transpose(1, 2))
        A_dyn = F.relu(A_dyn)
        alpha = torch.sigmoid(self.learnable_alpha)
        A = (1 - alpha) * self.static_adj + alpha * A_dyn

        A = self.normalize_adjacency(A)

        for i in range(self.num_layers):
            x = self.convs[i](x, A)
            x = self.norms[i](x) 
            x = F.relu(x)
            x = self.dropout(x)
        atten_scores = torch.tanh(self.gate(x))
        atten_weights = F.softmax(atten_scores, dim=1)
        embedding = (x * atten_weights).sum(dim=1)
        
        logits = self.fc(embedding)
        if subject_ids is not None:
            logits = logits + self.subject_bias(subject_ids)
            
        return logits