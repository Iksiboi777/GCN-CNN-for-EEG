import torch
import torch.nn as nn

class SubjectSpecificBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_subjects=15):
        super(SubjectSpecificBatchNorm1d, self).__init__()
        # Create a list of BN layers, one for each subject
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_features) for _ in range(num_subjects + 1)
        ])
        
    def forward(self, x, subject_ids):
        # x: (Batch, Features) or (Batch, Features, Nodes)
        # We need to process sample-by-sample or group-by-group, which is slow in pure PyTorch loops.
        # Faster approach: Split input by subject, apply respective BN, recombine.
        
        output = torch.zeros_like(x)
        unique_subs = torch.unique(subject_ids)
        
        for sub_id in unique_subs:
            mask = (subject_ids == sub_id)
            # Apply the specific BN for this subject
            output[mask] = self.bns[sub_id](x[mask])
            
        return output