import torch
import numpy as np
import os
import copy
import time
import inspect
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def is_dense_model(model):
    """Detects Dense vs Sparse models based on class name or structure."""
    name = model.__class__.__name__.lower()
    return "dgcnn" in name or "dense" in name

def train_epoch(model, loader, optimizer, criterion, device, base_edge_index, in_features):
    model.train()
    total_loss, correct, total = 0, 0, 0
    dense_mode = is_dense_model(model)
    
    for batch_data in loader:
        # 1. Unpack Memory-Mapped Batch
        # The CustomDataset returns: (X, y, subject_id)
        batch_X, batch_y, batch_sub = batch_data
        
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_sub = batch_sub.to(device)
        
        # Add noise during training for robustness
        batch_X = batch_X + torch.randn_like(batch_X) * 0.01

        optimizer.zero_grad()
        
        if dense_mode:
            # Dense Models expect: (Batch, Nodes, Features)
            # Pass subject_ids_graph directly (1 ID per sample in batch)
            outputs = model(batch_X, subject_ids_graph=batch_sub)
        else:
            # Sparse Models (GCN, GraphSAGE) expect: (TotalNodes, Features)
            # We must Flatten: (Batch, Note, Feat) -> (Batch*Node, Feat)
            curr_batch_size, num_nodes, _ = batch_X.shape
            
            # Flatten X
            batch_X_flat = batch_X.view(-1, in_features)
            
            # Create Batch Index (0,0,0... 1,1,1...)
            batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(num_nodes)
            
            # Expand Subject IDs (Sample 1 is Subj A -> 62 nodes of Subj A)
            # Input batch_sub is (Batch,), we need (Batch*62,)
            batch_sub_full = batch_sub.repeat_interleave(num_nodes)

            # Create Shifted Edge Index
            # base_edge_index is (2, NumEdges). We need to shift it for every graph in batch.
            offsets = (torch.arange(curr_batch_size, device=device) * num_nodes).view(-1, 1, 1)
            edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
            outputs = model(batch_X_flat, edge_index, batch_idx, subject_ids_full=batch_sub_full)
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient Clipping (Important for deep GCNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader, base_edge_index, criterion, device, in_features, 
             return_preds=False, return_embeddings=False):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    dense_mode = is_dense_model(model)

    with torch.no_grad():
        for batch_data in loader:
            batch_X, batch_y, batch_sub = batch_data
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_sub = batch_sub.to(device)
            
            if dense_mode:
                outputs = model(batch_X, subject_ids_graph=batch_sub)
            else:
                curr_batch_size, num_nodes, _ = batch_X.shape
                batch_X_flat = batch_X.view(-1, in_features)
                batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(num_nodes)
                batch_sub_full = batch_sub.repeat_interleave(num_nodes)
                offsets = (torch.arange(curr_batch_size, device=device) * num_nodes).view(-1, 1, 1)
                edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
                
                outputs = model(batch_X_flat, edge_index, batch_idx, subject_ids_full=batch_sub_full)
            
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
    acc = 100 * correct / total
    avg_loss = val_loss / len(loader) if len(loader) > 0 else 0
    
    if return_preds:
        return avg_loss, acc, np.array(all_preds), np.array(all_labels)
    return avg_loss, acc