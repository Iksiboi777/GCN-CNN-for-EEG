import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import multiprocessing
import argparse
import time

from Models.var_A import CNNGCNModel
from Models.graph_construction import get_knn_adjacency_matrix

# --- Configuration ---
DATA_FOLDER = "Data/Raw_Data_For_CNN"
LOCS_FILE = "channel_62_pos.locs"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Train GCN-CNN for EEG Emotion Recognition")
    parser.add_argument('--mode', type=str, required=True, choices=['sub_dep', 'sub_indep'],
                        help="Training mode: 'subject_dependent' (Session split) or 'subject_independent' (LOSO)")
    parser.add_argument('--test_subject', type=int, default=1, 
                        help="ID of the subject to leave out for testing (only for subject_independent mode)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    return parser.parse_args()

def main():
    args = get_args()
    num_workers = min(multiprocessing.cpu_count(), 4)
    print(f"Using device: {DEVICE} with {num_workers} workers.")
    print(f"Mode: {args.mode}")

    # 1. Load Data
    print("Loading data...")
    X = np.load(os.path.join(DATA_FOLDER, "X_raw.npy")) 
    y = np.load(os.path.join(DATA_FOLDER, "y_labels.npy"))
    sessions = np.load(os.path.join(DATA_FOLDER, "sessions.npy"))
    subjects = np.load(os.path.join(DATA_FOLDER, "subjects.npy"))
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # 2. Define Split Strategy
    if args.mode == 'sub_dep':
        print("Splitting data (Train: Sess 1+2, Test: Sess 3)...")
        train_mask = (sessions == 1) | (sessions == 2)
        test_mask = (sessions == 3)
        
    elif args.mode == 'sub_indep':
        print(f"Splitting data (LOSO): Leaving out Subject {args.test_subject}...")
        train_mask = (subjects != args.test_subject)
        test_mask = (subjects == args.test_subject)
        
    # Apply Mask
    X_train = X_tensor[train_mask]
    y_train = y_tensor[train_mask]
    X_test = X_tensor[test_mask]
    y_test = y_tensor[test_mask]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=num_workers, persistent_workers=True, prefetch_factor=2)

    # 3. Construct Graph
    print("Constructing Graph...")
    edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5)
    edge_index = edge_index.to(DEVICE)
    
    # 4. Initialize Model
    model = CNNGCNModel(num_nodes=62, time_steps=400).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("Starting Training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            batch_idx = torch.arange(batch_X.size(0), device=DEVICE).repeat_interleave(62)
            
            optimizer.zero_grad()
            outputs = model(batch_X, edge_index, batch_idx)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        
        if (epoch + 1) % 5 == 0:
            evaluate(model, test_loader, edge_index, criterion)

    end_time = time.time()  # <--- End Timer
    elapsed_time = end_time - start_time
    print(f"\nTraining Finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")


def evaluate(model, loader, edge_index, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            batch_idx = torch.arange(batch_X.size(0), device=DEVICE).repeat_interleave(62)
            
            outputs = model(batch_X, edge_index, batch_idx)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    print(f"Validation Loss: {val_loss/len(loader):.4f}, Validation Acc: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()