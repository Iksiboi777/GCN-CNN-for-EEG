import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import argparse
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from Models.var_A import CNNGCNModel
from Models.graph_construction import get_knn_adjacency_matrix

# --- Configuration ---
DATA_FOLDER = "Data/Raw_Data_w_Bands"
OUTPUT_FOLDER = "Models/Trained_Models_w_Bands"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LOCS_FILE = "channel_62_pos.locs"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0005 # Kept low
WEIGHT_DECAY = 1e-4
PATIENCE = 15 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Train GCN-CNN for EEG Emotion Recognition")
    parser.add_argument('--mode', type=str, required=True, choices=['sub_dep', 'sub_indep'],
                        help="Training mode: 'subject_dependent' (Session split) or 'subject_independent' (LOSO)")
    parser.add_argument('--test_subject', type=int, default=1, 
                        help="ID of the subject to leave out for testing (only for subject_independent mode)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--band', type=str, default='standard', choices=['standard', 'gamma'], 
                        help="Frequency band to use: 'standard' (1-49Hz) or 'gamma' (50-75Hz)")
    return parser.parse_args()

def main():
    args = get_args()
    num_workers = 0 
    print(f"Using device: {DEVICE} with {num_workers} workers.")
    print(f"Mode: {args.mode} | Band: {args.band}")

    # 1. Load Data
    print(f"Loading data for band: {args.band}...")
    X = np.load(os.path.join(DATA_FOLDER, f"X_raw_{args.band}.npy")) 
    y = np.load(os.path.join(DATA_FOLDER, f"y_labels_{args.band}.npy"))
    sessions = np.load(os.path.join(DATA_FOLDER, f"sessions_{args.band}.npy"))
    subjects = np.load(os.path.join(DATA_FOLDER, f"subjects_{args.band}.npy"))
    
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
        
    X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
    X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 3. Construct Graph
    print("Constructing Graph...")
    base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)
    
    # 4. Initialize Model
    model = CNNGCNModel(num_nodes=62, time_steps=400).to(DEVICE)

    # Verify output dimension
    print("Verifying model output shape...")
    dummy_input = torch.randn(2, 62, 400).to(DEVICE)
    dummy_batch = torch.arange(2, device=DEVICE).repeat_interleave(62)
    offsets = (torch.arange(2, device=DEVICE) * 62).view(-1, 1, 1)
    dummy_edge_expanded = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

    with torch.no_grad():
        dummy_out = model(dummy_input, dummy_edge_expanded, dummy_batch)
    print(f"Model Output Shape: {dummy_out.shape} (Should be [2, 3])")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)
    
    # NUCLEAR OPTION: Force the model to care about all classes equally or slightly boost class 2
    # Even if data is balanced, this helps break the "ignore class 2" habit.
    # Weights: [1.0, 1.0, 1.2] -> 20% higher penalty for missing Class 2 (Positive)
    class_weights = torch.tensor([1.0, 1.0, 1.2]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 5. Training Loop
    print("Starting Training...")
    start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            curr_batch_size = batch_X.size(0)
            batch_idx = torch.arange(curr_batch_size, device=DEVICE).repeat_interleave(62)
            
            offsets = (torch.arange(curr_batch_size, device=DEVICE) * 62).view(-1, 1, 1)
            edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
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
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate with detailed report every epoch
        val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, base_edge_index, criterion, return_preds=True)
        
        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # PRINT REPORT EVERY EPOCH to see if Class 2 is being predicted
        print(classification_report(val_labels, val_preds, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # --- SAVE & EVALUATE BEST MODEL ---
    print("\nLoading best model weights...")
    model.load_state_dict(best_model_wts)
    
    # Save to disk
    model_path = os.path.join(OUTPUT_FOLDER, f"best_model_{args.mode}_{args.band}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")

    # Detailed Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_acc, preds, true_labels = evaluate(model, test_loader, base_edge_index, criterion, return_preds=True)
    
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=['Negative', 'Neutral', 'Positive']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    # Save History
    history_path = os.path.join(OUTPUT_FOLDER, f"training_history_{args.mode}_{args.band}.npy")
    np.save(history_path, history)
    print(f"History saved to {history_path}")
    
    # Save Detailed Predictions for Debugging
    debug_data = {
        'y_true': true_labels,
        'y_pred': preds
    }
    np.save(os.path.join(OUTPUT_FOLDER, f"debug_predictions_{args.mode}_{args.band}.npy"), debug_data)
    print(f"Debug predictions saved to debug_predictions_{args.mode}_{args.band}.npy")
    
    print(f"Total Time: {time.time() - start_time:.2f}s")


def evaluate(model, loader, base_edge_index, criterion, return_preds=False):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            curr_batch_size = batch_X.size(0)
            batch_idx = torch.arange(curr_batch_size, device=DEVICE).repeat_interleave(62)
            
            # Expand edge_index for the batch
            offsets = (torch.arange(curr_batch_size, device=DEVICE) * 62).view(-1, 1, 1)
            edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            

            outputs = model(batch_X, edge_index, batch_idx)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
            
    acc = 100 * correct / total
    avg_loss = val_loss / len(loader)
    
    if return_preds:
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc

if __name__ == "__main__":
    main()