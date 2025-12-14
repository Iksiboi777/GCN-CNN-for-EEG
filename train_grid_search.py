import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import itertools
import pandas as pd
import sys
import argparse

# Import existing utilities
from train_de import load_de_data, evaluate, LOCS_FILE
from Models.var_B import GCN_DE_Model
from Models.graph_construction import get_knn_adjacency_matrix
from utils.training_utils import train_model_with_interrupt

# --- Grid Search Configuration ---
PARAM_GRID = {
    'hidden_dim': [32, 64],
    'num_layers': [2, 3],
    'learning_rate': [0.001, 0.0005],
    'dropout': [0.5], 
    'weight_decay': [1e-3] 
}

PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Grid Search for GCN-DE")
    parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'],
                        help="Training mode: 'sub_dep' (Session split) or 'sub_indep' (LOSO)")
    parser.add_argument('--window_size', type=str, default='4s', choices=['1s', '4s'],
                        help="Feature window size: '1s' or '4s'")
    parser.add_argument('--test_subject', type=int, default=1, 
                        help="ID of the subject to leave out for testing (only for sub_indep mode)")
    parser.add_argument('--epochs', type=int, default=60, help="Number of training epochs per grid combination")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    return parser.parse_args()

def run_grid_search():
    args = get_args()
    print(f"Starting Grid Search on {DEVICE}...")
    print(f"Configuration: Mode={args.mode}, Window={args.window_size}, Epochs={args.epochs}")
    
    # 1. Load Data
    if args.window_size == '1s':
        data_folder = "Data/ExtractedFeatures_1s"
    else:
        data_folder = "Data/ExtractedFeatures_4s"
    label_file = os.path.join(data_folder, "label.mat")
    
    X, y, sessions, subjects = load_de_data(data_folder, label_file)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Apply Split Logic based on args.mode
    if args.mode == 'sub_dep':
        train_mask = (sessions == 1) | (sessions == 2)
        test_mask = (sessions == 3)
    elif args.mode == 'sub_indep':
        train_mask = (subjects != args.test_subject)
        test_mask = (subjects == args.test_subject)
    
    X_train, y_train = X_tensor[train_mask], y_tensor[train_mask]
    X_test, y_test = X_tensor[test_mask], y_tensor[test_mask]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    
    base_edge_index = get_knn_adjacency_matrix(LOCS_FILE, k=5).to(DEVICE)
    
    # 2. Generate Combinations
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    print(f"Total Combinations to test: {len(combinations)}")
    
    # Base folder name for this grid search session
    grid_session_name = f"GridSearch_{args.window_size}_{args.mode}"
    
    for i, params in enumerate(combinations):
        print(f"\n\n=== Running Combination {i+1}/{len(combinations)} ===")
        print(f"Params: {params}")
        
        # Specific run name
        run_name = f"L{params['num_layers']}_H{params['hidden_dim']}_LR{params['learning_rate']}"
        
        # Setup Directories: Results/GridSearch_4s_sub_dep/L2_H32_LR0.001
        results_dir = os.path.join("Results", grid_session_name, run_name)
        params_dir = os.path.join("Params", grid_session_name, run_name)
        errors_dir = os.path.join("Errors", grid_session_name, run_name)
        
        # Initialize Model
        model = GCN_DE_Model(
            num_nodes=62, 
            in_features=5, 
            hidden_dim=params['hidden_dim'], 
            num_classes=3, 
            dropout_rate=params['dropout'],
            num_layers=params['num_layers']
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        best_acc = train_model_with_interrupt(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=args.epochs,
            device=DEVICE,
            patience=PATIENCE,
            results_dir=results_dir,
            params_dir=params_dir,
            errors_dir=errors_dir,
            base_edge_index=base_edge_index,
            evaluate_fn=evaluate
        )
        
        # Log Result
        result_entry = params.copy()
        result_entry['best_val_acc'] = best_acc
        results.append(result_entry)
        
        # Save intermediate CSV to the root of the grid session folder
        os.makedirs(os.path.join("Results", grid_session_name), exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join("Results", grid_session_name, "grid_search_results.csv"), index=False)
        
    print("\n\n=== Grid Search Complete ===")
    df = pd.DataFrame(results)
    df = df.sort_values(by='best_val_acc', ascending=False)
    print(df)
    
    # Save final results
    df.to_csv(os.path.join("Results", grid_session_name, "grid_search_results_final.csv"), index=False)

if __name__ == "__main__":
    run_grid_search()