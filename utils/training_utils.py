import torch
import numpy as np
import os
import copy
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Single Epoch Runner (The Logic Fix) ---
def train_epoch(model, loader, optimizer, criterion, device, base_edge_index, in_features):
    """
    Runs exactly ONE epoch of training.
    Handles dynamic input feature reshaping to avoid dimension mismatches.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        curr_batch_size = batch_X.size(0)
        batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(62)
        
        offsets = (torch.arange(curr_batch_size, device=device) * 62).view(-1, 1, 1)
        edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        
        # --- FIX: Dynamic Flattening ---
        # Uses in_features passed from the main script (e.g., 10) instead of hardcoded 5
        batch_X_flat = batch_X.view(-1, in_features)
        
        outputs = model(batch_X_flat, edge_index, batch_idx)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    return total_loss / len(loader), 100 * correct / total

# --- 2. Training Manager (State & Saving) ---
class TrainingManager:
    def __init__(self, results_dir, params_dir, errors_dir, model, args=None):
        self.results_dir = results_dir
        self.params_dir = params_dir
        self.errors_dir = errors_dir
        self.model = model
        self.args = args
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        self.val_preds_history = [] 
        self.val_embeddings_history = {} 
        
        self.best_val_acc = 0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.errors_dir, exist_ok=True)
        
        print(f"TrainingManager initialized.")
        print(f"  Results -> {self.results_dir}")
        print(f"  Params  -> {self.params_dir}")

    def update_history(self, t_loss, t_acc, v_loss, v_acc):
        self.history['train_loss'].append(t_loss)
        self.history['train_acc'].append(t_acc)
        self.history['val_loss'].append(v_loss)
        self.history['val_acc'].append(v_acc)

    def update_detailed_history(self, epoch, preds, embeddings=None):
        self.val_preds_history.append(preds)
        if embeddings is not None:
            self.val_embeddings_history[epoch] = embeddings

    def save_checkpoint(self, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            torch.save(self.model.state_dict(), os.path.join(self.params_dir, "best_model_checkpoint.pth"))
            return True 
        return False 

    def handle_interrupt(self):
        print("\n\n>>> Training interrupted by user! Saving current state...")
        # Save the current model state as 'interrupted'
        torch.save(self.model.state_dict(), os.path.join(self.params_dir, "model_interrupted.pth"))
        self._save_artifacts()
        print(">>> Emergency save complete.")

    def save_final_results(self, preds, true_labels):
        print(f"\nSaving final results...")
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), os.path.join(self.params_dir, "best_model_final.pth"))
        
        self._save_artifacts(preds, true_labels)
        
        np.save(os.path.join(self.results_dir, "evolution_history.npy"), {
            'preds_history': self.val_preds_history,
            'embeddings_history': self.val_embeddings_history,
            'true_labels': true_labels 
        })
        print("All files saved successfully.")

    def _save_artifacts(self, preds=None, true_labels=None):
        np.save(os.path.join(self.results_dir, "training_history.npy"), self.history)
        self._plot_history()

        if preds is not None and true_labels is not None:
            debug_data = {'y_true': true_labels, 'y_pred': preds}
            np.save(os.path.join(self.errors_dir, "predictions.npy"), debug_data)
            
            # --- FIX: Dynamic Target Names ---
            unique_labels = np.unique(true_labels)
            num_classes = len(unique_labels)
            
            # Determine names based on class count
            if num_classes == 2:
                # Binary Diagnostic Mode (Neg vs Neu)
                target_names = ['Negative', 'Neutral']
            elif num_classes == 3:
                # Standard Mode (Neg, Neu, Pos)
                target_names = ['Negative', 'Neutral', 'Positive']
            else:
                # Fallback for unexpected cases
                target_names = [str(i) for i in unique_labels]

            try:
                report = classification_report(true_labels, preds, target_names=target_names)
            except ValueError:
                # Fallback if unique labels don't match expected count (e.g. test set missing a class)
                report = classification_report(true_labels, preds)

            with open(os.path.join(self.errors_dir, "classification_report.txt"), "w") as f:
                f.write(f"Model: {type(self.model).__name__}\n")
                if self.args: f.write(f"Args: {self.args}\n")
                f.write(report)
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(confusion_matrix(true_labels, preds)))

    def _plot_history(self):
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_acc'], label='Train Acc')
            plt.plot(self.history['val_acc'], label='Val Acc')
            plt.title('Accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Val Loss')
            plt.title('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.results_dir, "training_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Could not generate plot: {e}")

# --- 3. The Master Loop (Restored & Updated) ---
def train_model_with_interrupt(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, device, results_dir, params_dir, errors_dir, base_edge_index, evaluate_fn, hyperparams=None, in_features=5):
    """
    Orchestrates the full training process.
    - Calls train_epoch() for the logic.
    - Handles Saving, History, and Interrupts via TrainingManager.
    """
    manager = TrainingManager(results_dir, params_dir, errors_dir, model, args=hyperparams)
    
    print("Starting Training... (Press Ctrl+C to stop and save)")
    start_time = time.time()
    patience_counter = 0

    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 1. Run One Epoch (using the fixed function)
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, base_edge_index, in_features
            )
            
            # 2. Evaluate
            capture_embeddings = (epoch % 10 == 0) or (epoch == epochs)
            if capture_embeddings:
                val_loss, val_acc, preds, true_labels, embeddings = evaluate_fn(
                    model, test_loader, base_edge_index, criterion, device, return_preds=True, return_embeddings=True
                )
                manager.update_detailed_history(epoch, preds, embeddings)
            else:
                val_loss, val_acc, preds, true_labels = evaluate_fn(
                    model, test_loader, base_edge_index, criterion, device, return_preds=True, return_embeddings=False
                )
                manager.update_detailed_history(epoch, preds, None)
            
            # 3. Update Manager & Save Checkpoint
            manager.update_history(train_loss, train_acc, val_loss, val_acc)
            improved = manager.save_checkpoint(val_acc)
            
            # 4. Logging
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} ({time.time() - epoch_start:.1f}s) | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # 5. Scheduler & Early Stopping
            scheduler.step(val_loss)
            
            # if improved:
            #     patience_counter = 0
            #     print(f"  >>> New Best Model Saved! ({val_acc:.2f}%)")
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print("Early stopping triggered.")
            #         break
                    
    except KeyboardInterrupt:
        manager.handle_interrupt()

    # --- Final Wrap-up ---
    print("\nRunning Final Evaluation on Best Model...")
    model.load_state_dict(manager.best_model_wts)
    test_loss, test_acc, preds, true_labels = evaluate_fn(
        model, test_loader, base_edge_index, criterion, device, return_preds=True
    )
    
    print(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    manager.save_final_results(preds, true_labels)
    print(f"Total Time: {time.time() - start_time:.2f}s")