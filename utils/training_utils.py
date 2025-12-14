import torch
import numpy as np
import os
import copy
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class TrainingManager:
    def __init__(self, results_dir, params_dir, errors_dir, model, args=None):
        self.results_dir = results_dir
        self.params_dir = params_dir
        self.errors_dir = errors_dir
        self.model = model
        self.args = args
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # --- NEW: Evolution History ---
        self.val_preds_history = [] # List of arrays (one per epoch)
        self.val_embeddings_history = {} # Dict: epoch -> array (every N epochs)
        
        self.best_val_acc = 0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.errors_dir, exist_ok=True)
        
        print(f"TrainingManager initialized.")
        print(f"  Results -> {self.results_dir}")
        print(f"  Params  -> {self.params_dir}")
        print(f"  Errors  -> {self.errors_dir}")


    def update_history(self, t_loss, t_acc, v_loss, v_acc):
        self.history['train_loss'].append(t_loss)
        self.history['train_acc'].append(t_acc)
        self.history['val_loss'].append(v_loss)
        self.history['val_acc'].append(v_acc)

    def update_detailed_history(self, epoch, preds, embeddings=None):
        """Stores per-epoch predictions and periodic embeddings"""
        self.val_preds_history.append(preds)
        if embeddings is not None:
            self.val_embeddings_history[epoch] = embeddings

    def save_checkpoint(self, val_acc):
        """Saves the model if validation accuracy improves."""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            torch.save(self.model.state_dict(), os.path.join(self.params_dir, "best_model_checkpoint.pth"))
            return True # Improved
        return False # Not improved

    def handle_interrupt(self):
        print("\n\n>>> Training interrupted by user! Saving current state...")
        self._save_artifacts()
        print(">>> Emergency save complete.")

    def save_final_results(self, preds, true_labels):
        print(f"\nSaving final results...")
        
        # Load best weights before saving final model state
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), os.path.join(self.params_dir, "best_model_final.pth"))
        
        # Save predictions and report
        self._save_artifacts(preds, true_labels)
        
        # --- NEW: Save Evolution Data ---
        print("Saving evolution history (this might be large)...")
        np.save(os.path.join(self.results_dir, "evolution_history.npy"), {
            'preds_history': self.val_preds_history,
            'embeddings_history': self.val_embeddings_history,
            'true_labels': true_labels # Saved once (assuming shuffle=False for val)
        })
        
        print("All files saved successfully.")

    def _save_artifacts(self, preds=None, true_labels=None):
        # Save History (Results Dir)
        np.save(os.path.join(self.results_dir, "training_history.npy"), self.history)
        self._plot_history()

        # Save Predictions & Report (Errors Dir)
        if preds is not None and true_labels is not None:
            debug_data = {'y_true': true_labels, 'y_pred': preds}
            np.save(os.path.join(self.errors_dir, "predictions.npy"), debug_data)
            
            report = classification_report(true_labels, preds, target_names=['Negative', 'Neutral', 'Positive'])
            with open(os.path.join(self.errors_dir, "classification_report.txt"), "w") as f:
                f.write(f"Model: {type(self.model).__name__}\n")
                if self.args: f.write(f"Args: {self.args}\n")
                f.write(report)
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(confusion_matrix(true_labels, preds)))

    def _plot_history(self):
        try:
            plt.figure(figsize=(12, 5))
            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_acc'], label='Train Acc')
            plt.plot(self.history['val_acc'], label='Val Acc')
            plt.title('Accuracy')
            plt.legend()
            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Val Loss')
            plt.title('Loss')
            plt.legend()
            
            plt.savefig(os.path.join(self.results_dir, "training_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Could not generate plot: {e}")

def train_model_with_interrupt(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, device, patience, results_dir, params_dir, errors_dir, base_edge_index, evaluate_fn, hyperparams=None):
    """
    Wrapper function to handle the training loop with KeyboardInterrupt protection.
    """
    manager = TrainingManager(results_dir, params_dir, errors_dir, model, args=hyperparams)
    
    print("Starting Training... (Press Ctrl+C to stop and save)")
    start_time = time.time()
    patience_counter = 0

    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            total_loss, correct, total = 0, 0, 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                curr_batch_size = batch_X.size(0)
                batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(62)
                
                offsets = (torch.arange(curr_batch_size, device=device) * 62).view(-1, 1, 1)
                edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
                
                # Flatten for DE model (assuming input is [Batch, 62, 5])
                batch_X_flat = batch_X.view(-1, 5)
                
                optimizer.zero_grad()
                outputs = model(batch_X_flat, edge_index, batch_idx)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
            train_acc = 100 * correct / total
            avg_train_loss = total_loss / len(train_loader)
            
            # --- NEW: Detailed Evaluation ---
            # Capture embeddings every 10 epochs or last epoch
            capture_embeddings = (epoch % 10 == 0) or (epoch == epochs - 1)
            
            if capture_embeddings:
                val_loss, val_acc, preds, true_labels, embeddings = evaluate_fn(model, test_loader, base_edge_index, criterion, device, return_preds=True, return_embeddings=True)
                manager.update_detailed_history(epoch, preds, embeddings)
            else:
                val_loss, val_acc, preds, true_labels = evaluate_fn(model, test_loader, base_edge_index, criterion, device, return_preds=True, return_embeddings=False)
                manager.update_detailed_history(epoch, preds, None)
            
            # Update Manager
            manager.update_history(avg_train_loss, train_acc, val_loss, val_acc)
            improved = manager.save_checkpoint(val_acc)
            
            # --- PRINT LR HERE ---
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time() - epoch_start:.1f}s) | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    except KeyboardInterrupt:
        manager.handle_interrupt()

    # --- Final Evaluation ---
    print("\nRunning Final Evaluation...")
    model.load_state_dict(manager.best_model_wts)
    # Just standard eval for final report
    test_loss, test_acc, preds, true_labels = evaluate_fn(model, test_loader, base_edge_index, criterion, device, return_preds=True)
    
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    manager.save_final_results(preds, true_labels)
    print(f"Total Time: {time.time() - start_time:.2f}s")