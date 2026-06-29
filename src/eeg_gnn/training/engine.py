import torch
import numpy as np
import os
import copy
import time
import inspect
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def is_dense_model(model):
    """Helper to detect if the model expects 3D Dense input or 2D Sparse input."""
    name = model.__class__.__name__.lower()
    return "dgcnn" in name or hasattr(model, 'adaptive_input')

def supports_embeddings(model):
    """Checks if the model's forward method accepts 'return_embedding'."""
    sig = inspect.signature(model.forward)
    return 'return_embedding' in sig.parameters

def supports_subject_ids(model):
    """Checks if the model's forward method accepts subject IDs (Phase 2)."""
    sig = inspect.signature(model.forward)
    return 'subject_ids' in sig.parameters or 'subject_ids_full' in sig.parameters

# --- 1. Single Epoch Runner ---
def train_epoch(model, loader, optimizer, criterion, device, base_edge_index, in_features):
    model.train()
    
    dense_mode = is_dense_model(model)
    sig = inspect.signature(model.forward)
    has_subject_ids = 'subject_ids' in sig.parameters
    has_subject_ids_full = 'subject_ids_full' in sig.parameters
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        if len(batch) == 3:
            batch_X, batch_y, batch_sub = batch
            batch_sub = batch_sub.to(device)
        else:
            batch_X, batch_y = batch
            batch_sub = None
            
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        if batch_X.is_floating_point():
             batch_X = batch_X + torch.randn_like(batch_X) * 0.005
        
        optimizer.zero_grad()
        
        # --- LOGIC BRANCHING ---
        if dense_mode:
            if batch_sub is not None and 'subject_ids_graph' in sig.parameters:
                outputs = model(batch_X, subject_ids_graph=batch_sub)
            else:
                outputs = model(batch_X)
        else:
            curr_batch_size, num_nodes, _ = batch_X.shape
            batch_X_flat = batch_X.view(-1, in_features)
            
            # Graph Indices
            batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(num_nodes)
            offsets = (torch.arange(curr_batch_size, device=device) * num_nodes).view(-1, 1, 1)
            edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
            if batch_sub is not None:
                if has_subject_ids:
                    outputs = model(batch_X_flat, edge_index, batch_idx, subject_ids=batch_sub)
                elif has_subject_ids_full:
                    batch_sub_full = batch_sub.repeat_interleave(num_nodes)
                    outputs = model(batch_X_flat, edge_index, batch_idx, subject_ids_full=batch_sub_full)
                else:
                    outputs = model(batch_X_flat, edge_index, batch_idx)
            else:
                outputs = model(batch_X_flat, edge_index, batch_idx)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    return train_loss / len(loader), 100 * correct / total


# --- 2. Universal Evaluate ---
def evaluate(model, loader, base_edge_index, criterion, device, in_features, 
             return_preds=False, return_embeddings=False):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    all_embeddings = []
    
    dense_mode = is_dense_model(model)
    
    # Pre-calculate signature ONCE to avoid repeated overhead per batch
    sig = inspect.signature(model.forward)
    has_subject_ids = 'subject_ids' in sig.parameters
    has_subject_ids_full = 'subject_ids_full' in sig.parameters
    supports_emb = 'return_embedding' in sig.parameters

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                batch_X, batch_y, batch_sub = batch_data
                batch_sub = batch_sub.to(device)
            else:
                batch_X, batch_y = batch_data
                batch_sub = None
                
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
    
            # Prepare Inputs for Sparse/Dense
            if not dense_mode:
                curr_batch_size, num_nodes, _ = batch_X.shape
                batch_X_flat = batch_X.view(-1, in_features)
                batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(num_nodes)
                offsets = (torch.arange(curr_batch_size, device=device) * num_nodes).view(-1, 1, 1)
                edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            
            kwargs = {}
            if return_embeddings and supports_emb:
                kwargs['return_embedding'] = True
            
            if dense_mode:
                if batch_sub is not None and 'subject_ids_graph' in sig.parameters:
                    out = model(batch_X, subject_ids_graph=batch_sub, **kwargs)
                else:
                    out = model(batch_X, **kwargs)
            else:
                if batch_sub is not None:
                    if has_subject_ids:
                        out = model(batch_X_flat, edge_index, batch_idx, subject_ids=batch_sub, **kwargs)
                    elif has_subject_ids_full:
                        batch_sub_full = batch_sub.repeat_interleave(num_nodes)
                        out = model(batch_X_flat, edge_index, batch_idx, subject_ids_full=batch_sub_full, **kwargs)
                    else:
                        out = model(batch_X_flat, edge_index, batch_idx, **kwargs)
                else:
                    out = model(batch_X_flat, edge_index, batch_idx, **kwargs)

            if isinstance(out, tuple):
                logits, emb = out
            else:
                logits = out
                emb = None

            loss = criterion(logits, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                if return_embeddings and emb is not None:
                    all_embeddings.extend(emb.cpu().numpy())
                
    if total == 0:
        acc = 0.0
        avg_loss = 0.0
    else:
        acc = 100 * correct / total
        avg_loss = val_loss / len(loader) if len(loader) > 0 else 0
    
    if return_embeddings:
        emb_data = np.array(all_embeddings) if all_embeddings else np.array([])
        return avg_loss, acc, np.array(all_preds), np.array(all_labels), emb_data
    
    if return_preds:
        return avg_loss, acc, np.array(all_preds), np.array(all_labels)
        
    return avg_loss, acc


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
        self.best_epoch = 0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.errors_dir, exist_ok=True)
        
        print(f"TrainingManager initialized.")
        print(f"  Results -> {self.results_dir}")
        print(f"  Params  -> {self.params_dir}")

        if self.args:
            self.save_training_setup()

    def save_training_setup(self):
        filepath = os.path.join(self.params_dir, "training_setup.txt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("=== TRAINING SETUP & HYPERPARAMETERS ===\n")
            f.write(f"Model Class: {self.model.__class__.__name__}\n\n")
            
            if isinstance(self.args, dict):
                for k, v in self.args.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(self.args))

    def update_history(self, t_loss, t_acc, v_loss, v_acc):
        self.history['train_loss'].append(t_loss)
        self.history['train_acc'].append(t_acc)
        self.history['val_loss'].append(v_loss)
        self.history['val_acc'].append(v_acc)

    def update_detailed_history(self, epoch, preds, embeddings=None):
        self.val_preds_history.append(preds)
        if embeddings is not None and len(embeddings) > 0:
            self.val_embeddings_history[epoch] = embeddings
    
    def save_checkpoint(self, val_acc, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
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
            
            unique_labels = np.unique(true_labels)
            num_classes = len(unique_labels)
            
            if num_classes == 2:
                target_names = ['Negative', 'Neutral']
            elif num_classes == 3:
                target_names = ['Negative', 'Neutral', 'Positive']
            else:
                target_names = [str(i) for i in unique_labels]

            try:
                report = classification_report(true_labels, preds, target_names=target_names)
            except ValueError:
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
def train_model_with_interrupt(model, train_loader, test_loader, optimizer, 
                               criterion, scheduler, epochs, device, results_dir, 
                               params_dir, errors_dir, subject_tag, base_edge_index, 
                               evaluate_fn, hyperparams=None, in_features=5, patience=20):
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
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, base_edge_index, in_features)
            
            capture = (epoch % 10 == 0) or (epoch == epochs)
            # Safe call to evaluate_fn
            eval_res = evaluate_fn(model, test_loader, base_edge_index, criterion, device, in_features, return_preds=True, return_embeddings=capture)
            
            if capture:
                val_loss, val_acc, preds, labels, embs = eval_res
                manager.update_detailed_history(epoch, preds, embs)
            else:
                val_loss, val_acc, preds, labels = eval_res
                manager.update_detailed_history(epoch, preds, None)
            
            manager.update_history(train_loss, train_acc, val_loss, val_acc)
            improved = manager.save_checkpoint(val_acc, epoch)
            
            
            current_lr = optimizer.param_groups[0]['lr']
            epoch_str = f"Epoch {epoch}/{epochs}"
            time_str = f"({time.time() - epoch_start:.1f}s)"
            lr_str = f"LR: {current_lr:.6f}"
            perf_str = f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            
            log_line = f"{subject_tag} {epoch_str} {time_str} | {lr_str} | {perf_str}"
            
            if improved:
                # Add a visual indicator for new best models
                log_line += f"  *** BEST (Ep {epoch})"
                # patience_counter = 0 
            else:
                pass
                # patience_counter += 1
            
            print(log_line)
            
            # --- Fix: Handle different scheduler types ---
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # if patience_counter >= patience:
            #     print("Early stopping triggered.")
            #     break
                    
    except KeyboardInterrupt:
        manager.handle_interrupt()


    print("\nRunning Final Evaluation on Best Model...")
    
    try:
        model.load_state_dict(manager.best_model_wts)
        test_loss, test_acc, preds, true_labels = evaluate_fn(
            model, test_loader, base_edge_index, criterion, device, in_features, return_preds=True, return_embeddings=False
        )
        
        print(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        manager.save_final_results(preds, true_labels)
    except Exception as e:
        print(f"\n[WARNING] Could not run final evaluation (Workers likely dead): {e}")
        print("Saving best model weights manually...")
        torch.save(manager.best_model_wts, os.path.join(manager.params_dir, "best_model_rescue.pth"))
    
    print(f"Total Time: {time.time() - start_time:.2f}s")
