import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
# Adjust these to match the specific run you want to analyze
MODEL_NAME = "ADAPTIVE_DGCNN_DE_4s"
ATTEMPT_ID = "Attempt_63_LOSO_Parallel"

# Directories based on your TrainingManager logic
RESULTS_ROOT = f"Results/{MODEL_NAME}/{ATTEMPT_ID}" 
ERRORS_ROOT = f"Errors/{MODEL_NAME}/{ATTEMPT_ID}"
OUTPUT_TXT = os.path.join(RESULTS_ROOT, "FINAL_EXTENDED_REPORT.txt")
CONFIG_FILE = "run_config.json"

def load_hyperparams():
    """Tries to read global config or returns default info."""
    info = {
        "Model": MODEL_NAME,
        "Attempt": ATTEMPT_ID,
        "Structure": "Batch-Parallel LOSO"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                info.update(config)
        except:
            pass
    return info

def plot_training_batches(subject_histories):
    """
    Concatenates training plots into 3 batches (Subjects 1-5, 6-10, 11-15).
    Saves them to the Results root.
    """
    batches = [
        (range(1, 6), "Batch_1_Subjects_1-5"),
        (range(6, 11), "Batch_2_Subjects_6-10"),
        (range(11, 16), "Batch_3_Subjects_11-15")
    ]

    print("\n[PLOTTING] Generating Batch Training Curves...")

    for subjects, batch_name in batches:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        has_data = False
        colors = plt.cm.tab10(np.linspace(0, 1, 5)) # Distinct colors for 5 subjects

        for i, sub_id in enumerate(subjects):
            if sub_id in subject_histories:
                hist = subject_histories[sub_id]
                epochs = range(1, len(hist['train_acc']) + 1)
                color = colors[i]
                
                # Plot Accuracy
                axes[0].plot(epochs, hist['val_acc'], label=f"S{sub_id} Val", 
                             color=color, linestyle='-', linewidth=2)
                axes[0].plot(epochs, hist['train_acc'], label=f"S{sub_id} Train", 
                             color=color, linestyle='--', alpha=0.5, linewidth=1)
                
                # Plot Loss
                axes[1].plot(epochs, hist['val_loss'], label=f"S{sub_id} Val", 
                             color=color, linestyle='-', linewidth=2)
                axes[1].plot(epochs, hist['train_loss'], label=f"S{sub_id} Train", 
                             color=color, linestyle='--', alpha=0.5, linewidth=1)
                
                has_data = True
        
        if not has_data:
            plt.close()
            continue
            
        # Formatting Subplot 1 (Accuracy)
        axes[0].set_title(f"{batch_name} - Accuracy Evolution")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend(loc='lower right', ncol=2, fontsize='small')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        # Formatting Subplot 2 (Loss)
        axes[1].set_title(f"{batch_name} - Loss Evolution")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Cross Entropy Loss")
        axes[1].legend(loc='upper right', ncol=2, fontsize='small')
        axes[1].grid(True, alpha=0.3)

        save_path = os.path.join(RESULTS_ROOT, f"Combined_Plot_{batch_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> Saved: {save_path}")

def aggregate_results():
    print(f"--- Analysis Started: {MODEL_NAME}/{ATTEMPT_ID} ---")
    
    all_preds_global = []
    all_true_global = []
    subject_scores = {}
    subject_histories = {}
    subject_reports = {} 

    # 1. Iterate Subjects
    for sub_id in range(1, 16):
        # Define paths based on TrainingManager logic
        res_sub_dir = os.path.join(RESULTS_ROOT, f"Subject_{sub_id}")
        err_sub_dir = os.path.join(ERRORS_ROOT, f"Subject_{sub_id}")
        
        # A. Load History (from Results)
        hist_path = os.path.join(res_sub_dir, "training_history.npy")
        if os.path.exists(hist_path):
            subject_histories[sub_id] = np.load(hist_path, allow_pickle=True).item()
        
        # B. Load Predictions (from Errors)
        # TrainingManager saves this as 'predictions.npy' containing {'y_true', 'y_pred'}
        # OR 'final_test_preds_subX.npy' if the manual test ran. We check both.
        
        pred_data = None
        # Priority 1: Check Errors folder (standard TrainingManager artifact)
        path_err = os.path.join(err_sub_dir, "predictions.npy")
        # Priority 2: Check Results folder (Manual final test save)
        path_res = os.path.join(res_sub_dir, f"final_test_preds_sub{sub_id}.npy")

        if os.path.exists(path_err):
            data = np.load(path_err, allow_pickle=True).item()
            # Handle different naming conventions in saving
            y_pred = data.get('y_pred', data.get('preds'))
            y_true = data.get('y_true', data.get('true'))
            pred_data = (y_true, y_pred)
            
        elif os.path.exists(path_res):
            data = np.load(path_res, allow_pickle=True).item()
            y_pred = data.get('preds')
            y_true = data.get('true')
            pred_data = (y_true, y_pred)

        if pred_data:
            y_true, y_pred = pred_data
            
            # Ensure Types
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            all_preds_global.extend(y_pred)
            all_true_global.extend(y_true)
            
            # Calculate Metrics
            acc = accuracy_score(y_true, y_pred) * 100
            subject_scores[sub_id] = acc
            
            # Generate Individual Report
            # Handle class names dynamically (e.g. if a subject only has 2 classes present)
            unique_labels = np.unique(y_true)
            target_names = ['Negative', 'Neutral', 'Positive'] # Default
            if len(unique_labels) < 3:
                # Fallback if specific classes are missing in test set
                target_names = [str(l) for l in unique_labels] 

            cls_rep = classification_report(y_true, y_pred, labels=unique_labels, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            subject_reports[sub_id] = {
                "acc": acc,
                "report": cls_rep,
                "cm": cm
            }
            print(f"  Subject {sub_id}: {acc:.2f}% (Loaded)")
        else:
            print(f"  Subject {sub_id}: [MISSING PREDICTIONS]")

    if not all_preds_global:
        print("No valid prediction files found. Exiting.")
        return

    # 2. Generate Batch Plots
    plot_training_batches(subject_histories)

    # 3. Calculate Global Metrics
    y_true_all = np.array(all_true_global)
    y_pred_all = np.array(all_preds_global)
    
    global_acc = accuracy_score(y_true_all, y_pred_all) * 100
    global_report = classification_report(y_true_all, y_pred_all, target_names=['Negative', 'Neutral', 'Positive'])
    global_cm = confusion_matrix(y_true_all, y_pred_all)
    params = load_hyperparams()

    # 4. Write Detailed Text Report
    print(f"\n[REPORT] Writing to: {OUTPUT_TXT}")
    with open(OUTPUT_TXT, "w") as f:
        # Header
        f.write("="*60 + "\n")
        f.write(f"      COMPREHENSIVE LOSO ANALYSIS: {ATTEMPT_ID}\n")
        f.write("="*60 + "\n\n")
        
        # Section 1: Configuration
        f.write("--- 1. CONFIGURATION & HYPERPARAMETERS ---\n")
        for k, v in params.items():
            f.write(f"{k:<25}: {v}\n")
        f.write("\n")

        # Section 2: Global Stats
        f.write("--- 2. GLOBAL PERFORMANCE SUMMARY ---\n")
        f.write(f"Global Mean Accuracy      : {global_acc:.2f}%\n")
        scores_list = list(subject_scores.values())
        f.write(f"Standard Deviation        : {np.std(scores_list):.2f}%\n")
        f.write(f"Best Subject              : ID {max(subject_scores, key=subject_scores.get)} ({max(scores_list):.2f}%)\n")
        f.write(f"Worst Subject             : ID {min(subject_scores, key=subject_scores.get)} ({min(scores_list):.2f}%)\n\n")
        
        f.write("Global Classification Report:\n")
        f.write(global_report)
        f.write("\nGlobal Confusion Matrix:\n")
        f.write(np.array2string(global_cm, separator=', '))
        f.write("\n\n")

        # Section 3: Per-Subject Details
        f.write("--- 3. INDIVIDUAL SUBJECT MATRICES & REPORTS ---\n")
        for sub_id in sorted(subject_reports.keys()):
            data = subject_reports[sub_id]
            f.write(f"\n{'='*20} SUBJECT {sub_id} (Acc: {data['acc']:.2f}%) {'='*20}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(data['cm'], separator=', '))
            f.write("\n\nClassification Report:\n")
            f.write(data['report'])
            f.write("\n")

    # 5. Global Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(global_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f"Global Confusion Matrix ({ATTEMPT_ID})\nMean Acc: {global_acc:.2f}%")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_cm_path = os.path.join(RESULTS_ROOT, "Global_Confusion_Matrix.png")
    plt.savefig(save_cm_path)
    plt.close()
    print(f"[DONE] Global CM saved: {save_cm_path}")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_ROOT):
        print(f"Error: Results root not found: {RESULTS_ROOT}")
    else:
        aggregate_results()