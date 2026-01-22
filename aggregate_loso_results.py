import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION (Adjusted for Attempt 54) ---
# This should be the root of your Parallel Run (where the Subject_X folders are)
BASE_DIR = "Results/GCN_DE_4s/Attempt_54_LOSO_Parallel" 
ERRORS_ROOT = "Errors/GCN_DE_4s/Attempt_54_LOSO_Parallel"
OUTPUT_FILE = os.path.join(BASE_DIR, "FINAL_LOSO_AGGREGATED_REPORT.txt")

def aggregate_attempt_54():
    all_preds = []
    all_true = []
    subject_scores = {}

    print(f"Aggregating LOSO Results from Attempt 54...")
    print(f"Searching for error artifacts in: {ERRORS_ROOT}")

    # 1. Loop through all 15 subjects
    for sub_id in range(1, 16):
        # The TrainingManager saves predictions.npy in the Errors directory structure
        sub_folder = os.path.join(ERRORS_ROOT, f"Subject_{sub_id}")
        file_path = os.path.join(sub_folder, "predictions.npy")

        if os.path.exists(file_path):
            # predictions.npy contains {'y_true': ..., 'y_pred': ...}
            data = np.load(file_path, allow_pickle=True).item()
            y_true_sub = data['y_true']
            y_pred_sub = data['y_pred']
            
            all_preds.extend(y_pred_sub)
            all_true.extend(y_true_sub)
            
            # Calculate individual subject accuracy
            acc = accuracy_score(y_true_sub, y_pred_sub) * 100
            subject_scores[sub_id] = acc
            print(f"  [FOUND] Subject {sub_id}: {acc:.2f}%")
        else:
            # Check if training_history.npy exists in Results as a backup check
            results_backup = os.path.join(BASE_DIR, f"Subject_{sub_id}", "training_history.npy")
            if os.path.exists(results_backup):
                history = np.load(results_backup, allow_pickle=True).item()
                acc = history['val_acc'][-1] # Take the last recorded val_acc
                print(f"  [BACKUP] Subject {sub_id}: ~{acc:.2f}% (Found history only)")
            else:
                print(f"  [MISSING] Subject {sub_id} files not found.")

    if not all_preds:
        print("\n!!! ERROR: No prediction artifacts found. Check your ERRORS_ROOT path !!!")
        return

    # 2. Convert to numpy arrays for global calculation
    y_true_all = np.array(all_true)
    y_pred_all = np.array(all_preds)

    # 3. Calculate Global Metrics
    global_acc = accuracy_score(y_true_all, y_pred_all) * 100
    report = classification_report(y_true_all, y_pred_all, 
                                   target_names=['Negative', 'Neutral', 'Positive'])
    cm = confusion_matrix(y_true_all, y_pred_all)

    # 4. Save Final Aggregated Report
    with open(OUTPUT_FILE, "w") as f:
        f.write("=== ATTEMPT 54: GLOBAL LOSO AGGREGATED REPORT ===\n")
        f.write(f"Source: {ERRORS_ROOT}\n")
        f.write(f"Timestamp: {os.path.basename(BASE_DIR)}\n")
        f.write(f"Global Mean Accuracy: {global_acc:.2f}%\n")
        f.write(f"Standard Deviation: {np.std(list(subject_scores.values())):.2f}%\n\n")
        f.write("Individual Subject Performance:\n")
        for sub, score in subject_scores.items():
            f.write(f"  Subject {sub}: {score:.2f}%\n")
        f.write("\nOverall Classification Report:\n")
        f.write(report)
        f.write("\nGlobal Confusion Matrix:\n")
        f.write(str(cm))

    print(f"\n" + "="*50)
    print(f"FINAL GLOBAL LOSO ACCURACY: {global_acc:.2f}%")
    print(f"Standard Deviation: {np.std(list(subject_scores.values())):.2f}%")
    print("="*50)
    print(f"Report saved to: {OUTPUT_FILE}")

    # 5. Plot Global Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=['Neg', 'Neu', 'Pos'], 
                yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f'Attempt 54: Global LOSO Confusion Matrix\nMean Accuracy: {global_acc:.2f}%')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.savefig(os.path.join(BASE_DIR, "Global_LOSO_CM_Attempt54.png"))
    # plt.show()

if __name__ == "__main__":
    aggregate_attempt_54()