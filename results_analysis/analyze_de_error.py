import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import argparse
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_de_data(data_folder, label_file):
    """
    Reusing the data loading logic to reconstruct metadata (Subjects/Sessions)
    """
    print(f"Loading metadata from {data_folder}...")
    try:
        label_mat = scipy.io.loadmat(label_file)
        trial_labels = label_mat['label'][0]
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file}")
        sys.exit(1)
    
    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = [label_map[l] for l in trial_labels]
    
    session_list = []
    subject_list = []
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    for f in files:
        parts = f.split('_')
        try:
            subj_id = int(parts[0])
        except ValueError: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)
        
    for subj_id in sorted(subject_files.keys()):
        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            for trial_i in range(1, 16):
                key = f"de_LDS{trial_i}"
                if key not in mat: continue
                data = mat[key]
                # We only need the count of samples to reconstruct the arrays
                num_samples = data.shape[1] # Shape is (62, samples, 5) or similar, usually dim 1 is time
                
                # Double check shape logic from train_de.py:
                # data = np.transpose(data, (1, 0, 2)) -> (samples, nodes, bands)
                # So original shape[1] is indeed samples.
                
                session_list.append(np.full(num_samples, session_id))
                subject_list.append(np.full(num_samples, subj_id))

    sessions = np.concatenate(session_list, axis=0)
    subjects = np.concatenate(subject_list, axis=0)
    return sessions, subjects

def main():
    parser = argparse.ArgumentParser(description="Analyze GCN-DE Errors")
    parser.add_argument('--run_id', type=int, required=True, help="Attempt ID (e.g., 3)")
    parser.add_argument('--window_size', type=str, default='4s', choices=['1s', '4s'])
    parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'])
    args = parser.parse_args()

    # Paths
    model_name = f"GCN_DE_{args.window_size}"
    run_name = f"Attempt_{args.run_id}"
    errors_dir = os.path.join("Errors", model_name, run_name)
    preds_file = os.path.join(errors_dir, "predictions.npy")
    
    if not os.path.exists(preds_file):
        print(f"Error: Predictions file not found at {preds_file}")
        print(f"Make sure the training finished and saved results.")
        return

    print(f"Loading predictions from {preds_file}...")
    data = np.load(preds_file, allow_pickle=True).item()
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    # Load Original Data to get Metadata (Subjects/Sessions)
    if args.window_size == '1s':
        data_folder = "Data/ExtractedFeatures_1s"
    else:
        data_folder = "Data/ExtractedFeatures_4s"
    label_file = os.path.join(data_folder, "label.mat")
    
    # Reconstruct metadata
    sessions, subjects = load_de_data(data_folder, label_file)
    
    # Recreate Mask
    if args.mode == 'sub_dep':
        test_mask = (sessions == 3)
    else:
        print("Warning: sub_indep analysis requires knowing the test_subject. Assuming sub_dep (Session 3).")
        test_mask = (sessions == 3)

    test_subjects = subjects[test_mask]
    
    # Validation
    if len(test_subjects) != len(y_true):
        print(f"Error: Mismatch in lengths. Data (Session 3): {len(test_subjects)}, Preds: {len(y_true)}")
        print("Did you use a different test split?")
        return

    # --- Metrics ---
    print("\n" + "="*30)
    print(f"Analysis for {model_name} | {run_name}")
    print("="*30)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    
    # --- Visualization 1: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f"Confusion Matrix ({model_name} - {run_name})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Errors per Subject ---
    errors = (y_pred != y_true)
    error_subjects = test_subjects[errors]
    
    # Calculate error rate per subject
    unique_subs = np.unique(test_subjects)
    error_rates = []
    for sub in unique_subs:
        sub_mask = (test_subjects == sub)
        total_sub = np.sum(sub_mask)
        err_sub = np.sum(errors & sub_mask)
        error_rates.append(err_sub / total_sub * 100)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(error_subjects, bins=np.arange(1, 17)-0.5, rwidth=0.8, color='salmon', edgecolor='black')
    plt.title("Total Error Count per Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Count")
    plt.xticks(range(1, 16))
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(unique_subs, error_rates, color='skyblue', edgecolor='black')
    plt.title("Error Rate (%) per Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Error Rate (%)")
    plt.xticks(range(1, 16))
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()