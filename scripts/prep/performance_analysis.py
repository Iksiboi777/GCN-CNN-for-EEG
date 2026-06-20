import os
import numpy as np
import pandas as pd
import scipy.io
import re

# Configuration
DATA_FOLDER = "Data/ExtractedFeatures_1s"
ERRORS_FOLDER = "Errors"
OUTPUT_FILE = "Subject_Performance_Report_1s.csv"

# Classification Thresholds
THRESHOLDS = {
    "Hard": (0.0, 0.60),
    "Medium": (0.60, 0.80),
    "Easy": (0.80, 1.01)
}

def get_classification(accuracy):
    for label, (low, high) in THRESHOLDS.items():
        if low <= accuracy < high:
            return label
    return "Unknown"

def get_subject_sample_counts():
    """
    Scans the data folder to count how many samples belong to Session 3 for each subject.
    Assumes 3 files per subject, sorted by date/name. The 3rd is Session 3 (Test Set).
    """
    counts = {}
    print("Mapping dataset structure from Data folder...")
    
    for subject_id in range(1, 16): # Subjects 1-15
        # Find files for this subject
        prefix = f"{subject_id}_"
        files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(prefix) and f.endswith('.mat')]
        files.sort() # Ensure chronological order (Session 1, 2, 3)
        
        if len(files) < 3:
            print(f"Warning: Subject {subject_id} has fewer than 3 sessions. Skipping.")
            counts[subject_id] = 0
            continue
            
        # Session 3 is the 3rd file (index 2)
        session_3_file = files[2]
        file_path = os.path.join(DATA_FOLDER, session_3_file)
        
        try:
            mat = scipy.io.loadmat(file_path)
            # Count samples across all 15 trials
            total_samples = 0
            for i in range(1, 16): 
                key = f"de_LDS{i}"
                if key in mat:
                    # Shape is usually (Channel, Samples, Bands) -> dim 1 is samples
                    total_samples += mat[key].shape[1]
            
            counts[subject_id] = total_samples
            
        except Exception as e:
            print(f"Error reading {session_3_file}: {e}")
            counts[subject_id] = 0
            
    return counts

def analyze_predictions_file(npy_path, subject_counts):
    """
    Loads a single predictions.npy and slices it by subject.
    """
    try:
        data = np.load(npy_path, allow_pickle=True)
        
        # Extract y_true, y_pred
        y_true, y_pred = None, None
        
        # Handle Dictionary format
        if isinstance(data, dict) or (data.shape == () and isinstance(data.item(), dict)):
            d = data.item() if data.shape == () else data
            y_true = d.get('y_true') or d.get('labels')
            y_pred = d.get('y_pred') or d.get('predictions')
        # Handle Array format [N, 2]
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
            y_true = data[:, 0]
            y_pred = data[:, 1]
            
        if y_true is None or y_pred is None:
            return None # Cannot analyze without labels
            
        # Convert to numpy array immediately to handle lists
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Validation
        total_preds = len(y_true)
        expected_total = sum(subject_counts.values())
        
        if total_preds != expected_total:
            # Size mismatch means this prediction file doesn't match the 1s dataset structure
            return None
            
        # Slice and Dice
        results = []
        current_idx = 0
        
        for subject_id in range(1, 16):
            n_samples = subject_counts[subject_id]
            if n_samples == 0: continue
            
            # Slice
            sub_true = y_true[current_idx : current_idx + n_samples]
            sub_pred = y_pred[current_idx : current_idx + n_samples]
            
            # Calc Accuracy
            acc = np.mean(sub_true == sub_pred)
            results.append({"Subject": subject_id, "Accuracy": acc})
            
            current_idx += n_samples
            
        return pd.DataFrame(results)

    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return None

def main():
    # 1. Get the Map
    subject_counts = get_subject_sample_counts()
    total_expected = sum(subject_counts.values())
    print(f"Total expected samples in Test Set (Session 3): {total_expected}")
    
    all_subject_accuracies = []

    # 2. Scan Errors folder
    print(f"Scanning '{ERRORS_FOLDER}' for predictions.npy (Filtering for '1s' models)...")
    valid_files = 0
    
    for root, dirs, files in os.walk(ERRORS_FOLDER):
        # --- FILTER: Only process folders containing "1s" ---
        if "1s" not in root:
            continue
            
        for file in files:
            if file == "predictions.npy":
                full_path = os.path.join(root, file)
                
                # Analyze this specific run
                df_run = analyze_predictions_file(full_path, subject_counts)
                
                if df_run is not None:
                    valid_files += 1
                    all_subject_accuracies.append(df_run)
                else:
                    pass 

    if not all_subject_accuracies:
        print("No valid prediction files found that match the dataset size and '1s' criteria.")
        return

    # 3. Aggregate
    combined_df = pd.concat(all_subject_accuracies)
    summary = combined_df.groupby("Subject")["Accuracy"].agg(['mean', 'min', 'max', 'count']).reset_index()
    summary.columns = ['Subject', 'Avg_Accuracy', 'Min_Accuracy', 'Max_Accuracy', 'Num_Runs']
    
    # 4. Classify
    summary['Classification'] = summary['Avg_Accuracy'].apply(get_classification)
    summary = summary.sort_values(by="Avg_Accuracy", ascending=False)

    # 5. Report
    print("\n" + "="*60)
    print("SUBJECT PERFORMANCE REPORT (Based on Session 3 Holdout)")
    print("="*60)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("="*60)
    
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"Report saved to '{OUTPUT_FILE}' with data from {valid_files} valid prediction files.")

if __name__ == "__main__":
    main()