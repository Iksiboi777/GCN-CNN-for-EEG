import os
import re
import numpy as np

# --- CONFIG ---
# Update this to match the folder where your results are stored
BASE_RESULTS_DIR = "Errors/GraphSAGE_Advanced_1s/Attempt_65_FullDataset"

def extract_accuracy_from_report(file_path):
    """Parses the classification_report.txt to find the 'accuracy' line."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        content = f.read()
        # Look for the accuracy line: 'accuracy                           0.79'
        match = re.search(r'accuracy\s+([\d\.]+)', content)
        if match:
            return float(match.group(1))
    return None

def main():
    print(f"{'Subject':<10} | {'Sess 1 Acc':<10} | {'Sess 2 Acc':<10} | {'Sess 3 Acc':<10} | {'Average':<10}")
    print("-" * 65)

    all_subject_means = []

    for sub_id in range(1, 16):
        sub_scores = []
        # We check each session holdout permutation
        for test_sess in [1, 2, 3]:
            report_path = os.path.join(BASE_RESULTS_DIR, f"Sub{sub_id}", f"Test{test_sess}", "classification_report.txt")
            acc = extract_accuracy_from_report(report_path)
            
            if acc is not None:
                sub_scores.append(acc)
            else:
                sub_scores.append(0.0) # Mark as missing if file not found

        avg_acc = np.mean(sub_scores) if sub_scores else 0
        all_subject_means.append(avg_acc)

        print(f"Sub {sub_id:<6} | {sub_scores[0]:<10.4f} | {sub_scores[1]:<10.4f} | {sub_scores[2]:<10.4f} | {avg_acc:<10.4f}")

    print("-" * 65)
    print(f"GLOBAL DATASET AVERAGE: {np.mean(all_subject_means):.4f}")

if __name__ == "__main__":
    main()