import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def plot_session_drift_comparison(results_root, subject_id):
    """
    Plots the Confusion Matrices of the 3 session holdouts side-by-side.
    Allows us to see which session has the most 'geometric drift'.
    """
    sessions = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Subject {subject_id}: Cross-Session Generalization Drift", fontsize=16)

    class_names = ['Negative', 'Neutral', 'Positive']

    for i, test_sess in enumerate(sessions):
        # Path to the specific permutation results
        p_tag = f"TestSess_{test_sess}"
        matrix_path = os.path.join(results_root, p_tag, "predictions.npy")
        
        if not os.path.exists(matrix_path):
            print(f"Warning: Matrix for Test Session {test_sess} not found.")
            axes[i].text(0.5, 0.5, f"Missing Session {test_sess}", ha='center')
            continue

        cm = np.load(matrix_path)
        # Normalize by row (true labels) to show recall percentage
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        
        axes[i].set_title(f"Held-out: Session {test_sess}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"Subject_{subject_id}_Session_Drift_Report.png"
    plt.savefig(save_path)
    print(f"Drift report saved to {save_path}")
    plt.show()

# Example Usage (Run this after train_sage_advanced.py completes):
plot_session_drift_comparison("Errors/GraphSAGE_Advanced_1s/Attempt_5_Phase2", subject_id=1)