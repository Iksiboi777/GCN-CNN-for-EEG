import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# --- Configuration ---
DATA_FOLDER = "Data/ExtractedFeatures_4s" # Using 4s as per recent attempts
LABEL_FILE = os.path.join(DATA_FOLDER, "label.mat")

def load_and_normalize_data():
    print(f"Loading DE features from {DATA_FOLDER}...")
    try:
        label_mat = scipy.io.loadmat(LABEL_FILE)
        trial_labels = label_mat['label'][0]
    except FileNotFoundError:
        print(f"Error: Label file not found at {LABEL_FILE}")
        return None

    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = [label_map[l] for l in trial_labels]
    
    X_list = []
    y_list = []
    session_list = []
    subject_list = []
    
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    for f in files:
        parts = f.split('_')
        try: subj_id = int(parts[0])
        except: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)
        
    for subj_id in sorted(subject_files.keys()):
        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            file_path = os.path.join(DATA_FOLDER, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            for trial_i in range(1, 16):
                key = f"de_LDS{trial_i}"
                if key not in mat: continue
                data = mat[key]
                data = np.transpose(data, (1, 0, 2))
                num_samples = data.shape[0]
                X_list.append(data)
                y_list.append(np.full(num_samples, mapped_labels[trial_i - 1]))
                session_list.append(np.full(num_samples, session_id))
                subject_list.append(np.full(num_samples, subj_id))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    sessions = np.concatenate(session_list, axis=0)
    subjects = np.concatenate(subject_list, axis=0)
    
    # --- Normalization (Same as Attempt 6) ---
    print("Applying Subject-Specific & Session-Specific Normalization...")
    group_ids = subjects * 1000 + sessions
    unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
    n_groups = len(unique_groups)
    
    group_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    group_sq_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    
    np.add.at(group_sums, group_indices, X)
    group_counts = np.bincount(group_indices)
    group_means = group_sums / group_counts[:, None, None]
    
    expanded_means = group_means[group_indices]
    X_centered = X - expanded_means
    
    np.add.at(group_sq_sums, group_indices, X_centered ** 2)
    group_stds = np.sqrt(group_sq_sums / group_counts[:, None, None])
    group_stds[group_stds < 1e-6] = 1.0
    
    expanded_stds = group_stds[group_indices]
    X_norm = X_centered / expanded_stds
    
    return X_norm, y, sessions, subjects

def analyze_drift(X, y, sessions):
    print("Analyzing Session Drift...")
    
    # Flatten features for analysis: (N, 62*5)
    X_flat = X.reshape(X.shape[0], -1)
    
    # Split into Train (S1+S2) and Test (S3)
    mask_train = (sessions == 1) | (sessions == 2)
    mask_test = (sessions == 3)
    
    X_train = X_flat[mask_train]
    y_train = y[mask_train]
    X_test = X_flat[mask_test]
    y_test = y[mask_test]
    
    print(f"Train Samples (S1+S2): {X_train.shape[0]}")
    print(f"Test Samples (S3): {X_test.shape[0]}")
    
    # 1. Centroid Distance Analysis
    classes = [0, 1, 2]
    class_names = ['Neg', 'Neu', 'Pos']
    
    print("\n--- Class Centroid Shifts (Euclidean Distance) ---")
    for c in classes:
        # Get samples for this class
        train_c = X_train[y_train == c]
        test_c = X_test[y_test == c]
        
        # Compute centroids
        mean_train = np.mean(train_c, axis=0)
        mean_test = np.mean(test_c, axis=0)
        
        # Distance
        dist = np.linalg.norm(mean_train - mean_test)
        
        # Baseline: Distance between random subsets of Train
        # To see if the S3 shift is significant
        perm = np.random.permutation(len(train_c))
        half = len(train_c) // 2
        base_dist = np.linalg.norm(np.mean(train_c[perm[:half]], axis=0) - np.mean(train_c[perm[half:]], axis=0))
        
        print(f"Class {class_names[c]}: Shift = {dist:.4f} (Baseline Noise = {base_dist:.4f})")
        if dist > 2 * base_dist:
            print(f"  -> SIGNIFICANT DRIFT DETECTED")

    # 2. Visualization (PCA + t-SNE)
    # We'll take a random subset to speed up t-SNE
    print("\nGenerating t-SNE visualization...")
    subset_size = 3000
    indices = np.random.choice(len(X_flat), subset_size, replace=False)
    
    X_sub = X_flat[indices]
    y_sub = y[indices]
    sess_sub = sessions[indices]
    
    # PCA first to reduce noise
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_sub)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot 1: Colored by Session
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=sess_sub, palette='viridis', alpha=0.6)
    plt.title("t-SNE Colored by Session (1, 2, 3)")
    
    # Plot 2: Colored by Class (Split by Session Shape)
    plt.subplot(1, 2, 2)
    # Create a custom hue/style
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_sub, style=sess_sub, palette='deep', alpha=0.6)
    plt.title("t-SNE Colored by Class (Shape=Session)")
    
    plt.tight_layout()
    plt.savefig("session_drift_analysis.png")
    print("Saved plot to session_drift_analysis.png")

if __name__ == "__main__":
    data = load_and_normalize_data()
    if data:
        X, y, sessions, subjects = data
        analyze_drift(X, y, sessions)