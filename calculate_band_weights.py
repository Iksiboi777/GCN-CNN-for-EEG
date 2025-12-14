import numpy as np
import os
import scipy.io
import json
from sklearn.feature_selection import f_classif

def load_subject_data(data_folder, subject_id):
    # Reuse the loading logic to get X and y
    label_file = os.path.join(data_folder, "label.mat")
    try:
        label_mat = scipy.io.loadmat(label_file)
        trial_labels = label_mat['label'][0]
    except: return None, None

    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = [label_map[l] for l in trial_labels]
    
    X_list = []
    y_list = []
    
    files = [f for f in os.listdir(data_folder) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    
    for fname in sorted(files):
        path = os.path.join(data_folder, fname)
        try: mat = scipy.io.loadmat(path)
        except: continue
        
        for trial_i in range(1, 16):
            key = f"de_LDS{trial_i}"
            if key not in mat: continue
            data = mat[key] # (62, N, 5)
            data = np.transpose(data, (1, 0, 2)) 
            
            # Average over the 62 channels to get global band importance
            # Shape becomes (N, 5)
            data_avg = np.mean(data, axis=1)
            
            X_list.append(data_avg)
            y_list.append(np.full(data.shape[0], mapped_labels[trial_i-1]))
            
    if not X_list: return None, None
    return np.concatenate(X_list), np.concatenate(y_list)

def main():
    data_folder = "Data/ExtractedFeatures_1s"
    output_file = "Params/subject_band_weights.json"
    os.makedirs("Params", exist_ok=True)
    
    weights_dict = {}
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    
    print("Calculating Fisher Scores (F-values) per subject...")
    
    for subject_id in range(1, 16):
        X, y = load_subject_data(data_folder, subject_id)
        if X is None: continue
        
        # Calculate ANOVA F-value for each of the 5 bands
        # f_classif returns (f_scores, p_values)
        f_scores, _ = f_classif(X, y)
        
        # Normalize weights
        # Option 1: Softmax (makes them sum to 1, good for attention)
        # Option 2: Relative scaling (sum to 5, so average weight is 1) -> Preserves magnitude better
        
        # Let's use Relative Scaling so we don't shrink the data too much
        # If a band is useless, its weight goes near 0. If useful, > 1.
        weights = (f_scores / np.sum(f_scores)) * 5
        
        weights_dict[str(subject_id)] = weights.tolist()
        
        print(f"Subject {subject_id} Weights:")
        for b, w in zip(bands, weights):
            print(f"  {b}: {w:.4f}")
            
    with open(output_file, 'w') as f:
        json.dump(weights_dict, f, indent=4)
    print(f"\nWeights saved to {output_file}")

if __name__ == "__main__":
    main()