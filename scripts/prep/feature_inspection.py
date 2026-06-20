import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io
from sklearn.preprocessing import StandardScaler

def load_subject_data(data_folder, subject_id):
    # ... (Same loading logic as before) ...
    print(f"Loading data for Subject {subject_id}...")
    label_file = os.path.join(data_folder, "label.mat")
    try:
        label_mat = scipy.io.loadmat(label_file)
        trial_labels = label_mat['label'][0]
    except:
        return None, None

    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = [label_map[l] for l in trial_labels]
    
    X_list = []
    y_list = []
    
    files = [f for f in os.listdir(data_folder) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    if not files: return None, None

    for fname in sorted(files):
        path = os.path.join(data_folder, fname)
        try: mat = scipy.io.loadmat(path)
        except: continue
        
        for trial_i in range(1, 16):
            key = f"de_LDS{trial_i}"
            if key not in mat: continue
            data = mat[key] 
            data = np.transpose(data, (1, 0, 2)) 
            data_flat = data.reshape(data.shape[0], -1)
            X_list.append(data_flat)
            y_list.append(np.full(data.shape[0], mapped_labels[trial_i-1]))
            
    if not X_list: return None, None
    return np.concatenate(X_list), np.concatenate(y_list)

def main():
    data_folder = "Data/ExtractedFeatures_1s"
    base_dir = "Features"
    bands_dir = os.path.join(base_dir, "All_Bands")
    os.makedirs(bands_dir, exist_ok=True)
    
    # Band names and their indices in the 5-band stack
    # 0: Delta, 1: Theta, 2: Alpha, 3: Beta, 4: Gamma
    bands = {
        "Delta": 0,
        "Theta": 1,
        "Alpha": 2,
        "Beta": 3,
        "Gamma": 4
    }
    
    # Let's look at Subject 1 and 4 (who had bad overlap)
    subjects_to_check = [1, 4] 
    
    for subject_id in subjects_to_check:
        X, y = load_subject_data(data_folder, subject_id)
        if X is None: continue
        
        # Create a figure with 5 subplots (one for each band)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"Subject {subject_id}: Energy Distribution per Band", fontsize=16)
        
        for i, (band_name, band_idx) in enumerate(bands.items()):
            # Extract indices for this band (every 5th index starting at band_idx)
            indices = np.arange(band_idx, 310, 5)
            band_energy = np.mean(X[:, indices], axis=1)
            
            sns.boxplot(x=y, y=band_energy, palette='viridis', ax=axes[i])
            axes[i].set_title(band_name)
            axes[i].set_xticklabels(['Neg', 'Neu', 'Pos'])
            axes[i].set_xlabel("")
            if i == 0: axes[i].set_ylabel("DE Energy")
            else: axes[i].set_ylabel("")
            
        plt.tight_layout()
        plt.savefig(os.path.join(bands_dir, f"subject_{subject_id}_all_bands.png"))
        plt.close()
        print(f"Saved plot for Subject {subject_id}")

if __name__ == "__main__":
    main()