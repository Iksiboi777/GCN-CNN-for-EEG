import os
import numpy as np
import torch
import scipy.io
from utils.feature_engineering import SmartPreprocessor, get_standard_channel_names

# --- CONFIG ---
DATA_FOLDER = "Data/ExtractedFeatures_1s"
CACHE_FOLDER = "Data/Cache_GraphSAGE_1s"
LABEL_FILE = os.path.join(DATA_FOLDER, "label.mat")
ROLLING_VAR_WINDOW = 9
os.makedirs(CACHE_FOLDER, exist_ok=True)

def compute_rolling_variance(data, window=ROLLING_VAR_WINDOW):
    pad_width = window // 2
    axis_time = 1 
    paddings = [(0,0)] * data.ndim
    paddings[axis_time] = (pad_width, pad_width)
    padded_data = np.pad(data, paddings, mode='edge')
    vars_list = []
    for i in range(data.shape[axis_time]):
        window_slice = padded_data[:, i : i + window, :]
        vars_list.append(np.var(window_slice, axis=1))
    return np.stack(vars_list, axis=1)

def run_caching():
    print(f"Starting Offline Caching to {CACHE_FOLDER}...")
    channel_names = get_standard_channel_names()
    preprocessor = SmartPreprocessor(channel_names)
    
    # Discovery
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    for f in files:
        sid = int(f.split('_')[0])
        if sid not in subject_files: subject_files[sid] = []
        subject_files[sid].append(f)

    for sub_id in sorted(subject_files.keys()):
        sess_files = sorted(subject_files[sub_id])
        for sess_idx, fname in enumerate(sess_files):
            sess_id = sess_idx + 1
            print(f"  Processing Subject {sub_id} | Session {sess_id}...")
            
            mat_data = scipy.io.loadmat(os.path.join(DATA_FOLDER, fname))
            for trial_id in range(1, 16):
                # Try common SEED keys
                data = mat_data.get(f"de_movingAve{trial_id}", mat_data.get(f"de_LDS{trial_id}"))
                if data is None: continue
                
                # Standardize shape to (62, Time, 5)
                # (Logic from your previous script to ensure 62 is axis 0)
                if data.shape[0] != 62:
                    if data.shape[1] == 62: data = data.transpose(1, 0, 2)
                    else: data = data.transpose(2, 0, 1)

                # Augment Features
                var_feat = compute_rolling_variance(data)
                combined = np.concatenate([data, var_feat], axis=-1) # (62, T, 10)
                
                # Normalize this specific session block
                # Preprocessor expects (T, 62, 10)
                combined_t = combined.transpose(1, 0, 2)
                normalized = preprocessor.process_subject(combined_t)
                
                # Save as Cache
                save_path = os.path.join(CACHE_FOLDER, f"Sub{sub_id}_Sess{sess_id}_Trial{trial_id}.npy")
                np.save(save_path, normalized)

    print("Caching Complete. You can now run the training script instantly.")

if __name__ == "__main__":
    run_caching()