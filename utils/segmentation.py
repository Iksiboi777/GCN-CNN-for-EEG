import os
import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from numpy.lib.format import open_memmap

def process_dataset(input_folder, suffix):
    print(f"\n--- Processing Dataset: {suffix} ---")
    output_folder = "Data/Raw_Data_w_Bands" 
    os.makedirs(output_folder, exist_ok=True)

    fs = 200
    window_sec = 2
    overlap_sec = 1
    window_size = int(window_sec * fs)
    step_size = int((window_sec - overlap_sec) * fs)

    TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    LABEL_MAP = {-1: 0, 0: 1, 1: 2}

    files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    subject_files = {}

    for f in files:
        parts = f.replace('.mat', '').split('_')
        if len(parts) < 2: continue
        
        subj_id = int(parts[0])
        if subj_id not in subject_files:
            subject_files[subj_id] = []
        subject_files[subj_id].append(f)

    file_to_session = {}
    for subj_id, file_list in subject_files.items():
        file_list.sort()
        for i, fname in enumerate(file_list):
            file_to_session[fname] = i + 1 

    print("Pass 1: Calculating total dataset size...")
    total_samples = 0
    valid_files = []
    
    for file_name in files:
        if file_name not in file_to_session: continue
        file_path = os.path.join(input_folder, file_name)
        
        try:
            mat_info = sio.whosmat(file_path)
            keys = sorted([x[0] for x in mat_info if 'eeg' in x[0] and not x[0].startswith('__')])
            if len(keys) != 15: continue
            
            file_samples = 0
            for key, shape, _ in mat_info:
                if key not in keys: continue
                
                if shape[0] == 62:
                    n_points = shape[1]
                else:
                    n_points = shape[0]
                
                if n_points >= window_size:
                    n_segments = (n_points - window_size) // step_size + 1
                    file_samples += n_segments
            
            total_samples += file_samples
            valid_files.append(file_name)
            
        except Exception as e:
            print(f"Skipping {file_name}: {e}")
            continue

    print(f"Total samples: {total_samples}. Allocating memory-mapped file ({total_samples * 62 * window_size * 4 / 1e9:.2f} GB)...")

    X_path = os.path.join(output_folder, f"X_raw_{suffix}.npy")
    X = open_memmap(X_path, mode='w+', dtype=np.float32, shape=(total_samples, 62, window_size))
    
    y = np.zeros(total_samples, dtype=np.int64)
    sessions = np.zeros(total_samples, dtype=np.int64)
    subjects = np.zeros(total_samples, dtype=np.int64)

    print("Pass 2: Segmentation and writing to disk...")
    current_idx = 0
    
    for file_name in valid_files:
        file_path = os.path.join(input_folder, file_name)
        session_id = file_to_session[file_name]
        subject_id = int(file_name.split('_')[0])
        
        try:
            mat_data = sio.loadmat(file_path)
        except:
            continue
            
        keys = sorted([k for k in mat_data.keys() if 'eeg' in k and not k.startswith('__')])
        
        for i, key in enumerate(keys):
            data = mat_data[key]
            if data.shape[0] != 62: data = data.T
            
            n_samples = data.shape[1]
            label = LABEL_MAP[TRIAL_LABELS[i]]
            
            trial_segments = []
            for start in range(0, n_samples - window_size + 1, step_size):
                end = start + window_size
                segment = data[:, start:end]
            
                segment = zscore(segment, axis=1)
                segment = np.nan_to_num(segment)
                trial_segments.append(segment)
            
            if not trial_segments:
                continue

            trial_segments_arr = np.array(trial_segments, dtype=np.float32)
            n_new = len(trial_segments_arr)
            
            X[current_idx : current_idx + n_new] = trial_segments_arr
            y[current_idx : current_idx + n_new] = label
            sessions[current_idx : current_idx + n_new] = session_id
            subjects[current_idx : current_idx + n_new] = subject_id
            
            current_idx += n_new

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Final Data Shape: {X.shape}")
    del X 
    
    np.save(os.path.join(output_folder, f"y_labels_{suffix}.npy"), y)
    np.save(os.path.join(output_folder, f"sessions_{suffix}.npy"), sessions)
    np.save(os.path.join(output_folder, f"subjects_{suffix}.npy"), subjects)
    print(f"Saved .npy files to {output_folder}")

if __name__ == "__main__":
    process_dataset("Data/Cleaned_EEG_ICA_1_49", "standard")  
    process_dataset("Data/Cleaned_EEG_ICA_50_75", "gamma")