import os
import numpy as np
import scipy.io as sio
from scipy.stats import zscore

# --- Configuration ---
input_folder = "Data/Cleaned_EEG_ICA"
output_folder = "Data/Raw_Data_For_CNN"
os.makedirs(output_folder, exist_ok=True)

# Parameters
fs = 200
window_sec = 2
overlap_sec = 1
window_size = int(window_sec * fs)
step_size = int((window_sec - overlap_sec) * fs)

# Labels Mapping
TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}

# --- 1. Map Files to Sessions ---
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

# --- 2. Processing Loop ---
all_X = []
all_y = []
all_sessions = []
all_subjects = [] # NEW: Store Subject ID

print(f"Found {len(files)} files. Starting segmentation...")

for file_name in files:
    if file_name not in file_to_session: continue
    
    file_path = os.path.join(input_folder, file_name)
    session_id = file_to_session[file_name]
    subject_id = int(file_name.split('_')[0]) # Extract Subject ID
    
    try:
        mat_data = sio.loadmat(file_path)
    except:
        continue
        
    keys = sorted([k for k in mat_data.keys() if 'eeg' in k and not k.startswith('__')])
    if len(keys) != 15: continue

    for i, key in enumerate(keys):
        data = mat_data[key]
        if data.shape[0] != 62: data = data.T
        
        n_samples = data.shape[1]
        label = LABEL_MAP[TRIAL_LABELS[i]]
        
        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            segment = data[:, start:end]
            
            segment = zscore(segment, axis=1)
            segment = np.nan_to_num(segment)
            
            all_X.append(segment)
            all_y.append(label)
            all_sessions.append(session_id)
            all_subjects.append(subject_id) # Store Subject ID

# Convert to Numpy Arrays
X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.int64)
sessions = np.array(all_sessions, dtype=np.int64)
subjects = np.array(all_subjects, dtype=np.int64) # NEW

print("-" * 30)
print(f"Processing Complete.")
print(f"Final Data Shape: {X.shape}")
print(f"Sessions Shape: {sessions.shape}")
print(f"Subjects Shape: {subjects.shape}")

# Save
np.save(os.path.join(output_folder, "X_raw.npy"), X)
np.save(os.path.join(output_folder, "y_labels.npy"), y)
np.save(os.path.join(output_folder, "sessions.npy"), sessions)
np.save(os.path.join(output_folder, "subjects.npy"), subjects) # NEW
print(f"Saved .npy files to {output_folder}")