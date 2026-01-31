# # import mne
# # import scipy.io as sio
# # import numpy as np
# # import os
# # from mne.preprocessing import ICA
# # import mne_icalabel

# # # --- Configuration ---
# # input_folder = "../Data/Preprocessed_EEG"
# # output_folder_standard = "../Data/Cleaned_EEG_ICA_1_49"
# # output_folder_gamma = "../Data/Cleaned_EEG_ICA_50_75"

# # os.makedirs(output_folder_standard, exist_ok=True)
# # os.makedirs(output_folder_gamma, exist_ok=True)

# # # Define Metadata
# # sfreq = 200
# # ch_names = [
# #     'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
# #     'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
# #     'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
# #     'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
# #     'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
# # ]
# # ch_types = ['eeg'] * len(ch_names)
# # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# # montage = mne.channels.read_custom_montage('../channel_62_pos.locs')

# # # --- Processing Loop ---
# # # Get all .mat files
# # files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

# # for file_name in files:
# #     print(f"Processing file: {file_name}")
# #     file_path = os.path.join(input_folder, file_name)
    
# #     try:
# #         mat_data = sio.loadmat(file_path)
# #     except Exception as e:
# #         print(f"Error loading {file_name}: {e}")
# #         continue
        
# #     cleaned_data_dict_standard = {}
# #     cleaned_data_dict_gamma = {}
    
# #     # Copy non-EEG keys (like labels) to the new dictionary
# #     for key in mat_data:
# #         if not key.startswith('__') and not key.startswith('eeg_') and not key.startswith('djc_eeg'):
# #              cleaned_data_dict_standard[key] = mat_data[key]
# #              cleaned_data_dict_gamma[key] = mat_data[key]

# #     # Iterate over trials (eeg_1 ... eeg_15)
# #     # Note: Keys might be 'eeg_1' or 'djc_eeg1' depending on the file version. 
# #     # We iterate all keys and check if they look like EEG data.
# #     for key in mat_data.keys():
# #         if key.startswith('__'): continue
        
# #         # Check if this key contains EEG data (usually shape (62, N) or (N, 62))
# #         data = mat_data[key]
# #         if not isinstance(data, np.ndarray) or data.ndim != 2:
# #             continue
            
# #         # Heuristic: EEG data usually has 62 channels
# #         if data.shape[0] == 62:
# #             pass # Correct orientation
# #         elif data.shape[1] == 62:
# #             data = data.T # Transpose
# #         else:
# #             continue # Not an EEG data array
            
# #         print(f"  Cleaning trial: {key}")
        
# #         # Create Raw object
# #         raw = mne.io.RawArray(data, info)
# #         raw.set_montage(montage)
        
# #         # Filter for ICA (1Hz highpass)
# #         # Note: We filter a COPY for ICA calculation, but apply cleaning to the ORIGINAL (or 0.1Hz filtered) data
# #         # to preserve low frequencies if needed. 
# #         raw_ica = raw.copy().filter(1.0, 40.0, verbose=False)
        
# #         # Run ICA
# #         ica = ICA(n_components=15, max_iter='auto', random_state=97, verbose=False)
# #         ica.fit(raw_ica)
        
# #         # Label and Exclude
# #         ic_labels = mne_icalabel.label_components(raw_ica, ica, method="iclabel")
# #         labels = ic_labels["labels"]
# #         exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        
# #         # Apply to original data
# #         ica.apply(raw, exclude=exclude_idx, verbose=False)
        
# #         # 1. Standard Band (1-49 Hz)
# #         raw_standard = raw.copy().filter(1.0, 75.0, fir_design='firwin', verbose=False)
# #         cleaned_data_dict_standard[key] = raw_standard.get_data()
        
# #         # 2. Gamma Band (50-75 Hz)
# #         raw_gamma = raw.copy().filter(50.0, 75.0, fir_design='firwin', verbose=False)
# #         cleaned_data_dict_gamma[key] = raw_gamma.get_data()
        
# #     # Save standard
# #     save_path_std = os.path.join(output_folder_standard, file_name)
# #     sio.savemat(save_path_std, cleaned_data_dict_standard)
# #     print(f"Saved standard file: {save_path_std}")
    
# #     # Save gamma
# #     save_path_gamma = os.path.join(output_folder_gamma, file_name)
# #     sio.savemat(save_path_gamma, cleaned_data_dict_gamma)
# #     print(f"Saved gamma file: {save_path_gamma}")

# # print("All files processed.")


# import os
# import numpy as np
# import scipy.io as sio
# from scipy.stats import zscore
# from numpy.lib.format import open_memmap
# import mne
# from mne.preprocessing import ICA
# import warnings

# # Suppress MNE/Warning spam for cleaner logs
# warnings.filterwarnings("ignore")
# mne.set_log_level("WARNING")

# def process_raw_pipeline():
#     # --- CONFIGURATION ---
#     input_folder = "Data/Preprocessed_EEG"
#     output_folder = "Data/Preprocessed_Raw_SOTA"
#     os.makedirs(output_folder, exist_ok=True)
    
#     print(f"\n🚀 Starting Preprocessing Pipeline")
#     print(f"   📂 Input:  {input_folder}")
#     print(f"   📂 Output: {output_folder}")

#     # Parameters
#     fs = 200
#     window_sec = 2
#     overlap_pct = 0.5
#     window_size = int(window_sec * fs)      # 400 pts
#     step_size = int(window_size * (1 - overlap_pct)) # 200 pts

#     # SEED Standard Labels (Fixed order for all subjects)
#     TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
#     LABEL_MAP = {-1: 0, 0: 1, 1: 2}

#     # Setup MNE Info (Needed for ICA)
#     ch_names = [
#         'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#         'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
#         'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#         'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
#         'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
#     ]
#     info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
#     montage = mne.channels.make_standard_montage('standard_1020')
#     info.set_montage(montage, on_missing='ignore')

#     # --- 1. FILE DISCOVERY ---
#     print("\n🔍 Scanning files...")
#     files = [f for f in os.listdir(input_folder) if f.endswith('.mat') and 'label' not in f]
#     files.sort()
    
#     # Map files to Session IDs (1, 2, 3)
#     # Assumes file naming convention: Subject_Date.mat
#     subject_map = {}
#     for f in files:
#         sub_id = int(f.split('_')[0])
#         if sub_id not in subject_map: subject_map[sub_id] = []
#         subject_map[sub_id].append(f)
    
#     file_to_session = {}
#     valid_files = []
    
#     for sub_id, sub_files in subject_map.items():
#         sub_files.sort() # Temporal order implies session 1->3
#         for idx, fname in enumerate(sub_files):
#             file_to_session[fname] = idx + 1
#             valid_files.append(fname)
            
#     print(f"   -> Found {len(valid_files)} subject/session files.")

#     # --- 2. PASS 1: COUNT SAMPLES ---
#     print("\n🧮 Pass 1: Calculating total dataset size...")
#     total_samples = 0
    
#     for file_name in valid_files:
#         try:
#             mat_info = sio.whosmat(os.path.join(input_folder, file_name))
#             keys = [x[0] for x in mat_info if 'eeg' in x[0] and not x[0].startswith('__')]
            
#             # Check for exactly 15 trials
#             valid_keys = 0
#             for key, shape, dtype in mat_info:
#                 if key not in keys: continue
#                 valid_keys += 1
#                 n_pts = shape[1] if shape[0] == 62 else shape[0]
                
#                 if n_pts >= window_size:
#                     total_samples += (n_pts - window_size) // step_size + 1
                    
#         except Exception as e:
#             print(f"   ⚠️ Error reading shape of {file_name}: {e}")

#     print(f"   -> Total Segments: {total_samples}")
#     print(f"   -> Allocating {total_samples * 62 * window_size * 4 / 1e9:.2f} GB on disk.")

#     # --- 3. ALLOCATE STORAGE ---
#     X_path = os.path.join(output_folder, f"X_Raw.npy")
#     X = open_memmap(X_path, mode='w+', dtype=np.float32, shape=(total_samples, 62, window_size))
    
#     y = np.zeros(total_samples, dtype=np.int64)
#     sessions = np.zeros(total_samples, dtype=np.int64)
#     subjects = np.zeros(total_samples, dtype=np.int64)

#     # --- 4. PASS 2: PROCESSING ---
#     print("\n⚙️ Pass 2: Processing (ICA -> ZScore -> Segment)...")
#     current_idx = 0
    
#     for file_name in valid_files:
#         path = os.path.join(input_folder, file_name)
#         session_id = file_to_session[file_name]
#         subject_id = int(file_name.split('_')[0])
        
#         print(f"   Processing: {file_name} (Sub {subject_id}, Sess {session_id})...", end="\r")
        
#         try:
#             mat_data = sio.loadmat(path)
#         except Exception as e:
#             print(f"\n   ❌ Corrupt file {file_name}: {e}")
#             continue

#         # Get keys eeg_1 ... eeg_15 sorted
#         keys = [k for k in mat_data.keys() if 'eeg_' in k and not k.startswith('__')]
#         keys.sort(key=lambda x: int(x.split('_')[1]))

#         for i, key in enumerate(keys):
#             if i >= 15: break
            
#             raw_data = mat_data[key] # (62, T) or (T, 62)
#             if raw_data.shape[0] != 62: raw_data = raw_data.T # Ensure (62, T)
            
#             # --- 1. ICA CLEANING ---
#             # Create MNE object
#             raw_mne = mne.io.RawArray(raw_data, info, verbose=False)
            
#             # Highpass filter 1Hz for stable ICA
#             raw_mne.filter(l_freq=1.0, h_freq=None, verbose=False)
            
#             # Fit & Apply FastICA (15 components is standard for SEED)
#             ica = ICA(n_components=15, method='fastica', random_state=42, verbose=False)
#             ica.fit(raw_mne, verbose=False)
#             cleaned_mne = ica.apply(raw_mne, verbose=False)
            
#             data_cleaned = cleaned_mne.get_data() # (62, T)

#             # --- 2. GLOBAL Z-SCORE ---
#             # Standardize across the ENTIRE trial (axis 1)
#             data_norm = zscore(data_cleaned, axis=1)
#             data_norm = np.nan_to_num(data_norm).astype(np.float32)

#             # --- 3. SEGMENTATION ---
#             n_pts = data_norm.shape[1]
#             label_val = LABEL_MAP[TRIAL_LABELS[i]]
            
#             trial_segments = []
#             for start in range(0, n_pts - window_size + 1, step_size):
#                 end = start + window_size
#                 segment = data_norm[:, start:end]
#                 trial_segments.append(segment)
            
#             if not trial_segments: continue
            
#             # --- 4. SAVE ---
#             n_new = len(trial_segments)
#             end_idx = current_idx + n_new
            
#             X[current_idx : end_idx] = np.array(trial_segments)
#             y[current_idx : end_idx] = label_val
#             sessions[current_idx : end_idx] = session_id
#             subjects[current_idx : end_idx] = subject_id
            
#             current_idx += n_new

#     # Cleanup
#     print(f"\n\n✅ DONE. Data saved to: {output_folder}")
#     del X
#     np.save(os.path.join(output_folder, "y_Raw.npy"), y)
#     np.save(os.path.join(output_folder, "sessions_Raw.npy"), sessions)
#     np.save(os.path.join(output_folder, "subjects_Raw.npy"), subjects)

# if __name__ == "__main__":
#     process_raw_pipeline()


import os
import numpy as np
import scipy.io as sio
from scipy.stats import zscore
import mne
from mne.preprocessing import ICA
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

def process_file_in_file_out():
    # ADJUSTED PATHS based on your previous run
    input_folder = "../Data/Preprocessed_EEG"
    output_folder = "../Data/Preprocessed_SOTA_Individual"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n🚀 STARTING: File-in -> File-out Pipeline (With Robust ICA)")
    print(f"   Input:  {input_folder}")
    print(f"   Output: {output_folder}")

    # CONFIGURATION
    fs = 200 
    window_sec = 2
    overlap_pct = 0.5
    
    window_size = int(window_sec * fs)      # 400 points
    step_size = int(window_size * (1 - overlap_pct)) # 200 points

    # CHANNEL SETUP (Required for ICA)
    ch_names = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
        'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
        'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    montage = mne.channels.read_custom_montage('channel_62_pos.locs')
    # Fallback to standard if custom mont. fails relative path
    if montage is None: 
         montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')

    # PROCESS FILES
    if not os.path.exists(input_folder):
        print(f"❌ ERROR: Input folder does not exist: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if f.endswith('.mat') and 'label' not in f]
    files.sort()

    if not files:
        print("❌ No matching .mat files found.")
        return

    for file_name in files:
        print(f"Processing: {file_name}...")
        
        try:
            # 1. Load Original File
            mat_path = os.path.join(input_folder, file_name)
            mat_data = sio.loadmat(mat_path)
            
            # --- ROBUST KEY FINDER ---
            # Find ANY key containing 'eeg' and not starting with underscore
            # AND ensure it matches the pattern of having a number in it.
            candidate_keys = [k for k in mat_data.keys() if 'eeg' in k.lower() and not k.startswith('__')]
            
            # Helper to extract trial number from string like 'djc_eeg1', 'eeg_5', 'eeg15'
            def get_trial_num(k):
                nums = re.findall(r'\d+', k)
                return int(nums[-1]) if nums else 999

            # Filter/Sort keys
            keys = sorted(candidate_keys, key=get_trial_num)

            if not keys:
                print(f"  ⚠️ Skipping: No EEG keys found in {file_name}. Keys were: {list(mat_data.keys())[:5]}")
                continue

            # --- PREPARE FOR ICA (Concatenation Strategy) ---
            trial_data_list = []
            trial_lengths = []
            
            valid_keys_processed = []

            for key in keys:
                raw_d = mat_data[key]
                # Fix Orientation
                if raw_d.shape[0] != 62: raw_d = raw_d.T 
                if raw_d.shape[0] != 62: continue # Not EEG data
                
                trial_data_list.append(raw_d)
                trial_lengths.append(raw_d.shape[1])
                valid_keys_processed.append(key)

            if not trial_data_list: 
                print("  ⚠️ Skipping: Data dimensions incorrect (expected 62 channels).")
                continue

            # Create one long continuous session
            concat_data = np.concatenate(trial_data_list, axis=1) # Shape (62, Total_Time)
            
            # --- APPLY MNE / ICA ---
            raw_mne = mne.io.RawArray(concat_data, info, verbose=False)
            
            # Filter specifically for ICA stability (1Hz Highpass)
            raw_for_ica = raw_mne.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
            
            # Fit ICA
            ica = ICA(n_components=15, method='fastica', random_state=42, verbose=False)
            ica.fit(raw_for_ica, verbose=False)
            
            # Apply cleaning to the full session
            cleaned_mne = ica.apply(raw_mne, verbose=False)
            cleaned_concat_data = cleaned_mne.get_data() # (62, Total_Time)

            # --- SPLIT BACK & SEGMENT ---
            processed_dict = {}
            current_ptr = 0
            
            for i, key in enumerate(valid_keys_processed):
                t_len = trial_lengths[i]
                
                # Extract the cleaned trial
                trial_cleaned = cleaned_concat_data[:, current_ptr : current_ptr + t_len]
                current_ptr += t_len
                
                # 1. Z-SCORE (Standardization per trial)
                data_norm = zscore(trial_cleaned, axis=1)
                data_norm = np.nan_to_num(data_norm)
                
                # 2. SEGMENTATION (2s window, 50% overlap)
                n_samples = data_norm.shape[1]
                segments = []
                
                for start in range(0, n_samples - window_size + 1, step_size):
                    end = start + window_size
                    seg = data_norm[:, start:end]
                    segments.append(seg)
                
                if segments:
                    # Save into dict as (N_segs, 62, 400)
                    processed_dict[key] = np.array(segments)

            # Save to disk
            if processed_dict:
                out_path = os.path.join(output_folder, file_name)
                sio.savemat(out_path, processed_dict)
                print(f"  -> ✅ Saved {len(processed_dict)} trials (Segmented) to {file_name}")
            else:
                print("  -> ⚠️ Processed dict empty (segmentation failed?).")

        except Exception as e:
            print(f"  -> ❌ FAILED {file_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ DONE. Files saved to Data/Preprocessed_SOTA_Individual")

if __name__ == "__main__":
    process_file_in_file_out()