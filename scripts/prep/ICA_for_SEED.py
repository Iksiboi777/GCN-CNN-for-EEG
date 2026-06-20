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