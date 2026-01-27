# ICA and no Per-Session Filtering
import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import RobustScaler
import joblib

# --- CONFIGURATION ---
RAW_DATA_DIR = "../Data/Preprocessed_EEG" 
OUTPUT_DIR = "../Data/Custom_2s_25overlap_FAST2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SFREQ = 200 
WINDOW_SEC = 2.0
OVERLAP_RATIO = 0.25 
STEP_SEC = WINDOW_SEC * (1 - OVERLAP_RATIO) 
BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}

# Indices for FAS (Frontal Alpha Asymmetry)
# Based on SEED 62-channel layout: F3 is 7, F4 is 11 (0-indexed)
IDX_F3, IDX_F4, IDX_ALPHA = 7, 11, 2 

# --- FAST MATH UTILS ---

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)

def apply_lds_fast(sequence):
    """ High-speed Kalman smoothing for DE features """
    n_win, n_ch, n_feat = sequence.shape
    smoothed = np.zeros_like(sequence)
    for c in range(n_ch):
        for f in range(n_feat):
            data = sequence[:, c, f]
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x, kf.F, kf.H, kf.P, kf.R, kf.Q = np.array([[data[0]]]), np.array([[1.]]), np.array([[1.]]), 10., 0.01, 0.001
            res = []
            for z in data:
                kf.predict()
                kf.update(z)
                res.append(kf.x[0,0])
            smoothed[:, c, f] = np.array(res)
    return smoothed



def process_fast_trial(trial_data, rest_data):
    """ 
    High-speed trial processing: 
    Baseline -> Filter -> DE -> LDS -> FAS
    """
    # 1. Baseline Correction
    trial_data = trial_data - np.mean(rest_data, axis=1, keepdims=True)
    
    # 2. Fast Band Filtering (Whole Trial)
    band_signals = []
    for (low, high) in BANDS.values():
        band_signals.append(butter_bandpass_filter(trial_data, low, high, SFREQ))
    band_signals = np.stack(band_signals, axis=0) # (5, 62, Time)

    # 3. Windowing & DE Extraction
    win_samples = int(WINDOW_SEC * SFREQ)
    step_samples = int(STEP_SEC * SFREQ)
    n_windows = (trial_data.shape[1] - win_samples) // step_samples + 1
    
    raw_feats = []
    for i in range(n_windows):
        start, end = i * step_samples, i * step_samples + win_samples
        win = band_signals[:, :, start:end] 
        # DE calculation
        variance = np.var(win, axis=-1)
        de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-6)) # (5, 62)
        raw_feats.append(np.hstack([de.T, variance.T])) # DE (5) + Var (5) = 10
    
    X_trial = np.stack(raw_feats) # (Windows, 62, 10)
    
    # 4. LDS Smoothing
    X_trial = apply_lds_fast(X_trial)

    # 5. FAS calculation (11th feature)
    fas = X_trial[:, IDX_F4, IDX_ALPHA] - X_trial[:, IDX_F3, IDX_ALPHA]
    fas_col = np.tile(fas[:, np.newaxis, np.newaxis], (1, 62, 1))
    
    return np.concatenate([X_trial, fas_col], axis=2)



# --- MAIN EXECUTION ---

def build_dataset_fast():
    label_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "label.mat"))
    TRIAL_LABELS = label_mat['label'][0]
    files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.mat') and f != 'label.mat'])
    
    total_windows, all_data = 0, []
    
    for f_name in files:
        subject_id = int(f_name.split('_')[0])
        print(f">>> Processing Subject {subject_id}...")
        mat = sio.loadmat(os.path.join(RAW_DATA_DIR, f_name))
        keys = [k for k in mat.keys() if 'eeg' in k.lower()]
        
        for idx, key in enumerate(keys):
            data = mat[key]
            # Split rest and movie
            rest, movie = data[:, :int(5*SFREQ)], data[:, int(5*SFREQ):]
            
            feats = process_fast_trial(movie, rest)
            label = TRIAL_LABELS[idx] + 1
            
            all_data.append({'feat': feats, 'y': label, 'sub': subject_id})
            total_windows += feats.shape[0]

    # Save to Memmap
    X_mm = np.memmap(os.path.join(OUTPUT_DIR, "X_custom.dat"), dtype='float32', mode='w+', shape=(total_windows, 62, 11))
    y_meta, sub_meta = np.zeros(total_windows), np.zeros(total_windows)
    
    curr = 0
    scaler = RobustScaler() 
    for entry in all_data:
        n = entry['feat'].shape[0]
        # Scaling per trial
        f_scaled = scaler.fit_transform(entry['feat'].reshape(-1, 11)).reshape(n, 62, 11)
        X_mm[curr:curr+n] = f_scaled
        y_meta[curr:curr+n] = entry['y']
        sub_meta[curr:curr+n] = entry['sub']
        curr += n
        
    X_mm.flush()
    np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y_meta)
    np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"), sub_meta)
    joblib.dump(X_mm.shape, os.path.join(OUTPUT_DIR, "X_shape.pkl"))
    print(f"Finished! Total samples: {total_windows}")

if __name__ == "__main__":
    build_dataset_fast()







# # ICA + Per-Trial Filtering
# import os
# import numpy as np
# import scipy.io as sio
# import mne
# import joblib
# from mne.preprocessing import ICA
# from mne_icalabel import label_components
# from filterpy.kalman import KalmanFilter
# from sklearn.preprocessing import RobustScaler

# # --- CONFIGURATION ---
# # CHECKLIST #1: Source is Preprocessed_EEG (Already 200Hz / Segmented)
# RAW_DATA_DIR = "../Data/Preprocessed_EEG" 
# OUTPUT_DIR = "../Data/Custom_2s_25overlap_Final"
# LOCS_FILE = 'channel_62_pos.locs' # CHECKLIST #13: Custom Positions
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFREQ = 200 
# WINDOW_SEC = 2.0
# OVERLAP_RATIO = 0.25 
# STEP_SEC = WINDOW_SEC * (1 - OVERLAP_RATIO) 
# BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}

# CHANNEL_NAMES = [
#     'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
#     'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
#     'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
#     'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
# ]

# # CHECKLIST #6: FAS Indices
# IDX_F3, IDX_F4, IDX_ALPHA = CHANNEL_NAMES.index('F3'), CHANNEL_NAMES.index('F4'), 2

# # --- 1. CORE MATH METHODS ---

# # CHECKLIST #7: LDS Smoothing (Kalman Filter)
# def apply_lds(sequence):
#     n_win, n_ch, n_feat = sequence.shape
#     smoothed = np.zeros_like(sequence)
#     for c in range(n_ch):
#         for f in range(n_feat):
#             data = sequence[:, c, f]
#             kf = KalmanFilter(dim_x=1, dim_z=1)
#             kf.x, kf.F, kf.H, kf.P = np.array([[data[0]]]), np.array([[1.]]), np.array([[1.]]), 10.
#             kf.R, kf.Q = 0.01, 0.001
#             res = []
#             for z in data:
#                 kf.predict()
#                 kf.update(z)
#                 res.append(kf.x[0,0])
#             smoothed[:, c, f] = np.array(res)
#     return smoothed

# # CHECKLIST #8: Moving Average
# def apply_moving_avg(data, window_size=5):
#     ret = np.cumsum(data, axis=0)
#     ret[window_size:] = ret[window_size:] - ret[:-window_size]
#     return ret[window_size - 1:] / window_size

# def compute_de(signal):
#     # CHECKLIST #4: Differential Entropy
#     return 0.5 * np.log(2 * np.pi * np.e * (np.var(signal, axis=1) + 1e-6))

# # --- 2. ICA CLEANING ---

# # CHECKLIST #2: Removal of eye/muscle artifacts via ICA
# def apply_ica_cleaning(raw, locs_path):
#     # CHECKLIST #13: Load custom locs to provide coordinates for ICLabel
#     montage = mne.channels.read_custom_montage(locs_path)
#     raw.set_montage(montage, on_missing='ignore')
    
#     # Requirement: 1Hz highpass and CAR for ICLabel
#     raw_ica = raw.copy().filter(l_freq=1.0, h_freq=99.0, verbose=False)
#     raw_ica.set_eeg_reference('average', projection=False, verbose=False)
    
#     ica = ICA(n_components=15, max_iter='auto', random_state=97, method='infomax', fit_params=dict(extended=True))
#     ica.fit(raw_ica, verbose=False)
    
#     # Automated classification
#     labels = label_components(raw_ica, ica, method='iclabel')['labels']
#     ica.exclude = [i for i, label in enumerate(labels) if label in ['eye', 'muscle', 'heart']]
    
#     cleaned_raw = raw.copy()
#     ica.apply(cleaned_raw, verbose=False)
#     return cleaned_raw

# # --- 3. TRIAL PROCESSOR ---

# def process_single_trial(trial_data, rest_data):
#     # CHECKLIST #5: Baseline Correction (Rest-period subtraction)
#     trial_data = trial_data - np.mean(rest_data, axis=1, keepdims=True)
    
#     info = mne.create_info(CHANNEL_NAMES, SFREQ, ch_types='eeg')
#     raw = mne.io.RawArray(trial_data, info, verbose=False)
    
#     # CHECKLIST #2: Apply ICA to the trial segment
#     raw = apply_ica_cleaning(raw, LOCS_FILE) 
    
#     # Band Filtering (No re-filtering 0-75, just sub-bands)
#     band_signals = []
#     for (low, high) in BANDS.values():
#         filt = raw.copy().filter(l_freq=low, h_freq=high, verbose=False)
#         band_signals.append(filt.get_data())
#     band_signals = np.stack(band_signals, axis=0)

#     # CHECKLIST #3: Windowing (2s, 0.5s overlap)
#     win_samples, step_samples = int(WINDOW_SEC * SFREQ), int(STEP_SEC * SFREQ)
#     n_windows = (trial_data.shape[1] - win_samples) // step_samples + 1
    
#     trial_raw_feats = []
#     for i in range(n_windows):
#         start, end = i * step_samples, i * step_samples + win_samples
#         win = band_signals[:, :, start:end] 
#         # DE (5) + Variance (5) = 10 Features
#         de = np.array([compute_de(win[b]) for b in range(5)]).T
#         pwr = np.array([np.var(win[b], axis=1) for b in range(5)]).T
#         trial_raw_feats.append(np.hstack([de, pwr]))
    
#     # Smoothing
#     feats = np.stack(trial_raw_feats) 
#     feats = apply_lds(feats) # CHECKLIST #7
#     feats = apply_moving_avg(feats, window_size=5) # CHECKLIST #8

#     # CHECKLIST #6: FAS (11th Feature)
#     fas = feats[:, IDX_F4, IDX_ALPHA] - feats[:, IDX_F3, IDX_ALPHA]
#     fas_col = np.tile(fas[:, np.newaxis, np.newaxis], (1, 62, 1))
    
#     return np.concatenate([feats, fas_col], axis=2)

# # --- 4. MAIN ASSEMBLY ---

# def build_dataset_memmap():
#     label_mat = sio.loadmat(os.path.join(RAW_DATA_DIR, "label.mat"))
#     TRIAL_LABELS = label_mat['label'][0]
#     files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.mat') and f != 'label.mat'])
    
#     total_windows, all_data = 0, []
    
#     # CHECKLIST #12: Instantiate Scaler for trial-level application
#     scaler = RobustScaler()

#     for f_name in files:
#         subject_id = int(f_name.split('_')[0])
#         print(f"--- Subject {subject_id} ---")
#         mat = sio.loadmat(os.path.join(RAW_DATA_DIR, f_name))
#         keys = [k for k in mat.keys() if 'eeg' in k.lower()]
        
#         for idx, key in enumerate(keys):
#             data = mat[key]
#             # CHECKLIST #5: Define 5s rest baseline
#             rest, movie = data[:, :int(5*SFREQ)], data[:, int(5*SFREQ):]
            
#             # Feature Extraction
#             feats = process_single_trial(movie, rest)
#             label = TRIAL_LABELS[idx] + 1 # CHECKLIST #9: Label map
            
#             # CHECKLIST #12: Apply Robust Scaling per trial
#             n_samples = feats.shape[0]
#             feats_scaled = scaler.fit_transform(feats.reshape(-1, 11)).reshape(n_samples, 62, 11)
            
#             all_data.append({'feat': feats_scaled, 'y': label, 'sub': subject_id})
#             total_windows += n_samples

#     # CHECKLIST #10: Store in Memmap
#     X_mm = np.memmap(os.path.join(OUTPUT_DIR, "X_custom.dat"), dtype='float32', mode='w+', shape=(total_windows, 62, 11))
#     y_meta, sub_meta = np.zeros(total_windows), np.zeros(total_windows)
    
#     curr = 0
#     for entry in all_data:
#         n = entry['feat'].shape[0]
#         X_mm[curr:curr+n] = entry['feat']
#         y_meta[curr:curr+n] = entry['y']
#         sub_meta[curr:curr+n] = entry['sub']
#         curr += n
        
#     X_mm.flush()
#     np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y_meta)
#     np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"), sub_meta)
#     joblib.dump(X_mm.shape, os.path.join(OUTPUT_DIR, "X_shape.pkl"))
#     print("Done.")

# if __name__ == "__main__":
#     build_dataset_memmap()