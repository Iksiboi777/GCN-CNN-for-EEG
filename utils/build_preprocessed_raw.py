# import os
# import numpy as np
# import scipy.io as sio
# import mne
# from mne.preprocessing import ICA
# from mne_icalabel import label_components

# # --- CONFIGURATION ---
# RAW_DATA_DIR = "../Data/Preprocessed_EEG" 
# OUTPUT_DIR = "../Data/Preprocessed_2s_25overlap"
# LOCS_FILE = "channel_62_pos.locs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFREQ = 200
# WINDOW_SEC = 2.0
# OVERLAP_RATIO = 0.25
# STEP_SEC = WINDOW_SEC * (1 - OVERLAP_RATIO)

# CHANNEL_NAMES = [
#     'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
#     'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
#     'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
#     'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
# ]

# def segment_trial(trial_data):
#     """Cuts a continuous trial (62, Time) into (N, 62, 400)"""
#     win_samples = int(WINDOW_SEC * SFREQ)
#     step_samples = int(STEP_SEC * SFREQ)
#     total_samples = trial_data.shape[1]
    
#     if total_samples < win_samples: return None

#     n_windows = (total_samples - win_samples) // step_samples + 1
#     segments = []
    
#     for i in range(n_windows):
#         start = i * step_samples
#         end = start + win_samples
#         segments.append(trial_data[:, start:end])
        
#     return np.stack(segments) if segments else None

# def main():
#     # Copy label file
#     label_path = os.path.join(RAW_DATA_DIR, "label.mat")
#     if os.path.exists(label_path):
#         import shutil
#         shutil.copy(label_path, os.path.join(OUTPUT_DIR, "label.mat"))

#     files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.mat') and f != 'label.mat'])
    
#     for f_name in files:
#         print(f"Processing Session: {f_name}...")
#         mat = sio.loadmat(os.path.join(RAW_DATA_DIR, f_name))
        
#         # 1. Identify valid EEG keys
#         keys = [k for k in mat.keys() if 'eeg' in k.lower()]
#         # Sort keys to ensure time order (e.g., eeg1, eeg2, eeg3...)
#         # Usually keys are 'djc_eeg1', etc. Sorting by string usually works for 1-9, 
#         # but 10 might come after 1. A robust sort is safer:
#         keys.sort(key=lambda x: int(''.join(filter(str.isdigit, x)))) 

#         # 2. Stitch Data for Global ICA
#         trial_lengths = []
#         concat_data = []
        
#         for k in keys:
#             t_data = mat[k] # (62, Time)
#             trial_lengths.append(t_data.shape[1])
#             concat_data.append(t_data)
            
#         full_session_data = np.concatenate(concat_data, axis=1)
        
#         # 3. Apply Global ICA & Filter
#         info = mne.create_info(CHANNEL_NAMES, SFREQ, ch_types='eeg')
#         raw = mne.io.RawArray(full_session_data, info, verbose=False)
#         montage = mne.channels.read_custom_montage(LOCS_FILE)
#         raw.set_montage(montage, on_missing='ignore')
        
#         # Bandpass Filter (1-60Hz) covers all bands + removes DC drift
#         raw.filter(l_freq=1.0, h_freq=60.0, verbose=False) 
        
#         # ICA
#         ica = ICA(n_components=15, max_iter='auto', random_state=97, method='infomax', fit_params=dict(extended=True))
#         ica.fit(raw, verbose=False)
#         labels = label_components(raw, ica, method='iclabel')['labels']
#         ica.exclude = [i for i, label in enumerate(labels) if label in ['eye', 'muscle', 'heart']]
#         ica.apply(raw, verbose=False)
        
#         cleaned_full = raw.get_data() # (62, TotalTime)
        
#         # 4. Split Back & Segment
#         output_mat = {}
#         current_idx = 0
        
#         for i, k in enumerate(keys):
#             length = trial_lengths[i]
#             # Slice the cleaned chunk corresponding to this trial
#             trial_clean = cleaned_full[:, current_idx : current_idx + length]
#             current_idx += length
            
#             # Segment into windows
#             segmented = segment_trial(trial_clean)
            
#             if segmented is not None:
#                 output_mat[k] = segmented # (N, 62, 400)
                
#         # 5. Save
#         sio.savemat(os.path.join(OUTPUT_DIR, f_name), output_mat)

# if __name__ == "__main__":
#     main()


# import os
# import numpy as np
# import scipy.io as sio
# import mne
# from mne.preprocessing import ICA
# from mne_icalabel import label_components

# # --- CONFIGURATION ---
# INPUT_DIR = "../Data/Preprocessed_EEG" 
# OUTPUT_DIR = "../Data/Preprocessed_2s_25overlap"
# LOCS_FILE = 'channel_62_pos.locs'
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFREQ = 200
# WINDOW_SEC = 2.0
# OVERLAP_RATIO = 0.25
# STEP_SEC = WINDOW_SEC * (1 - OVERLAP_RATIO)

# CHANNEL_NAMES = [
#     'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
#     'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
#     'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
#     'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
# ]

# def apply_session_ica(mat_data):
#     keys = [k for k in mat_data.keys() if 'eeg' in k.lower()]
#     all_trials = np.concatenate([mat_data[k] for k in keys], axis=1)
    
#     info = mne.create_info(CHANNEL_NAMES, SFREQ, ch_types='eeg')
#     raw = mne.io.RawArray(all_trials, info, verbose=False)
#     montage = mne.channels.read_custom_montage(LOCS_FILE)
#     raw.set_montage(montage, on_missing='ignore')
    
#     # 1-99Hz Filter for ICLabel stability
#     raw_ica = raw.copy().filter(l_freq=1.0, h_freq=99.0, verbose=False)
#     raw_ica.set_eeg_reference('average', projection=False, verbose=False)
    
#     ica = ICA(n_components=15, max_iter='auto', random_state=97, method='infomax')
#     ica.fit(raw_ica, verbose=False)
    
#     labels = label_components(raw_ica, ica, method='iclabel')['labels']
#     ica.exclude = [i for i, label in enumerate(labels) if label in ['eye', 'muscle', 'heart']]
    
#     return ica

# def segment_voltage(data):
#     win_samples = int(WINDOW_SEC * SFREQ)
#     step_samples = int(STEP_SEC * SFREQ)
#     n_windows = (data.shape[1] - win_samples) // step_samples + 1
    
#     segments = []
#     for i in range(n_windows):
#         start = i * step_samples
#         end = start + win_samples
#         segments.append(data[:, start:end])
#     return np.array(segments) # Shape: (Windows, 62, 400)

# def build_voltage_dataset():
#     files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mat') and f != 'label.mat']
    
#     for f_name in sorted(files):
#         print(f"Cleaning & Segmenting: {f_name}")
#         mat = sio.loadmat(os.path.join(INPUT_DIR, f_name))
#         session_ica = apply_session_ica(mat)
        
#         output_dict = {}
#         keys = [k for k in mat.keys() if 'eeg' in k.lower()]
#         for k in keys:
#             # 1. ICA Apply
#             raw_trial = mne.io.RawArray(mat[k], mne.create_info(CHANNEL_NAMES, SFREQ, 'eeg'), verbose=False)
#             session_ica.apply(raw_trial, verbose=False)
            
#             # 2. Segment (2s windows)
#             output_dict[k] = segment_voltage(raw_trial.get_data())
            
#         sio.savemat(os.path.join(OUTPUT_DIR, f_name), output_dict)

# if __name__ == "__main__":
#     build_voltage_dataset()



import os
import numpy as np
import scipy.io as sio
import mne
import shutil
import psutil
from mne.preprocessing import ICA
from mne_icalabel import label_components
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import RobustScaler
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. GLOBAL CONFIGURATION ---
INPUT_DIR = "../Data/Preprocessed_EEG"
VOLTAGE_OUTPUT = "../Data/Preprocessed_2s_25overlap"
FEATURE_OUTPUT = "../Data/ExtractedFeatures_2s_25overlap"

os.makedirs(VOLTAGE_OUTPUT, exist_ok=True)
os.makedirs(FEATURE_OUTPUT, exist_ok=True)

# Parameters
SFREQ = 200
WINDOW_SEC = 2.0
OVERLAP_RATIO = 0.25
STEP_SEC = WINDOW_SEC * (1 - OVERLAP_RATIO)
BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 75)}

CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

# CHECKLIST #13: Define Global Montage and Info ONCE
GLOBAL_MONTAGE = mne.channels.read_custom_montage('channel_62_pos.locs')
GLOBAL_INFO = mne.create_info(CHANNEL_NAMES, SFREQ, ch_types='eeg')
GLOBAL_INFO.set_montage(GLOBAL_MONTAGE)

# Indices for FAS
IDX_F3, IDX_F4, IDX_ALPHA = CHANNEL_NAMES.index('F3'), CHANNEL_NAMES.index('F4'), 2

# --- 2. CORE PHYSIOLOGICAL METHODS ---

def log_mem(f_name, stage):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"[{f_name}] Stage: {stage} | Worker RAM: {mem:.2f} MB")

def get_session_ica(mat_data, f_name):
    """ Fits ICA once per 45-trial file for speed and stability """
    log_mem(f_name, "Fitting ICA")
    keys = [k for k in mat_data.keys() if 'eeg' in k.lower()]
    all_voltage = np.concatenate([mat_data[k] for k in keys], axis=1)
    
    raw = mne.io.RawArray(all_voltage, GLOBAL_INFO, verbose=False)
    # CHECKLIST #2: ICA Cleaning Logic
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=99.0, verbose=False)
    raw_ica.set_eeg_reference('average', projection=False, verbose=False)
    
    ica = ICA(n_components=15, max_iter='auto', random_state=97, method='infomax')
    ica.fit(raw_ica, verbose=False)
    
    labels = label_components(raw_ica, ica, method='iclabel')['labels']
    ica.exclude = [i for i, label in enumerate(labels) if label in ['eye', 'muscle', 'heart']]
    return ica

def apply_lds_smoothing(features):
    """ CHECKLIST #7: LDS Smoothing (Kalman Filter) """
    n_win, n_ch, n_feat = features.shape
    smoothed = np.zeros_like(features)
    
    # Pre-define constants 
    F = np.array([[1.]])
    H = np.array([[1.]])
    Q = 0.001
    R = 0.01

    for c in range(n_ch):
        for f in range(n_feat):
            data = features[:, c, f]
            
            # --- FIX STARTS HERE ---
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([[data[0]]]) # Initial State
            kf.F = F.copy()              # State Transition
            kf.H = H.copy()              # Measurement Function
            kf.P *= 10.                  # Initial Covariance
            kf.R = R                     # Measurement Noise
            kf.Q = Q                     # Process Noise
            # --- FIX ENDS HERE ---
            
            res = []
            for z in data:
                kf.predict()
                kf.update(z)
                res.append(kf.x[0,0])
            smoothed[:, c, f] = np.array(res)
    return smoothed


# --- 3. PROCESSING PIPELINE ---

def process_and_save_session(f_name):
    mat = sio.loadmat(os.path.join(INPUT_DIR, f_name))
    print(f"Processing File: {f_name}")
    keys = [k for k in mat.keys() if 'eeg' in k.lower()]
    
    # A. Get Session ICA
    session_ica = get_session_ica(mat, f_name)
    
    voltage_dict = {}
    feature_dict = {}
    scaler = RobustScaler()

    for k in keys:
        # B. Baseline Correct & Clean Trial
        # Checklist #5: First 5s as rest baseline
        rest = np.mean(mat[k][:, :int(5*SFREQ)], axis=1, keepdims=True)
        movie = mat[k][:, int(5*SFREQ):] - rest
        
        raw_trial = mne.io.RawArray(movie, GLOBAL_INFO, verbose=False)
        session_ica.apply(raw_trial, verbose=False)
        cleaned_voltage = raw_trial.get_data()

        # C. Windowing (2s, 0.5s overlap)
        win_samples = int(WINDOW_SEC * SFREQ)
        step_samples = int(STEP_SEC * SFREQ)
        n_windows = (cleaned_voltage.shape[1] - win_samples) // step_samples + 1
        
        trial_segments = []
        for i in range(n_windows):
            start = i * step_samples
            end = start + win_samples
            trial_segments.append(cleaned_voltage[:, start:end])
        
        # Save to Voltage Dict (Windows, 62, 400)
        voltage_dict[k] = np.array(trial_segments)

        # D. FEATURE EXTRACTION (DE + Var + FAS)
        de_var_list = []
        for b_name, (low, high) in BANDS.items():
            # Band-specific filtering
            filt_volts = raw_trial.copy().filter(low, high, verbose=False).get_data()
            # Segment the filtered band
            seg_band = []
            for i in range(n_windows):
                seg_band.append(filt_volts[:, i*step_samples : i*step_samples+win_samples])
            seg_band = np.array(seg_band) # (Windows, 62, 400)
            
            variance = np.var(seg_band, axis=-1)
            de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-6))
            de_var_list.append(de)
            de_var_list.append(variance)
        
        # Checklist #4: Stack features (Windows, 62, 10)
        feats = np.stack(de_var_list, axis=-1)
        
        # Checklist #7 & #8: Smoothing
        feats = apply_lds_smoothing(feats)
        # Moving Average (window=5)
        feats = np.array([np.mean(feats[max(0, i-2):i+3], axis=0) for i in range(n_windows)])

        # Checklist #6: FAS (11th feature)
        fas = feats[:, IDX_F4, 2] - feats[:, IDX_F3, 2] # Index 2 is Alpha
        fas_col = np.tile(fas[:, np.newaxis, np.newaxis], (1, 62, 1))
        final_feats = np.concatenate([feats, fas_col], axis=2)

        # Checklist #12: Robust Scale per trial
        f_shape = final_feats.shape
        final_feats = scaler.fit_transform(final_feats.reshape(-1, 11)).reshape(f_shape)

        # Save to Feature Dict with SEED keys
        feature_dict[k.replace('eeg', 'de_LDS')] = final_feats

    # E. Save .mat files
    log_mem(f_name, "Saving Outputs")
    sio.savemat(os.path.join(VOLTAGE_OUTPUT, f_name), voltage_dict)
    sio.savemat(os.path.join(FEATURE_OUTPUT, f_name), feature_dict)

def main():
    files = sorted([f for f in os.listdir(INPUT_DIR)[15:] if f.endswith('.mat') and f != 'label.mat'])
    
    print(f"Starting Parallel Build with 8 Workers...")
    # n_jobs=8 is optimal for 16GB RAM
    results = Parallel(n_jobs=8)(
        delayed(process_and_save_session)(f) for f in tqdm(files, desc="Total Progress")
    )
    
    # Copy labels
    shutil.copy(os.path.join(INPUT_DIR, "label.mat"), os.path.join(VOLTAGE_OUTPUT, "label.mat"))
    shutil.copy(os.path.join(INPUT_DIR, "label.mat"), os.path.join(FEATURE_OUTPUT, "label.mat"))
    print("\nSUCCESS: All data generated in 45-file structure.")

if __name__ == "__main__":
    main()