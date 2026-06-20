# import os
# import numpy as np
# import scipy.io as sio
# from scipy.signal import butter, lfilter
# from filterpy.kalman import KalmanFilter

# # --- CONFIGURATION ---
# # IMPORTANT: Source from the CLEANED Raw segments we just made, 
# # OR start from scratch? 
# # To ensure consistency, let's start from Preprocessed_EEG again so this script is standalone.
# RAW_DATA_DIR = "../Data/Preprocessed_EEG"
# OUTPUT_DIR = "../Data/ExtractedFeatures_2s_25overlap"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFREQ = 200 
# WINDOW_SEC = 2.0
# OVERLAP_RATIO = 0.25 
# STEP_SEC = WINDOW_SEC * (1 - OVERLAP_RATIO) 
# BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}

# # Based on 62-channel layout
# CHANNEL_NAMES = [
#     'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
#     'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
#     'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
#     'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
# ]
# IDX_F3, IDX_F4, IDX_ALPHA = CHANNEL_NAMES.index('F3'), CHANNEL_NAMES.index('F4'), 2

# # --- FILTERING & SMOOTHING ---
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return lfilter(b, a, data, axis=-1)

# def apply_lds(sequence):
#     """ Linear Dynamic System Smoothing (Kalman) """
#     if sequence.ndim != 3: return sequence
#     n_win, n_ch, n_feat = sequence.shape
#     smoothed = np.zeros_like(sequence)
#     for c in range(n_ch):
#         for f in range(n_feat):
#             data = sequence[:, c, f]
#             kf = KalmanFilter(dim_x=1, dim_z=1)
#             kf.x, kf.F, kf.H, kf.P, kf.R, kf.Q = np.array([[data[0]]]), np.array([[1.]]), np.array([[1.]]), 10., 0.01, 0.001
#             res = []
#             for z in data:
#                 kf.predict()
#                 kf.update(z)
#                 res.append(kf.x[0,0])
#             smoothed[:, c, f] = np.array(res)
#     return smoothed

# def apply_moving_avg(data, window_size=5):
#     """ Simple Moving Average Smoothing """
#     if data.shape[0] < window_size: return data
#     ret = np.cumsum(data, axis=0)
#     ret[window_size:] = ret[window_size:] - ret[:-window_size]
#     return ret[window_size - 1:] / window_size

# # --- PROCESSING SINGLE TRIAL ---
# def process_trial_to_features(data):
#     # data: (62, Time)
    
#     # 1. Filter Bands (Entire Trial)
#     band_signals = []
#     for (low, high) in BANDS.values():
#         band_signals.append(butter_bandpass_filter(data, low, high, SFREQ))
#     band_signals = np.stack(band_signals, axis=0) # (5, 62, Time)
    
#     # 2. Windowing
#     win_samples = int(WINDOW_SEC * SFREQ)
#     step_samples = int(STEP_SEC * SFREQ)
#     n_windows = (data.shape[1] - win_samples) // step_samples + 1
    
#     features_list = []
    
#     for i in range(n_windows):
#         start = i * step_samples
#         end = start + win_samples
#         win = band_signals[:, :, start:end] 
        
#         # 3. Extract Features: DE + Variance
#         variance = np.var(win, axis=-1) # (5, 62)
#         de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-6))
        
#         # Combine: [DE_delta, ..., DE_gamma, Var_delta, ..., Var_gamma]
#         # Shape: (62, 10). Transpose needed? 
#         # Usually we want (62, 10) or (10, 62).
#         # Standard format in your previous datasets seems to be (62, Features) or (Features, 62).
#         # Let's clean up: 
#         # current variance shape: (5 bands, 62 chans). 
#         # Stack: (10, 62) -> Transpose to (62, 10)
        
#         feat_vec = np.concatenate([de, variance], axis=0).T # (62, 10)
#         features_list.append(feat_vec)
        
#     if not features_list: return None
    
#     # Stack windows: (N_Windows, 62, 10)
#     trial_feats = np.stack(features_list)
    
#     # 4. Smoothing (LDS + Moving Avg)
#     trial_feats = apply_lds(trial_feats)
#     # trial_feats = apply_moving_avg(trial_feats) # Optional, can blur distinct emotions too much if windows are large
    
#     # 5. FAS Calculation (Feature 11)
#     # Alpha band is index 2 in our BANDS list.
#     # DE_Alpha is index 2.
#     fas = trial_feats[:, IDX_F4, 2] - trial_feats[:, IDX_F3, 2] # (N_Windows,)
#     fas_col = np.tile(fas[:, np.newaxis, np.newaxis], (1, 62, 1)) # (N, 62, 1)
    
#     final_feats = np.concatenate([trial_feats, fas_col], axis=2) # (N, 62, 11)
    
#     # 6. Final Reshape compatibility
#     # The standard loader expects: (62, N_samples * n_features)?
#     # actually, looking at your old loader:
#     # X = np.transpose(trial_data, (1, 0, 2)) -> (62, N, 5) usually implies (Chan, Time, Band)
    
#     # However, to be perfectly compatible with 'load_de_data' which likely iterates keys and stacks them:
#     # We will save as (Channels, TotalFeatureVectorLength) to mimic continuous recording?
#     # OR save as (N_Windows, 62, 11) which is clearer.
#     # Since you wrote the loader, you know best.
#     # But usually .mat files in SEED are (62, TimePoints).
#     # Since these are features, let's permute to: (62, 11 * N_Windows) or similar?
#     # NO. Let's stick to the most logical "Tensor" format: (11, 62, N_Windows) or (62, 11, N_Windows).
    
#     # To be safe and easy to reshape in python: (N_Windows, 62, 11)
#     return final_feats

# def main():
#     # Copy labels
#     label_path = os.path.join(RAW_DATA_DIR, "label.mat")
#     if os.path.exists(label_path):
#         import shutil
#         shutil.copy(label_path, os.path.join(OUTPUT_DIR, "label.mat"))
        
#     files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.mat') and f != 'label.mat'])

#     for f_name in files:
#         print(f"Processing Features: {f_name}...")
#         mat = sio.loadmat(os.path.join(RAW_DATA_DIR, f_name))
#         output_mat = {}
        
#         keys = [k for k in mat.keys() if 'eeg' in k.lower()]
#         for key in keys:
#             # We assume the input is clean enough for Feature Extraction 
#             # (or add ICA call here if not pre-cleaned)
#             data = mat[key]
            
#             # Simple Baseline correction?
#             # data = data - np.mean(data, axis=1, keepdims=True)
            
#             feats = process_trial_to_features(data)
            
#             if feats is not None:
#                 # Shape: (N, 62, 11)
#                 # We save it back to the .mat file under the same key
#                 output_mat[key] = feats
        
#         sio.savemat(os.path.join(OUTPUT_DIR, f_name), output_mat)

# if __name__ == "__main__":
#     main()


import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter
from filterpy.kalman import KalmanFilter

# --- CONFIGURATION ---
# INPUT: The output of  (Already ICA cleaned & Segmented)
INPUT_DIR = "../Data/Preprocessed_2s_25overlap" 
OUTPUT_DIR = "../Data/ExtractedFeatures_2s_25overlap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SFREQ = 200 
# Note: Windows are already cut to 2s (400 samples) in the input files.
BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}

# Based on 62-channel layout
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]
IDX_F3, IDX_F4 = CHANNEL_NAMES.index('F3'), CHANNEL_NAMES.index('F4')
IDX_ALPHA = 2 # 0:Delta, 1:Theta, 2:Alpha, 3:Beta, 4:Gamma

# --- MATH UTILS ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    # data shape: (N_windows, 62, 400) or similar
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)

def apply_lds(sequence):
    """ Linear Dynamic System Smoothing (Kalman) on the time-sequence of features """
    # sequence shape expected: (N_windows, 62, 11)
    if sequence.ndim != 3: return sequence
    n_win, n_ch, n_feat = sequence.shape
    smoothed = np.zeros_like(sequence)
    
    # We smooth "across windows" (axis 0)
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

def process_segmented_trial(trial_segments):
    """
    Input: (N_windows, 62, 400) - Raw EEG segments
    Output: (N_windows, 62, 11) - Features
    """
    n_wins, n_ch, n_samples = trial_segments.shape
    
    # 1. Filter Bands for ALL segments at once
    # band_signals shape: (5 bands, N_win, 62, 400)
    band_signals = []
    for (low, high) in BANDS.values():
        filtered = butter_bandpass_filter(trial_segments, low, high, SFREQ)
        band_signals.append(filtered)
    band_signals = np.stack(band_signals, axis=0) 
    
    # 2. Calculate Features in Vectorized manner
    # Variance across time (axis 3, the 400 samples)
    # Shape becomes: (5 bands, N_win, 62)
    variance = np.var(band_signals, axis=-1) 
    
    # DE Formula
    de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-6))
    
    # 3. Reshape/Stack to (N_win, 62, 10)
    # Current: (5, N, 62). We want N first, then 62, then 10 features.
    
    # Move axis: (N, 62, 5)
    variance = np.transpose(variance, (1, 2, 0)) 
    de = np.transpose(de, (1, 2, 0))
    
    # Concatenate features: First 5 are DE, next 5 are Var
    combined = np.concatenate([de, variance], axis=2) # (N, 62, 10)
    
    # 4. LDS Smoothing
    # Crucial: Apply LDS *before* FAS, so FAS is based on smoothed alpha
    smoothed_feats = apply_lds(combined)
    
    # 5. FAS Calculation (Feature 11)
    # Indices: DE Alpha is index 2
    # FAS = F4_alpha - F3_alpha
    fas = smoothed_feats[:, IDX_F4, IDX_ALPHA] - smoothed_feats[:, IDX_F3, IDX_ALPHA] # (N_win,)
    
    # Expand to match dimensions: (N, 62, 1) - Same value repeated for all channels for that window
    fas_col = np.tile(fas[:, np.newaxis, np.newaxis], (1, 62, 1))
    
    # 6. Final Concatenation
    final_feats = np.concatenate([smoothed_feats, fas_col], axis=2) # (N, 62, 11)
    
    return final_feats

def main():
    # Copy label.mat
    try:
        label_path = os.path.join(INPUT_DIR, "label.mat")
        if os.path.exists(label_path):
            import shutil
            shutil.copy(label_path, os.path.join(OUTPUT_DIR, "label.mat"))
    except Exception as e:
        print(f"Note: label.mat processing skipped or failed ({e})")

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.mat') and f != 'label.mat'])
    
    print(f"Found {len(files)} files in {INPUT_DIR}")

    for f_name in files:
        print(f"extracting features for: {f_name}...")
        mat = sio.loadmat(os.path.join(INPUT_DIR, f_name))
        output_mat = {}
        
        # Only process EEG keys
        keys = [k for k in mat.keys() if 'eeg' in k.lower()]
        
        for key in keys:
            # Segmented Data: (N_windows, 62, 400)
            data_segments = mat[key]
            
            # Guard against empty trials or wrong shapes
            if data_segments.ndim == 3 and data_segments.shape[2] == 400:
                feats = process_segmented_trial(data_segments)
                output_mat[key] = feats # (N, 62, 11)
            else:
                print(f"Skipping key {key} due to shape {data_segments.shape}")
        
        # Save
        sio.savemat(os.path.join(OUTPUT_DIR, f_name), output_mat)
        
    print("Feature Extraction Complete.")

if __name__ == "__main__":
    main()