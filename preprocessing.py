import numpy as np
from scipy.signal import butter, filtfilt


# --- 1. Define the DE Feature Extraction Functions ---

def get_de_features(segment, fs):
    """
    Extracts Differential Entropy (DE) features from a 1-second EEG segment.
    
    Input:
    - segment: A (128, 14) array (1s segment)
    - fs: Sampling frequency (128 Hz)
    
    Output:
    - de_features: A (14, 5) array (14 channels, 5 DE features)
    """
    
    # Define EEG bands
    BANDS = {
        'Delta': (1, 4),   #
        'Theta': (4, 8),   #
        'Alpha': (8, 13),  #
        'Beta': (13, 30), #
        'Gamma': (30, 45)  #
    }
    
    nyquist = 0.5 * fs
    de_features = np.zeros((14, 5)) # 14 channels, 5 bands
    
    for i, (band_name, (low, high)) in enumerate(BANDS.items()):
        # Get filter coefficients
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        for ch in range(14):
            # 1. Filter the channel's 1s segment for the current band
            channel_signal = segment[:, ch]
            filtered_band = filtfilt(b, a, channel_signal)
            
            # 2. Compute variance
            band_variance = np.var(filtered_band)
            
            # 3. Compute DE
            # Formula from paper: 0.5 * log(2 * pi * e * variance)
            de = 0.5 * np.log(2 * np.pi * np.e * band_variance)
            
            de_features[ch, i] = de
            
    return de_features

def z_score_normalize(sample):
    """Applies Z-score normalization to a single (14, 5) sample."""
    # This matches the paper: "normalize... for each sample separately"
    mean = np.mean(sample)
    std = np.std(sample)
    
    if std == 0: # Avoid division by zero
        return sample
        
    return (sample - mean) / std

def baseline_correct(stimuli_signal, baseline_signal):
    """Applies baseline correction to a raw trial."""
    baseline_mean = np.mean(baseline_signal, axis=0)
    return stimuli_signal - baseline_mean

# --- 2. Define the New Main Preprocessing Function ---

def extract_features_and_labels(stimuli_list, baseline_list, label_list, fs):
    """
    Extracts DE features and labels for all trials, following the paper's method.
    """
    
    X_all, y_all = [], []
    
    window_size = int(fs) # 1 second * 128 Hz = 128 samples
    num_seconds = 60      # Use last 60 seconds
    
    for stimuli, baseline, label_score in zip(stimuli_list, baseline_list, label_list):
        
        # 1. Apply baseline correction
        corrected_signal = baseline_correct(stimuli, baseline)
        
        # 2. Truncate to last 60 seconds
        trunc_samples = num_seconds * window_size # 60 * 128 = 7680
        
        if corrected_signal.shape[0] >= trunc_samples:
            truncated_signal = corrected_signal[-trunc_samples:, :]
        else:
            # Pad with zeros at the beginning
            pad_width = trunc_samples - corrected_signal.shape[0]
            truncated_signal = np.pad(corrected_signal, ((pad_width, 0), (0, 0)), 'constant')
        
        # 3. Binarize the label (Low <= 3, High > 3)
        label = 1 if label_score > 3 else 0
        
        # 4. Create non-overlapping 1s segments and extract DE features
        for i in range(num_seconds): # 60 segments
            start = i * window_size
            end = start + window_size
            
            segment = truncated_signal[start:end, :] # Shape: (128, 14)
            
            # 5. Extract DE features for the segment
            de_matrix = get_de_features(segment, fs) # Shape: (14, 5)
            
            # 6. Normalize the (14, 5) sample
            normalized_de_matrix = z_score_normalize(de_matrix)
            
            X_all.append(normalized_de_matrix)
            y_all.append(label)
            
    return np.array(X_all), np.array(y_all)