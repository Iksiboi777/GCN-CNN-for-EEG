import numpy as np
import mne
from sklearn.preprocessing import RobustScaler

class SmartPreprocessor:
    def __init__(self, channel_names, sfreq=200):
        self.channel_names = channel_names
        self.sfreq = sfreq
        self.scaler = RobustScaler()
        
        # 1. Montage & Info setup (Crucial for Interpolation)
        self.info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        mapping = {'CB1': 'PO9', 'CB2': 'PO10'}
        rename_dict = {k: v for k, v in mapping.items() if k in channel_names}
        if rename_dict:
            mne.rename_channels(self.info, rename_dict)
        
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            self.info.set_montage(montage)
        except Exception as e:
            print(f"Montage Warning: {e}")

    def detect_bad_channels(self, X):
        """
        X: (Samples, 62, 10)
        Detects 'Sinkholes' (Dead) and 'Screaming' (High Noise) channels 
        by analyzing feature energy across samples.
        """
        # Calculate global statistics per node across all features/samples
        node_variance = np.var(X, axis=(0, 2)) # Shape: (62,)
        node_mean = np.mean(np.abs(X), axis=(0, 2)) # Shape: (62,)
        
        bad_channels = []
        
        # 1. SINKHOLE DETECTION (Dead Channels)
        # If mean energy is < 10% of the global average
        dead_threshold = np.mean(node_mean) * 0.1
        for i, m in enumerate(node_mean):
            if m < dead_threshold:
                bad_channels.append(self.channel_names[i])

        # 2. SCREAMING DETECTION (Noise/Artifacts)
        # If variance is > 5x the global median variance
        noise_threshold = np.median(node_variance) * 5.0
        for i, v in enumerate(node_variance):
            if v > noise_threshold:
                if self.channel_names[i] not in bad_channels:
                    bad_channels.append(self.channel_names[i])
        
        return bad_channels

    def interpolate_bad_nodes(self, X, bad_channels):
        """
        X: (Samples, 62, 10)
        Uses MNE's spherical spline interpolation to fix the bad nodes 
        in the feature space.
        """
        if not bad_channels:
            return X
        
        n_samples, n_nodes, n_features = X.shape
        X_interpolated = np.zeros_like(X)
        
        # We must interpolate each of the 10 features independently
        for f in range(n_features):
            # MNE expects (Samples, Channels, Time)
            # We treat our 'Samples' as trials and 'Nodes' as channels
            # We'll do this trial-by-trial for precision
            feature_slice = X[:, :, f] # (Samples, 62)
            
            # Create MNE Evoked object to use the interpolation engine
            evoked = mne.EvokedArray(feature_slice.T, self.info, verbose=False)
            evoked.info['bads'] = bad_channels
            evoked.interpolate_bads(reset_bads=True, mode='accurate')
            
            X_interpolated[:, :, f] = evoked.data.T
            
        return X_interpolated

    def process_subject(self, X):
        """
        The Full Pipeline: Detect -> Interpolate -> Scale
        X: (Samples, 62, 10)
        """
        # 1. Detect
        bads = self.detect_bad_channels(X)
        if bads:
            print(f"  [SmartClean] Found {len(bads)} bad channels: {bads}")
        
        # 2. Interpolate (Fix the 'Stone' subjects like Sub 12)
        X_clean = self.interpolate_bad_nodes(X, bads)
        
        # 3. Robust Scaling (Feature-wise)
        n_samples, n_nodes, n_features = X_clean.shape
        X_flat = X_clean.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_flat)
        
        return X_scaled.reshape(n_samples, n_nodes, n_features)

# Example Usage Helper
def get_standard_channel_names():
    # Based on the SEED dataset standard 62-channel layout
    # CORRECTED NAMES TO MATCH MNE STANDARD (Mixed Case)
    return [
        'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
    ]
