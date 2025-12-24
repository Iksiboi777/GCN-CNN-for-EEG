import numpy as np
import mne
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

class SmartPreprocessor:
    def __init__(self, channel_names, sfreq=200):
        """
        Initializes the SmartPreprocessor.
        
        Args:
            channel_names (list): List of channel names (strings).
            sfreq (int): Sampling frequency (default 200Hz, used for MNE info).
        """
        self.channel_names = channel_names
        self.sfreq = sfreq
        
        # 1. Create Info object
        self.info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        
        # 2. Handle Non-Standard Channel Names (SEED specific)
        # CB1/CB2 are not in standard_1005. We map them to the closest standard equivalents:
        # CB1 -> PO9 (Left low posterior)
        # CB2 -> PO10 (Right low posterior)
        mapping = {'CB1': 'PO9', 'CB2': 'PO10'}
        
        # Only rename if they exist in our list
        rename_dict = {k: v for k, v in mapping.items() if k in channel_names}
        if rename_dict:
            print(f"Mapping non-standard channels for montage: {rename_dict}")
            mne.rename_channels(self.info, rename_dict)
        
        # 3. Set Montage (Standard 1005 High Density)
        self.montage = mne.channels.make_standard_montage('standard_1005')
        
        # Now we can be strict. If something is still missing, we want to know.
        try:
            self.info.set_montage(self.montage, on_missing='warn')
        except ValueError as e:
            print(f"Montage Error: {e}")
            print("Proceeding, but spatial interpolation may fail for missing channels.")

        self.scaler = RobustScaler()
        self.is_fitted = False

    def detect_bad_channels(self, data, var_threshold_sigma=3.0, energy_threshold_sigma=2.0):
        """
        Automatically identifies bad channels based on Variance (Noise) and Mean Energy (Dead).
        
        Args:
            data (np.array): Shape (n_channels, n_timepoints).
            var_threshold_sigma (float): How many std devs above mean variance to flag as NOISY.
            energy_threshold_sigma (float): How many std devs below mean energy to flag as DEAD.
            
        Returns:
            list: List of bad channel names.
        """
        bad_channels = []
        
        # 1. Calculate Statistics
        # Variance across time for each channel
        ch_variances = np.var(data, axis=1)
        # Mean energy (absolute amplitude) across time
        ch_means = np.mean(np.abs(data), axis=1)
        
        # Global Statistics (Robust stats: Median and IQR are better, but Mean/Std is standard for Z-score)
        global_var_mean = np.mean(ch_variances)
        global_var_std = np.std(ch_variances)
        
        global_energy_mean = np.mean(ch_means)
        global_energy_std = np.std(ch_means)
        
        # 2. Identify "Screaming" Channels (High Variance)
        # Logic: If variance is massive compared to the rest of the head.
        # We use a higher threshold because some variance is good (alpha waves).
        noisy_indices = np.where(ch_variances > (global_var_mean + var_threshold_sigma * global_var_std))[0]
        
        # 3. Identify "Sinkholes" (Dead Channels)
        # Logic: If signal is flat compared to the rest.
        dead_indices = np.where(ch_means < (global_energy_mean - energy_threshold_sigma * global_energy_std))[0]
        
        # Combine and get names
        all_bad_indices = np.unique(np.concatenate((noisy_indices, dead_indices)))
        
        for idx in all_bad_indices:
            bad_channels.append(self.channel_names[idx])
            
        # specific check for known mechanical failures if they slip through stats
        # (Optional: You can hardcode 'Cz' here if you want to be 100% sure for Subject 7)
        
        return bad_channels

    def interpolate_bads(self, data, bad_channels):
        """
        Interpolates bad channels using Spherical Splines (MNE).
        
        Args:
            data (np.array): Shape (n_channels, n_timepoints).
            bad_channels (list): List of bad channel names.
            
        Returns:
            np.array: Cleaned data with interpolated values.
        """
        if not bad_channels:
            return data

        # Create MNE Evoked object (container for data arrays)
        # We treat the data as an "Evoked" response to use MNE's tools easily
        evoked = mne.EvokedArray(data, self.info)
        
        # Mark bads
        # Only mark bads that actually exist in the montage (to avoid interpolation errors)
        valid_bads = [ch for ch in bad_channels if ch in self.info.ch_names]
        
        if not valid_bads:
            return data

        evoked.info['bads'] = valid_bads
        
        # Interpolate
        # origin=(0, 0, 0) assumes head center. 
        try:
            evoked.interpolate_bads(reset_bads=True, method=dict(eeg='spline'), verbose=False)
            return evoked.data
        except Exception as e:
            print(f"Warning: Interpolation failed for {valid_bads}. Reason: {e}")
            return data


    def fit_scaler(self, data):
        """
        Fits the RobustScaler on training data.
        Data shape should be (n_samples, n_features) or (n_channels * n_timepoints).
        """
        # Flatten for scaling if necessary, or scale per channel.
        # Usually for EEG, we scale per channel or globally. 
        # Here we assume data is (n_samples, n_channels, n_timepoints)
        # We stack to (n_samples * n_timepoints, n_channels) to fit scaler per channel
        
        n_samples, n_channels, n_time = data.shape
        data_reshaped = np.transpose(data, (0, 2, 1)).reshape(-1, n_channels)
        
        self.scaler.fit(data_reshaped)
        self.is_fitted = True

    def apply_scaler(self, data):
        """
        Applies RobustScaler.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
            
        n_samples, n_channels, n_time = data.shape
        
        # Reshape to (N, Channels)
        data_reshaped = np.transpose(data, (0, 2, 1)).reshape(-1, n_channels)
        
        # Transform
        data_scaled = self.scaler.transform(data_reshaped)
        
        # Reshape back to (n_samples, n_channels, n_time)
        data_scaled = data_scaled.reshape(n_samples, n_time, n_channels)
        data_scaled = np.transpose(data_scaled, (0, 2, 1))
        
        return data_scaled

    def process_subject(self, data_tensor):
        """
        Full pipeline wrapper for a single subject's data tensor.
        
        Args:
            data_tensor (np.array): Shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            np.array: Cleaned and Scaled tensor.
        """
        n_trials, _, _ = data_tensor.shape
        cleaned_data = np.zeros_like(data_tensor)
        
        print(f"Starting Smart Preprocessing for {n_trials} trials...")
        
        for i in range(n_trials):
            trial_data = data_tensor[i] # (62, time)
            
            # 1. Detect Bads
            bads = self.detect_bad_channels(trial_data)
            
            # 2. Interpolate
            if bads:
                # print(f"  Trial {i}: Interpolating {len(bads)} channels: {bads}")
                cleaned_trial = self.interpolate_bads(trial_data, bads)
            else:
                cleaned_trial = trial_data
                
            cleaned_data[i] = cleaned_trial
            
        # 3. Scale (Fit on this subject if Subject Dependent, or use pre-fitted)
        # For Subject Dependent, we fit on the whole block
        self.fit_scaler(cleaned_data)
        final_data = self.apply_scaler(cleaned_data)
        
        return final_data

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

def main():
    """
    Main function to demonstrate the usage of SmartPreprocessor.
    """
    print("Initializing Smart Preprocessing Pipeline...")
    
    # 1. Setup
    channel_names = get_standard_channel_names()
    preprocessor = SmartPreprocessor(channel_names)
    
    # 2. Create Dummy Data (Simulating a subject with bad channels)
    # Shape: (Trials, Channels, Timepoints)
    n_trials = 5
    n_channels = 62
    n_time = 200 # 1 second at 200Hz
    
    print(f"Generating dummy data: {n_trials} trials, {n_channels} channels, {n_time} timepoints")
    dummy_data = np.random.normal(0, 1, (n_trials, n_channels, n_time))
    
    # Simulate a "Dead" Channel (Sinkhole) - e.g., Cz (Index 27)
    print("Simulating dead channel at index 27 (Cz)...")
    dummy_data[:, 27, :] = dummy_data[:, 27, :] * 0.01 
    
    # Simulate a "Screaming" Channel (High Variance) - e.g., F7 (Index 5)
    print("Simulating noisy channel at index 5 (F7)...")
    dummy_data[:, 5, :] = dummy_data[:, 5, :] * 20.0 
    
    # 3. Run Pipeline
    print("\n--- Running Processing ---")
    cleaned_data = preprocessor.process_subject(dummy_data)
    
    # 4. Verify Results
    print("\n--- Verification ---")
    
    # Check if variance of F7 is tamed
    orig_var_f7 = np.var(dummy_data[:, 5, :])
    clean_var_f7 = np.var(cleaned_data[:, 5, :])
    print(f"F7 Variance: Original={orig_var_f7:.2f} -> Cleaned={clean_var_f7:.2f}")
    
    # Check if mean of Cz is restored (not zero)
    orig_mean_cz = np.mean(np.abs(dummy_data[:, 27, :]))
    clean_mean_cz = np.mean(np.abs(cleaned_data[:, 27, :]))
    print(f"Cz Mean Energy: Original={orig_mean_cz:.2f} -> Cleaned={clean_mean_cz:.2f}")
    
    print("\nPipeline test completed.")

if __name__ == "__main__":
    main()