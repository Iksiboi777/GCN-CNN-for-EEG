import mne
import scipy.io as sio
import numpy as np
import os
from mne.preprocessing import ICA
import mne_icalabel

# --- Configuration ---
input_folder = "Preprocessed_EEG"
output_folder = "Cleaned_EEG_ICA"
os.makedirs(output_folder, exist_ok=True)

# Define Metadata
sfreq = 200
ch_names = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
montage = mne.channels.read_custom_montage('channel_62_pos.locs')

# --- Processing Loop ---
# Get all .mat files
files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

for file_name in files:
    print(f"Processing file: {file_name}")
    file_path = os.path.join(input_folder, file_name)
    
    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        continue
        
    cleaned_data_dict = {}
    
    # Copy non-EEG keys (like labels) to the new dictionary
    for key in mat_data:
        if not key.startswith('__') and not key.startswith('eeg_') and not key.startswith('djc_eeg'):
             cleaned_data_dict[key] = mat_data[key]

    # Iterate over trials (eeg_1 ... eeg_15)
    # Note: Keys might be 'eeg_1' or 'djc_eeg1' depending on the file version. 
    # We iterate all keys and check if they look like EEG data.
    for key in mat_data.keys():
        if key.startswith('__'): continue
        
        # Check if this key contains EEG data (usually shape (62, N) or (N, 62))
        data = mat_data[key]
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            continue
            
        # Heuristic: EEG data usually has 62 channels
        if data.shape[0] == 62:
            pass # Correct orientation
        elif data.shape[1] == 62:
            data = data.T # Transpose
        else:
            continue # Not an EEG data array
            
        print(f"  Cleaning trial: {key}")
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        raw.set_montage(montage)
        
        # Filter for ICA (1Hz highpass)
        # Note: We filter a COPY for ICA calculation, but apply cleaning to the ORIGINAL (or 0.1Hz filtered) data
        # to preserve low frequencies if needed. 
        raw_ica = raw.copy().filter(1.0, 40.0, verbose=False)
        
        # Run ICA
        ica = ICA(n_components=15, max_iter='auto', random_state=97, verbose=False)
        ica.fit(raw_ica)
        
        # Label and Exclude
        ic_labels = mne_icalabel.label_components(raw_ica, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        
        # Apply to original data
        ica.apply(raw, exclude=exclude_idx, verbose=False)
        
        # Store cleaned data back to dictionary
        cleaned_data_dict[key] = raw.get_data()
        
    # Save the new .mat file
    save_path = os.path.join(output_folder, file_name)
    sio.savemat(save_path, cleaned_data_dict)
    print(f"Saved cleaned file: {save_path}")

print("All files processed.")