import scipy.io as sio

# Load the .mat file
mat_data = sio.loadmat('DREAMER.mat')

# This is an educated guess on the structure. 
# You will need to print(mat_data.keys()) and explore.
data = mat_data['DREAMER']['Data'][0,0] 
print("Data shape:", data.shape)
num_subjects = data['EEG'].shape[0] # e.g., 23 subjects

all_trials_data = []
all_trials_labels = []

# Loop through each subject
for i in range(num_subjects):
    # Loop through each of the 18 trials for this subject
    for j in range(data['EEG'][i,0].shape[0]): # e.g., 18 trials
        
        # 1. Extract EEG Signal (Our Node Features)
        # Shape: (samples, 14_channels)
        eeg_signal = data['EEG'][i,0][j,0] 
        all_trials_data.append(eeg_signal)

        # 2. Extract Labels (Our Graph-Level Target)
        # Shape: (3,) -> [valence, arousal, dominance]
        ratings = data['Score'][i,0][j,0] 
        all_trials_labels.append(ratings)

print(f"Loaded {len(all_trials_data)} trials in total.")
print(f"Example EEG data shape: {all_trials_data[0].shape}")
print(f"Example label: {all_trials_labels[0]}")