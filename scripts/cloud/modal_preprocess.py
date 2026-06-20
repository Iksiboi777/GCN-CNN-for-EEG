import modal
import os


# 1. DEFINE THE ENVIRONMENT
# This builds your Docker-like environment automatically
image = (
    modal.Image.debian_slim(python_version="3.11.5")
    .pip_install("torch", "torch_geometric", "numpy", "scikit-learn", "scipy", "matplotlib", 
                 "mne","mne_icalabel", "seaborn", "filterpy")
    .workdir("/data")
)

app = modal.App("eeg-preprocess")
volume = modal.Volume.from_name("eeg-data-volume")

@app.function(image=image,volumes={"/data": volume})
def prepare_data():
    import sys
    import os
    sys.path.append("/data") # Ensure we can import from the mounted volume
    os.chdir("/data") # Change working directory to the mounted volume

    from eeg_gnn.data.features import load_de_data
    import torch

    # Run your heavy SS & SS normalization ONCE
    data_folder = "Data/ExtractedFeatures_4s"
    label_file = os.path.join(data_folder, "label.mat")
    X, y, subjects, sessions, _ = load_de_data(data_folder, label_file)
    
    # Save as a single, high-speed binary file
    processed_data = {
        "X": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long),
        "subjects": torch.tensor(subjects, dtype=torch.long)
    }
    torch.save(processed_data, "processed_seed_4s.pt")
    print("✅ Pre-processed data saved to Volume.")