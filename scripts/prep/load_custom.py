import numpy as np
import joblib
import os

OUTPUT_DIR = "../Data/Custom_2s_25overlap_FAST2"

# 1. Load the shape
shape = joblib.load(os.path.join(OUTPUT_DIR, "X_shape.pkl"))

# 2. Load the data using memmap
X = np.memmap(
    os.path.join(OUTPUT_DIR, "X_custom.dat"),
    dtype='float32',
    mode='r',
    shape=shape  # Uses the shape we just loaded
)

# 3. Load labels and subjects
y = np.load(os.path.join(OUTPUT_DIR, "y_labels.npy"))
subjects = np.load(os.path.join(OUTPUT_DIR, "subject_ids.npy"))

print(f"Loaded X with shape: {X.shape}")
print(f"Loaded y with shape: {y.shape}")