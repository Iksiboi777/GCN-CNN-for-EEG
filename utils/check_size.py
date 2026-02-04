import scipy.io as sio
import os
import numpy as np

# --- CONFIGURATION ---
# Change this filename to one that exists in your new folder
TARGET_FILE = "../Data/ExtractedFeatures_1s/1_20131030.mat" 
# ---------------------

def inspect_mat(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # 1. Check File Size
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n📂 FILE: {os.path.basename(file_path)}")
    print(f"💾 DISK SIZE: {size_mb:.2f} MB")
    print("=" * 50)

    # 2. Load and Check Structure
    try:
        mat = sio.loadmat(file_path)
        
        # Filter (ignore header/version metadata)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        # Sort by trial ID usually inside strings
        try:
            keys.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        except:
            keys.sort()

        print(f"{'KEY':<12} | {'SHAPE (Win, Ch, Time)':<25} | {'DTYPE':<10}")
        print("-" * 50)

        total_windows = 0
        
        for key in keys:
            data = mat[key]
            shape = data.shape
            print(f"{key:<12} | {str(shape):<25} | {data.dtype}")
            
            # # Sanity Check
            # if len(shape) == 3:
            #     total_windows += shape[0]
            #     if shape[2] != 400:
            #         print(f"  ⚠️ WARNING: Time dimension is {shape[2]}, expected 400!")
            #     if shape[1] != 62:
            #         print(f"  ⚠️ WARNING: Channel dimension is {shape[1]}, expected 62!")

        print("=" * 50)
        print(f"✅ Total Windows in Session: {total_windows}")

    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    inspect_mat(TARGET_FILE)