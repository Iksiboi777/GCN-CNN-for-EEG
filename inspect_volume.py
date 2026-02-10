# import modal
# import os

# app = modal.App("volume-inspector")
# volume = modal.Volume.from_name("eeg-data-volume")

# image = modal.Image.debian_slim()

# @app.function(volumes={"/data": volume}, image=image)
# def list_files():
#     os.chdir("/data")  # Change to the mounted volume directory
#     print("🧐 INSPECTING VOLUME CONTENT...")
    
#     target_dir = "Data/ExtractedFeatures_1s"
    
#     # 1. Check if the specific folder exists
#     if not os.path.exists(target_dir):
#         print(f"❌ ERROR: Directory {target_dir} does not exist!")
#         print("Listing root /data to see what IS there:")
#         print(os.listdir("/data"))
#         return

#     # 2. Count MAT files per subject
#     files = os.listdir(target_dir)
#     print(f"✅ Found directory. Total files: {len(files)}")
    
#     subject_counts = {}
#     for f in files:
#         if f.endswith(".mat") and "_" in f:
#             try:
#                 sub_id = int(f.split("_")[0])
#                 subject_counts[sub_id] = subject_counts.get(sub_id, 0) + 1
#             except: pass
            
#     print("\n📊 FILE COUNT PER SUBJECT:")
#     found_subs = sorted(subject_counts.keys())
#     for s in found_subs:
#         print(f"   Subject {s}: {subject_counts[s]} files")
        
#     missing = [s for s in range(1, 16) if s not in found_subs]
#     if missing:
#         print(f"\n❌ MISSING SUBJECTS: {missing}")
#         print("   (This explains why your code crashes on Subject 4)")
#     else:
#         print("\n✅ All 15 subjects seem to be present.")

# @app.local_entrypoint()
# def main():
#     list_files.remote()

import modal
import os
import scipy.io as sio
import numpy as np

app = modal.App("label-inspector")
volume = modal.Volume.from_name("eeg-data-volume")

image = modal.Image.debian_slim().pip_install("scipy", "numpy")

@app.function(volumes={"/data": volume}, image=image)
def inspect_labels():
    print("🧐 INSPECTING LABEL.MAT CONTENT...")
    
    # Try finding the file
    possible_paths = [
        "/data/Data/ExtractedFeatures_1s/label.mat",
        "/data/Data/ExtractedFeatures_1s/Label.mat"
    ]
    
    label_path = None
    for p in possible_paths:
        if os.path.exists(p):
            label_path = p
            break
            
    if not label_path:
        print("❌ CRITICAL: Could not find label.mat or Label.mat")
        return

    print(f"✅ Found label file at: {label_path}")
    
    try:
        mat = sio.loadmat(label_path)
        print(f"   Keys in mat file: {mat.keys()}")
        
        # Usually labels are in 'label' or 'labels' key
        # Adjust based on your specific format, assuming 'label' based on previous scripts
        if 'label' in mat:
            labels = mat['label'][0]
            print(f"   Total Labels Found: {len(labels)}")
            # If there is no subject info in label.mat, we infer it from length
            # But usually load_de_data aligns this.
        else:
            print("   ⚠️ Key 'label' not found in .mat file")

    except Exception as e:
        print(f"❌ Error reading .mat file: {e}")

@app.local_entrypoint()
def main():
    inspect_labels.remote()