import modal
import os


app = modal.App("upload-eeg-code-and-resources")
volume = modal.Volume.from_name("eeg-data-volume")


files_list = ['utils/training_utils.py']

# dirs_list = ['Data/ExtractedFeatures_1s', 'Data/ExtractedFeatures_4s', 'Models', 'utils']


# --- ADD THIS FUNCTION TO YOUR SCRIPT ---
@app.local_entrypoint()
def upload_code_and_resources():
    print("📦 Starting Batch Upload to 'eeg-data-volume'...")
    
    # 1. Upload Files
    for file_name in files_list:
        try:
            with volume.batch_upload() as batch:
                print(f"  -> Uploading file: {file_name}")
                batch.put_file(file_name, f"/{file_name}")
        except FileNotFoundError:
            print(f"  ⚠️ Warning: File '{file_name}' not found locally. Skipping.")

    # with volume.batch_upload() as batch:
    #     for dir_name in dirs_list:
    #         if not os.path.exists(dir_name):
    #             print(f"  ⚠️ Warning: Directory '{dir_name}' not found. Skipping.")
    #             continue
                
    #         print(f"  -> Scanning directory: {dir_name}/")
            
    #         # Walk through the directory structure
    #         for root, dirs, files in os.walk(dir_name):
    #             for file in files:
    #                 if file.endswith('.npy'):
    #                     continue # SKIP .npy files
                    
    #                 local_path = os.path.join(root, file)
    #                 # Create remote path relative to the root (e.g. Models/var_A.py -> /Models/var_A.py)
    #                 remote_path = "/" + local_path.replace(os.sep, "/")
                    
    #                 batch.put_file(local_path, remote_path)

    print("✅ Upload Complete!")