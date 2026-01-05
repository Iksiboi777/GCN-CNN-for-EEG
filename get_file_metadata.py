import os
import datetime
import platform

def get_file_creation_time(path):
    try:
        # On Windows, os.path.getctime gets creation time. 
        # On Unix, it gets the time of the last metadata change.
        # Since you are on Windows, this works for creation time.
        timestamp = os.path.getctime(path)
        
        # Convert to readable format
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        return dt_object
    except FileNotFoundError:
        return None

# Define the path based on your workspace
file_path = r"Errors\GCN_DE_1s\Attempt_16_session_holdout\classification_report.txt"

# Check if file exists and get time
if os.path.exists(file_path):
    creation_time = get_file_creation_time(file_path)
    print(f"File: {file_path}")
    print(f"Created on: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Also check modification time just in case it was moved/copied
    mod_time = os.path.getmtime(file_path)
    dt_mod = datetime.datetime.fromtimestamp(mod_time)
    print(f"Last modified on: {dt_mod.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print(f"The file {file_path} does not exist.")