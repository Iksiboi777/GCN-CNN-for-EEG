import os
import sys
from unittest.mock import patch
import traceback

# Import the main logic from the per-trial script
from results_analysis.analyze_per_trial import main as analyze_main

def parse_run_info(root_path):
    """
    Infers mode, model, window, and run_id from the directory path.
    Handles:
    1. Errors/Diagnostic/{RunName}
    2. Errors/{Model}_{Window}/Attempt_{ID} (Your current structure)
    3. Errors/{Mode}/{Model}_{Window}/Attempt_{ID} (Future proof structure)
    """
    parts = root_path.split(os.sep)
    
    # Find where 'Errors' is in the path
    try:
        err_idx = parts.index("Errors")
    except ValueError:
        return None
        
    # We need at least one folder after Errors
    if len(parts) <= err_idx + 1:
        return None
        
    first_folder = parts[err_idx + 1] # e.g., 'Diagnostic', 'GCN_DE_1s', or 'sub_dep'
    
    # --- CASE 1: Diagnostic Run ---
    if first_folder.lower() == 'diagnostic':
        mode = 'diagnostic'
        run_id = parts[-1] # The folder name is the ID
        
        # Guess model/window from name
        model = 'DGCNN' if 'DGCNN' in run_id else 'GCN'
        window = '4s' if '4s' in run_id else '1s'
        return mode, model, window, run_id

    # --- CASE 2: Standard Run (Directly in Errors) ---
    # Structure: Errors/GCN_DE_1s/Attempt_27
    elif 'GCN' in first_folder: # Matches GCN_DE_1s or DGCNN_DE_4s
        mode = 'sub_dep' # Default assumption since you haven't done sub_indep yet
        
        # Parse Model and Window from folder name
        if 'DGCNN' in first_folder: model = 'DGCNN'
        else: model = 'GCN'
        
        if '4s' in first_folder: window = '4s'
        else: window = '1s'
        
        # Parse Run ID
        run_folder = parts[-1]
        if 'Attempt_' in run_folder:
            run_id = run_folder.split('_')[-1]
        else:
            return None # Not an attempt folder
            
        return mode, model, window, run_id

    # --- CASE 3: Nested Mode (Future Proofing) ---
    # Structure: Errors/sub_dep/GCN_DE_1s/Attempt_27
    elif first_folder.lower() in ['sub_dep', 'sub_indep']:
        mode = first_folder
        if len(parts) <= err_idx + 2: return None
        
        model_folder = parts[err_idx + 2]
        if 'DGCNN' in model_folder: model = 'DGCNN'
        else: model = 'GCN'
        
        if '4s' in model_folder: window = '4s'
        else: window = '1s'
        
        run_folder = parts[-1]
        if 'Attempt_' in run_folder:
            run_id = run_folder.split('_')[-1]
        else:
            return None
            
        return mode, model, window, run_id

    return None

def main():
    errors_root = "Errors"
    if not os.path.exists(errors_root):
        print(f"Directory '{errors_root}' not found. Run from project root.")
        return

    print(f"Scanning '{errors_root}' for predictions.npy...")
    
    found_runs = []
    
    for root, dirs, files in os.walk(errors_root):
        if "predictions.npy" in files:
            info = parse_run_info(root)
            if info:
                found_runs.append(info)

    print(f"Found {len(found_runs)} runs to process.")
    print("-" * 50)
    
    for mode, model, window, run_id in found_runs:
        print(f"Processing: Mode={mode}, Model={model}, Window={window}, ID={run_id}")
        
        # Simulate command line arguments for the analyze_per_trial script
        sys_args = [
            "analyze_per_trial.py",
            "--run_id", run_id,
            "--window_size", window,
            "--model_type", model,
            "--mode", mode
        ]
        
        # Patch sys.argv so argparse inside analyze_main reads our simulated args
        with patch.object(sys, 'argv', sys_args):
            try:
                analyze_main()
            except Exception as e:
                print(f"Error processing run {run_id}: {e}")
                # traceback.print# filepath: c:\Users\User\OneDrive - fer.hr\Desktop\Asus\Desktop\FER\DIPLOMSKI RAD\DATASETOVI_LIT\trial_plots_all.py
                traceback.print_exc()
            print("-" * 50)

if __name__ == "__main__":
    main()