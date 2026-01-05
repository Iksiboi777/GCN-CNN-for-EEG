# import numpy as np
# import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except Exception as e:
#     print(f"Warning: Could not set TkAgg backend: {e}")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import argparse
# import scipy.io
# import pandas as pd

# def resolve_run_path(args):
#     """Resolves the full path to the results directory based on arguments."""
#     if args.mode == 'diagnostic':
#         base_results_dir = os.path.join("Results", "Diagnostic")
#         run_name = args.run_id
        
#         if not os.path.exists(os.path.join(base_results_dir, run_name)):
#              if os.path.exists(base_results_dir):
#                  candidates = [d for d in os.listdir(base_results_dir) if args.run_id in d]
#                  if candidates:
#                      run_name = candidates[0]
#                  else:
#                      raise FileNotFoundError(f"Could not find diagnostic run '{args.run_id}' in {base_results_dir}")
#              else:
#                  raise FileNotFoundError(f"Directory {base_results_dir} does not exist.")
#     else:
#         model_name = f"{args.model_type}_DE_{args.window_size}"
#         base_results_dir = os.path.join("Results", args.mode, model_name)
        
#         run_prefix = f"Attempt_{args.run_id}"
#         found_dirs = []
#         if os.path.exists(base_results_dir):
#             found_dirs = [d for d in os.listdir(base_results_dir) 
#                           if d.startswith(run_prefix) and os.path.isdir(os.path.join(base_results_dir, d))]
        
#         if not found_dirs:
#             raise FileNotFoundError(f"No run directory found starting with '{run_prefix}' in {base_results_dir}")
#         run_name = found_dirs[0]

#     return base_results_dir, run_name

# def get_save_directory(args, run_name):
#     """Creates the directory structure for saving plots."""
#     base_plot_dir = "Trial_Plots"
    
#     if args.mode == 'diagnostic':
#         # Structure: Trial_Plots/Diagnostics/{Run_Name}
#         category_dir = "Diagnostics"
#     else:
#         # Structure: Trial_Plots/{Model}_{WindowSize}/{Run_Name}
#         category_dir = f"{args.model_type}_{args.window_size}"
        
#     full_save_dir = os.path.join(base_plot_dir, category_dir, run_name)
    
#     if not os.path.exists(full_save_dir):
#         os.makedirs(full_save_dir)
#         print(f"Created directory: {full_save_dir}")
        
#     return full_save_dir

# def load_metadata_and_preds(base_results_dir, run_name, window_size='1s', mode='sub_dep'):
#     print(f"Analyzing: {run_name}")
    
#     # 1. Load Predictions
#     errors_base = base_results_dir.replace("Results", "Errors")
#     print("Main base: ", base_results_dir)
#     pred_path = os.path.join(errors_base, run_name, "predictions.npy")
#     print(f"Loading predictions from directory: {pred_path}")
    
#     if not os.path.exists(pred_path):
#         pred_path = os.path.join(base_results_dir, run_name, "predictions.npy")
#         if not os.path.exists(pred_path):
#             raise FileNotFoundError(f"predictions.npy not found at {pred_path}")
        
#     print(f"Loading predictions from {pred_path}...")
#     pred_data = np.load(pred_path, allow_pickle=True).item()
#     y_pred = pred_data['y_pred']
#     y_true = pred_data['y_true']
    
#     unique_classes = np.unique(y_true)
#     is_binary = (len(unique_classes) == 2)
#     print(f"Detected {len(unique_classes)} classes. Mode: {'Binary' if is_binary else '3-Class'}")

#     # 2. Reconstruct Metadata
#     data_folder = f"Data/ExtractedFeatures_{window_size}"
#     label_file = os.path.join(data_folder, "label.mat")
    
#     try:
#         label_mat = scipy.io.loadmat(label_file)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Label file not found at {label_file}")

#     trial_labels = label_mat['label'][0]
#     label_map = {-1: 0, 0: 1, 1: 2}
#     mapped_labels = np.array([label_map[l] for l in trial_labels])
    
#     keep_mask = (mapped_labels != 2) if is_binary else np.ones(len(mapped_labels), dtype=bool)
    
#     meta_list = []
    
#     files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
#     subject_files = {}
#     for f in files:
#         parts = f.split('_')
#         try: subj_id = int(parts[0])
#         except: continue
#         if subj_id not in subject_files: subject_files[subj_id] = []
#         subject_files[subj_id].append(f)
        
#     test_sub_id_filter = None
#     if mode == 'sub_indep':
#         try:
#             parts = run_name.split('_sub')
#             if len(parts) > 1:
#                 test_sub_id_filter = int(parts[-1])
#                 print(f"Filtering metadata for Test Subject: {test_sub_id_filter}")
#         except: pass

#     for subj_id in sorted(subject_files.keys()):
#         if test_sub_id_filter is not None and subj_id != test_sub_id_filter:
#             continue

#         s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        
#         for sess_idx, fname in enumerate(s_files):
#             session_id = sess_idx + 1
            
#             if (mode == 'sub_dep' or mode == 'diagnostic') and session_id != 3:
#                 continue
                
#             file_path = os.path.join(data_folder, fname)
#             try: mat = scipy.io.loadmat(file_path)
#             except: continue
            
#             for trial_i in range(1, 16):
#                 trial_idx = trial_i - 1
                
#                 if is_binary and not keep_mask[trial_idx]:
#                     continue
                    
#                 key = f"de_LDS{trial_i}"
#                 if key not in mat: continue
#                 data = mat[key]
#                 num_samples = data.shape[1]
                
#                 lbl_val = mapped_labels[trial_idx]
#                 if lbl_val == 0: lbl_name = "Negative"
#                 elif lbl_val == 1: lbl_name = "Neutral"
#                 else: lbl_name = "Positive"
                
#                 for _ in range(num_samples):
#                     meta_list.append({
#                         'Subject': subj_id,
#                         'Trial': trial_i,
#                         'Label': lbl_name,
#                         'Label_Int': lbl_val
#                     })
                    
#     df = pd.DataFrame(meta_list)
    
#     if len(df) != len(y_pred):
#         print(f"WARNING: Metadata length ({len(df)}) != Prediction length ({len(y_pred)})")
#         print("Truncating to shorter length (this might misalign data if not careful!)")
#         min_len = min(len(df), len(y_pred))
#         df = df.iloc[:min_len]
#         y_pred = y_pred[:min_len]
#         y_true = y_true[:min_len]
        
#     df['Prediction'] = y_pred
#     df['Correct'] = (df['Prediction'] == y_true)
    
#     return df, run_name, is_binary

# def plot_trial_heatmap(df, run_name, save_dir):
#     heatmap_data = df.groupby(['Subject', 'Trial'])['Correct'].mean().unstack()
    
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=1)
#     plt.title(f"Accuracy per Trial (Subject vs Trial ID)\nRun: {run_name}")
#     plt.xlabel("Trial ID (Video Clip)")
#     plt.ylabel("Subject ID")
#     plt.tight_layout()
    
#     save_path = os.path.join(save_dir, "accuracy_heatmap.png")
#     plt.savefig(save_path)
#     print(f"Saved accuracy heatmap to {save_path}")
#     plt.close()

# def plot_prediction_bias(df, run_name, is_binary, save_dir):
#     heatmap_data = df.groupby(['Subject', 'Trial'])['Prediction'].mean().unstack()
    
#     plt.figure(figsize=(12, 8))
    
#     if is_binary:
#         sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", vmin=0, vmax=1)
#         plt.title(f"Prediction Bias (0=All Neg, 1=All Neu)\nRun: {run_name}")
#     else:
#         sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=2)
#         plt.title(f"Prediction Bias (0=Neg, 1=Neu, 2=Pos)\nRun: {run_name}")
        
#     plt.xlabel("Trial ID (Video Clip)")
#     plt.ylabel("Subject ID")
#     plt.tight_layout()
    
#     save_path = os.path.join(save_dir, "bias_heatmap.png")
#     plt.savefig(save_path)
#     print(f"Saved bias heatmap to {save_path}")
#     plt.close()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run_id', type=str, required=True, 
#                         help="Attempt ID (int) OR Folder Name (str) for Diagnostic runs")
#     parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'])
#     parser.add_argument('--model_type', type=str, default='DGCNN', choices=['GCN', 'DGCNN'], 
#                         help="Model type used for training (GCN or DGCNN)")
#     parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep', 'diagnostic'],
#                         help="Training mode: 'sub_dep', 'sub_indep', or 'diagnostic'")
#     args = parser.parse_args()
    
#     try:
#         base_results_dir, run_name = resolve_run_path(args)
#         df, run_name, is_binary = load_metadata_and_preds(base_results_dir, run_name, args.window_size, args.mode)
        
#         # Create Save Directory
#         save_dir = get_save_directory(args, run_name)
        
#         print("\n--- Generating Accuracy Heatmap ---")
#         plot_trial_heatmap(df, run_name, save_dir)
        
#         print("\n--- Generating Bias Heatmap ---")
#         plot_prediction_bias(df, run_name, is_binary, save_dir)
        
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()   

# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception as e:
    print(f"Warning: Could not set TkAgg backend: {e}")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import scipy.io
import pandas as pd

def resolve_run_path(args):
    """
    Resolves the correct run name and base directory.
    Prioritizes finding the run in the 'Errors' directory because that's where predictions.npy is.
    """
    # 1. Determine Base Directories
    if args.mode == 'diagnostic':
        base_results_rel = os.path.join("Results", "Diagnostic")
        base_errors_rel = os.path.join("Errors", "Diagnostic")
    else:
        model_name = f"{args.model_type}_DE_{args.window_size}"
        base_results_rel = os.path.join("Results", model_name)
        base_errors_rel = os.path.join("Errors", model_name)

    # 2. Check if the run exists in Errors (Priority)
    run_name = args.run_id
    
    # Check exact match in Errors
    if os.path.exists(os.path.join(base_errors_rel, run_name)):
        print(f"Found run in Errors: {run_name}")
        return base_results_rel, run_name

    # Check fuzzy match in Errors (e.g. if ID is passed but folder is Attempt_ID)
    if os.path.exists(base_errors_rel):
        candidates = [d for d in os.listdir(base_errors_rel) if run_name in d]
        if candidates:
            print(f"Found fuzzy match in Errors: {candidates[0]}")
            return base_results_rel, candidates[0]

    # 3. Fallback: Check Results (Old logic)
    if os.path.exists(os.path.join(base_results_rel, run_name)):
        return base_results_rel, run_name
        
    if os.path.exists(base_results_rel):
        candidates = [d for d in os.listdir(base_results_rel) if run_name in d]
        if candidates:
            return base_results_rel, candidates[0]

    raise FileNotFoundError(f"Could not find run '{run_name}' in {base_errors_rel} or {base_results_rel}")

def get_save_directory(args, run_name):
    """Creates the directory structure for saving plots."""
    base_plot_dir = "Trial_Plots"
    
    if args.mode == 'diagnostic':
        category_dir = "Diagnostics"
    else:
        category_dir = f"{args.model_type}_{args.window_size}"
        
    full_save_dir = os.path.join(base_plot_dir, category_dir, run_name)
    
    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir)
        print(f"Created directory: {full_save_dir}")
        
    return full_save_dir

def load_metadata_and_preds(base_results_dir, run_name, window_size='1s', mode='sub_dep'):
    print(f"Analyzing Run: {run_name}")
    
    # 1. Construct Paths
    # We manually construct the Errors path to be safe
    if "Results" in base_results_dir:
        base_errors_dir = base_results_dir.replace("Results", "Errors")
    else:
        # Fallback if base_results_dir is weird
        base_errors_dir = os.path.join("Errors", *base_results_dir.split(os.sep)[1:])

    pred_path_errors = os.path.join(base_errors_dir, run_name, "predictions.npy")
    pred_path_results = os.path.join(base_results_dir, run_name, "predictions.npy")
    
    print(f"Looking for predictions at: {os.path.abspath(pred_path_errors)}")
    
    final_pred_path = None
    if os.path.exists(pred_path_errors):
        final_pred_path = pred_path_errors
    elif os.path.exists(pred_path_results):
        print("Not found in Errors, checking Results...")
        final_pred_path = pred_path_results
    else:
        raise FileNotFoundError(f"predictions.npy not found.\nChecked:\n  {pred_path_errors}\n  {pred_path_results}")
        
    print(f"Loading predictions...")
    pred_data = np.load(final_pred_path, allow_pickle=True).item()
    y_pred = pred_data['y_pred']
    y_true = pred_data['y_true']
    
    unique_classes = np.unique(y_true)
    is_binary = (len(unique_classes) == 2)
    print(f"Detected {len(unique_classes)} classes. Mode: {'Binary' if is_binary else '3-Class'}")

    # 2. Reconstruct Metadata
    data_folder = f"Data/ExtractedFeatures_{window_size}"
    label_file = os.path.join(data_folder, "label.mat")
    
    try:
        label_mat = scipy.io.loadmat(label_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found at {label_file}")

    trial_labels = label_mat['label'][0]
    label_map = {-1: 0, 0: 1, 1: 2}
    mapped_labels = np.array([label_map[l] for l in trial_labels])
    
    keep_mask = (mapped_labels != 2) if is_binary else np.ones(len(mapped_labels), dtype=bool)
    
    meta_list = []
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    for f in files:
        parts = f.split('_')
        try: subj_id = int(parts[0])
        except: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)
        
    test_sub_id_filter = None
    if mode == 'sub_indep':
        try:
            parts = run_name.split('_sub')
            if len(parts) > 1:
                test_sub_id_filter = int(parts[-1])
                print(f"Filtering metadata for Test Subject: {test_sub_id_filter}")
        except: pass

    for subj_id in sorted(subject_files.keys()):
        if test_sub_id_filter is not None and subj_id != test_sub_id_filter:
            continue

        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            
            if (mode == 'sub_dep' or mode == 'diagnostic') and session_id != 3:
                continue
                
            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            
            for trial_i in range(1, 16):
                trial_idx = trial_i - 1
                
                if is_binary and not keep_mask[trial_idx]:
                    continue
                    
                key = f"de_LDS{trial_i}"
                if key not in mat: continue
                data = mat[key]
                num_samples = data.shape[1]
                
                lbl_val = mapped_labels[trial_idx]
                if lbl_val == 0: lbl_name = "Negative"
                elif lbl_val == 1: lbl_name = "Neutral"
                else: lbl_name = "Positive"
                
                for _ in range(num_samples):
                    meta_list.append({
                        'Subject': subj_id,
                        'Trial': trial_i,
                        'Label': lbl_name,
                        'Label_Int': lbl_val
                    })
                    
    df = pd.DataFrame(meta_list)
    
    if len(df) != len(y_pred):
        print(f"WARNING: Metadata length ({len(df)}) != Prediction length ({len(y_pred)})")
        print("Truncating to shorter length (this might misalign data if not careful!)")
        min_len = min(len(df), len(y_pred))
        df = df.iloc[:min_len]
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        
    df['Prediction'] = y_pred
    df['Correct'] = (df['Prediction'] == y_true)
    
    return df, run_name, is_binary

def plot_trial_heatmap(df, run_name, save_dir):
    heatmap_data = df.groupby(['Subject', 'Trial'])['Correct'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title(f"Accuracy per Trial (Subject vs Trial ID)\nRun: {run_name}")
    plt.xlabel("Trial ID (Video Clip)")
    plt.ylabel("Subject ID")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "accuracy_heatmap.png")
    plt.savefig(save_path)
    print(f"Saved accuracy heatmap to {save_path}")
    plt.close()

def plot_prediction_bias(df, run_name, is_binary, save_dir):
    heatmap_data = df.groupby(['Subject', 'Trial'])['Prediction'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    
    if is_binary:
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", vmin=0, vmax=1)
        plt.title(f"Prediction Bias (0=All Neg, 1=All Neu)\nRun: {run_name}")
    else:
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=2)
        plt.title(f"Prediction Bias (0=Neg, 1=Neu, 2=Pos)\nRun: {run_name}")
        
    plt.xlabel("Trial ID (Video Clip)")
    plt.ylabel("Subject ID")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "bias_heatmap.png")
    plt.savefig(save_path)
    print(f"Saved bias heatmap to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, 
                        help="Attempt ID (int) OR Folder Name (str) for Diagnostic runs")
    parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'])
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'DGCNN'], 
                        help="Model type used for training (GCN or DGCNN)")
    parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep', 'diagnostic'],
                        help="Training mode: 'sub_dep', 'sub_indep', or 'diagnostic'")
    args = parser.parse_args()
    
    try:
        base_results_dir, run_name = resolve_run_path(args)
        df, run_name, is_binary = load_metadata_and_preds(base_results_dir, run_name, args.window_size, args.mode)
        
        # Create Save Directory
        save_dir = get_save_directory(args, run_name)
        
        print("\n--- Generating Accuracy Heatmap ---")
        plot_trial_heatmap(df, run_name, save_dir)
        
        print("\n--- Generating Bias Heatmap ---")
        plot_prediction_bias(df, run_name, is_binary, save_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


