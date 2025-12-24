# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import argparse
# from sklearn.metrics import confusion_matrix
# from sklearn.manifold import TSNE
# import scipy.io

# def load_metadata(data_folder, label_file):
#     """Reconstructs Subject and Session IDs for the test set."""
#     print(f"Loading metadata from {data_folder}...")
#     try:
#         label_mat = scipy.io.loadmat(label_file)
#     except FileNotFoundError:
#         print("Label file not found.")
#         return None, None

#     # Reconstruct metadata logic (same as train_de.py)
#     session_list = []
#     subject_list = []
    
#     files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
#     subject_files = {}
#     for f in files:
#         parts = f.split('_')
#         try: subj_id = int(parts[0])
#         except: continue
#         if subj_id not in subject_files: subject_files[subj_id] = []
#         subject_files[subj_id].append(f)
        
#     for subj_id in sorted(subject_files.keys()):
#         s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
#         for sess_idx, fname in enumerate(s_files):
#             session_id = sess_idx + 1
#             file_path = os.path.join(data_folder, fname)
#             try: mat = scipy.io.loadmat(file_path)
#             except: continue
#             for trial_i in range(1, 16):
#                 key = f"de_LDS{trial_i}"
#                 if key not in mat: continue
#                 data = mat[key]
#                 num_samples = data.shape[1]
#                 session_list.append(np.full(num_samples, session_id))
#                 subject_list.append(np.full(num_samples, subj_id))

#     sessions = np.concatenate(session_list, axis=0)
#     subjects = np.concatenate(subject_list, axis=0)
#     return sessions, subjects

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run_id', type=int, required=True, help="Attempt ID")
#     parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'])
#     parser.add_argument('--model_type', type=str, default='DGCNN', choices=['GCN', 'DGCNN'], 
#                         help="Model type used for training (GCN or DGCNN)")
#     parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'],
#                         help="Training mode: 'sub_dep' or 'sub_indep'")
#     args = parser.parse_args()

#     # Paths - Updated to use model_type and mode
#     model_name = f"{args.model_type}_DE_{args.window_size}"
#     base_results_dir = os.path.join("Results", args.mode, model_name)
    
#     # Search for the directory starting with Attempt_{run_id}
#     run_prefix = f"Attempt_{args.run_id}"
#     found_dirs = []
#     if os.path.exists(base_results_dir):
#         found_dirs = [d for d in os.listdir(base_results_dir) 
#                       if d.startswith(run_prefix) and os.path.isdir(os.path.join(base_results_dir, d))]
    
#     if not found_dirs:
#         print(f"Error: No run directory found starting with '{run_prefix}' in {base_results_dir}")
#         return
        
#     # Use the first match
#     run_name = found_dirs[0]
#     print(f"Analyzing Run: {run_name}")
    
#     results_dir = os.path.join(base_results_dir, run_name)
#     history_file = os.path.join(results_dir, "evolution_history.npy")
    
#     if not os.path.exists(history_file):
#         print(f"Error: File not found at {history_file}")
#         return

#     print(f"Loading history from {history_file}...")
#     data = np.load(history_file, allow_pickle=True).item()
#     preds_history = data['preds_history'] # List of arrays
#     embeddings_history = data['embeddings_history'] # Dict: epoch -> array
#     y_true = data['true_labels']
    
#     y_true_arr = np.array(y_true)
#     print(f"y_true_arr shape: {y_true_arr.shape}")

#     # Load Metadata for Error Analysis
#     data_folder = f"Data/ExtractedFeatures_{args.window_size}"
#     label_file = os.path.join(data_folder, "label.mat")
#     sessions, subjects = load_metadata(data_folder, label_file)
    
#     # Filter metadata based on mode
#     test_subjects = None
#     if args.mode == 'sub_dep':
#         # Filter for Test Set (Session 3)
#         test_mask = (sessions == 3)
#         test_subjects = subjects[test_mask]
#     else:
#         # For sub_indep, infer test subject from run name
#         try:
#             # Expected format: Attempt_X_LOSO_subY
#             parts = run_name.split('_sub')
#             if len(parts) > 1:
#                 test_sub_id = int(parts[-1])
#                 print(f"Inferred Test Subject ID: {test_sub_id}")
#                 test_mask = (subjects == test_sub_id)
#                 test_subjects = subjects[test_mask]
#             else:
#                 print("Could not infer test subject from run name.")
#         except Exception as e:
#             print(f"Error parsing subject ID: {e}")

#     if test_subjects is not None:
#         if len(test_subjects) != len(y_true):
#             print(f"Warning: Metadata length mismatch ({len(test_subjects)} vs {len(y_true)}). Skipping subject-specific error analysis.")
#             test_subjects = None

#     # --- 1. Confusion Matrix Evolution ---
#     print("Generating Confusion Matrix Evolution...")
#     epochs_to_plot = [0, 10, 30, 60, len(preds_history)-1]
#     epochs_to_plot = [e for e in epochs_to_plot if e < len(preds_history)]
    
#     fig, axes = plt.subplots(1, len(epochs_to_plot), figsize=(20, 4))
#     if len(epochs_to_plot) == 1: axes = [axes]
    
#     for i, epoch in enumerate(epochs_to_plot):
#         cm = confusion_matrix(y_true, preds_history[epoch])
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
#                     xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
#         axes[i].set_title(f"Epoch {epoch+1}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_dir, "cm_evolution.png"))
#     plt.show()

#     # --- 2. Error Heatmap (Subject vs Epoch) ---
#     if test_subjects is not None:
#         print("Generating Error Heatmap...")
#         unique_subs = np.unique(test_subjects)
#         error_matrix = np.zeros((len(unique_subs), len(preds_history)))
    
#         for epoch_idx, preds in enumerate(preds_history):
#             preds_arr = np.array(preds)
#             errors = (preds_arr != y_true_arr)
#             for i, sub_id in enumerate(unique_subs):
#                 sub_mask = (test_subjects == sub_id)
#                 sub_error_rate = np.mean(errors[sub_mask])
#                 error_matrix[i, epoch_idx] = sub_error_rate

#         plt.figure(figsize=(15, max(4, len(unique_subs)*0.5))) # Adjust height based on num subjects
#         sns.heatmap(error_matrix, cmap='Reds', xticklabels=10, yticklabels=unique_subs)
#         plt.title("Error Rate per Subject over Epochs")
#         plt.xlabel("Epoch")
#         plt.ylabel("Subject ID")
#         plt.savefig(os.path.join(results_dir, "error_heatmap.png"))
#         plt.show()

#     # --- 3. t-SNE Clustering Evolution ---
#     print("Generating t-SNE Clustering...")
#     avail_epochs = sorted(embeddings_history.keys())
#     if not avail_epochs:
#         print("No embeddings found in history.")
#     else:
#         # Pick first, middle, last
#         selected_epochs = [avail_epochs[0], avail_epochs[len(avail_epochs)//2], avail_epochs[-1]]
#         selected_epochs = sorted(list(set(selected_epochs)))
        
#         fig, axes = plt.subplots(1, len(selected_epochs), figsize=(20, 6))
#         if len(selected_epochs) == 1: axes = [axes] 
        
#         # Subsample for t-SNE speed (max 2000 points)
#         indices = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
        
#         for i, epoch in enumerate(selected_epochs):
#             emb = np.array(embeddings_history[epoch])[indices]
#             labels = np.array(y_true_arr)[indices]
            
#             tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#             emb_2d = tsne.fit_transform(emb)
            
#             scatter = axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
#             axes[i].set_title(f"Epoch {epoch+1} Embeddings")
            
#         plt.legend(handles=scatter.legend_elements()[0], labels=['Neg', 'Neu', 'Pos'])
#         plt.tight_layout()
#         plt.savefig(os.path.join(results_dir, "tsne_evolution.png"))
#         plt.show()

# if __name__ == "__main__":
#     main()




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import scipy.io

def load_metadata(data_folder, label_file):
    """Reconstructs Subject and Session IDs for the test set."""
    print(f"Loading metadata from {data_folder}...")
    try:
        label_mat = scipy.io.loadmat(label_file)
    except FileNotFoundError:
        print("Label file not found.")
        return None, None

    # Reconstruct metadata logic (same as train_de.py)
    session_list = []
    subject_list = []
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.mat') and f != 'label.mat']
    subject_files = {}
    for f in files:
        parts = f.split('_')
        try: subj_id = int(parts[0])
        except: continue
        if subj_id not in subject_files: subject_files[subj_id] = []
        subject_files[subj_id].append(f)
        
    for subj_id in sorted(subject_files.keys()):
        s_files = sorted(subject_files[subj_id], key=lambda x: x.split('_')[1])
        for sess_idx, fname in enumerate(s_files):
            session_id = sess_idx + 1
            file_path = os.path.join(data_folder, fname)
            try: mat = scipy.io.loadmat(file_path)
            except: continue
            for trial_i in range(1, 16):
                key = f"de_LDS{trial_i}"
                if key not in mat: continue
                data = mat[key]
                num_samples = data.shape[1]
                session_list.append(np.full(num_samples, session_id))
                subject_list.append(np.full(num_samples, subj_id))

    sessions = np.concatenate(session_list, axis=0)
    subjects = np.concatenate(subject_list, axis=0)
    return sessions, subjects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True, help="Attempt ID")
    parser.add_argument('--window_size', type=str, default='1s', choices=['1s', '4s'])
    parser.add_argument('--model_type', type=str, default='DGCNN', choices=['GCN', 'DGCNN'], 
                        help="Model type used for training (GCN or DGCNN)")
    parser.add_argument('--mode', type=str, default='sub_dep', choices=['sub_dep', 'sub_indep'],
                        help="Training mode: 'sub_dep' or 'sub_indep'")
    args = parser.parse_args()

    # Paths - Updated to use model_type and mode
    model_name = f"{args.model_type}_DE_{args.window_size}"
    
    # Look in Results/{MODE}/{MODEL_NAME}
    base_results_dir = os.path.join("Results", args.mode, model_name)
    
    # Search for the directory starting with Attempt_{run_id}
    run_prefix = f"Attempt_{args.run_id}"
    found_dirs = []
    if os.path.exists(base_results_dir):
        found_dirs = [d for d in os.listdir(base_results_dir) 
                      if d.startswith(run_prefix) and os.path.isdir(os.path.join(base_results_dir, d))]
    
    if not found_dirs:
        print(f"Error: No run directory found starting with '{run_prefix}' in {base_results_dir}")
        # Fallback check for old flat structure (just in case)
        old_base = os.path.join("Results", model_name)
        if os.path.exists(old_base):
             print(f"Checking old structure: {old_base}")
             found_dirs = [d for d in os.listdir(old_base) if d.startswith(run_prefix)]
             if found_dirs:
                 print(f"Found in old structure! Please move folders to {base_results_dir} for consistency.")
                 base_results_dir = old_base
             else:
                 return
        else:
            return
        
    # Use the first match
    run_name = found_dirs[0]
    print(f"Analyzing Run: {run_name}")
    
    results_dir = os.path.join(base_results_dir, run_name)
    history_file = os.path.join(results_dir, "evolution_history.npy")
    
    if not os.path.exists(history_file):
        print(f"Error: File not found at {history_file}")
        return

    print(f"Loading history from {history_file}...")
    data = np.load(history_file, allow_pickle=True).item()
    preds_history = data['preds_history'] # List of arrays
    embeddings_history = data['embeddings_history'] # Dict: epoch -> array
    y_true = data['true_labels']
    
    y_true_arr = np.array(y_true)
    print(f"y_true_arr shape: {y_true_arr.shape}")

    # Load Metadata for Error Analysis
    data_folder = f"Data/ExtractedFeatures_{args.window_size}"
    label_file = os.path.join(data_folder, "label.mat")
    sessions, subjects = load_metadata(data_folder, label_file)
    
    # Filter metadata based on mode
    test_subjects = None
    if args.mode == 'sub_dep':
        # Filter for Test Set (Session 3)
        test_mask = (sessions == 3)
        test_subjects = subjects[test_mask]
    else:
        # For sub_indep, infer test subject from run name
        try:
            # Expected format: Attempt_X_LOSO_subY
            parts = run_name.split('_sub')
            if len(parts) > 1:
                test_sub_id = int(parts[-1])
                print(f"Inferred Test Subject ID: {test_sub_id}")
                test_mask = (subjects == test_sub_id)
                test_subjects = subjects[test_mask]
            else:
                print("Could not infer test subject from run name.")
        except Exception as e:
            print(f"Error parsing subject ID: {e}")

    if test_subjects is not None:
        if len(test_subjects) != len(y_true):
            print(f"Warning: Metadata length mismatch ({len(test_subjects)} vs {len(y_true)}). Skipping subject-specific error analysis.")
            test_subjects = None

    # --- 1. Confusion Matrix Evolution ---
    print("Generating Confusion Matrix Evolution...")
    epochs_to_plot = [0, 10, 30, 60, len(preds_history)-1]
    epochs_to_plot = [e for e in epochs_to_plot if e < len(preds_history)]
    
    fig, axes = plt.subplots(1, len(epochs_to_plot), figsize=(20, 4))
    if len(epochs_to_plot) == 1: axes = [axes]
    
    for i, epoch in enumerate(epochs_to_plot):
        cm = confusion_matrix(y_true, preds_history[epoch])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
        axes[i].set_title(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cm_evolution.png"))
    plt.show()

    # --- 2. Error Heatmap (Subject vs Epoch) ---
    if test_subjects is not None:
        print("Generating Error Heatmap...")
        unique_subs = np.unique(test_subjects)
        error_matrix = np.zeros((len(unique_subs), len(preds_history)))
    
        for epoch_idx, preds in enumerate(preds_history):
            preds_arr = np.array(preds)
            errors = (preds_arr != y_true_arr)
            for i, sub_id in enumerate(unique_subs):
                sub_mask = (test_subjects == sub_id)
                sub_error_rate = np.mean(errors[sub_mask])
                error_matrix[i, epoch_idx] = sub_error_rate

        plt.figure(figsize=(15, max(4, len(unique_subs)*0.5))) # Adjust height based on num subjects
        sns.heatmap(error_matrix, cmap='Reds', xticklabels=10, yticklabels=unique_subs)
        plt.title("Error Rate per Subject over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Subject ID")
        plt.savefig(os.path.join(results_dir, "error_heatmap.png"))
        plt.show()

    # --- 3. t-SNE Clustering Evolution ---
    print("Generating t-SNE Clustering...")
    avail_epochs = sorted(embeddings_history.keys())
    if not avail_epochs:
        print("No embeddings found in history.")
    else:
        # Pick first, middle, last
        selected_epochs = [avail_epochs[0], avail_epochs[len(avail_epochs)//2], avail_epochs[-1]]
        selected_epochs = sorted(list(set(selected_epochs)))
        
        fig, axes = plt.subplots(1, len(selected_epochs), figsize=(20, 6))
        if len(selected_epochs) == 1: axes = [axes] 
        
        # Subsample for t-SNE speed (max 2000 points)
        indices = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
        
        for i, epoch in enumerate(selected_epochs):
            emb = np.array(embeddings_history[epoch])[indices]
            labels = np.array(y_true_arr)[indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            emb_2d = tsne.fit_transform(emb)
            
            scatter = axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
            axes[i].set_title(f"Epoch {epoch+1} Embeddings")
            
        plt.legend(handles=scatter.legend_elements()[0], labels=['Neg', 'Neu', 'Pos'])
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "tsne_evolution.png"))
        plt.show()

if __name__ == "__main__":
    main()