# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import scipy.io
# from sklearn.preprocessing import StandardScaler

# # Configuration
# DATA_FOLDER = "Data/ExtractedFeatures_1s"
# OUTPUT_FOLDER = "Analysis_Results"
# BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
# EMOTIONS = ['Negative', 'Neutral', 'Positive']
# COLORS = {'Negative': '#e74c3c', 'Neutral': '#f1c40f', 'Positive': '#2ecc71'}
# CHANNEL_NAMES = [
#     'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
#     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
#     'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
#     'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
#     'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
# ]


# def create_subject_structure(subject_label):
#     """
#     Creates the folder structure:
#     Analysis_Results/Subject_{label}/
#         Alpha/
#         Beta/
#         ...
#         Summary/ (for non-band specific plots)
#     Returns the base path for the subject.
#     """
#     base_path = os.path.join(OUTPUT_FOLDER, f"Subject_{subject_label}")
    
#     # Create Summary folder
#     os.makedirs(os.path.join(base_path, "Summary"), exist_ok=True)
    
#     # Create Band folders
#     for band in BANDS:
#         os.makedirs(os.path.join(base_path, band), exist_ok=True)
        
#     return base_path


# def load_subject_data(subject_id):
#     """Loads DE features for a specific subject."""
#     print(f"Loading Subject {subject_id}...")
#     label_file = os.path.join(DATA_FOLDER, "label.mat")
#     try:
#         label_mat = scipy.io.loadmat(label_file)
#         trial_labels = label_mat['label'][0] # -1, 0, 1
#     except:
#         print("Label file missing.")
#         return None, None

#     label_map = {-1: 0, 0: 1, 1: 2} # Map to 0, 1, 2
    
#     X_list = []
#     y_list = []
    
#     # Find files for this subject
#     files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    
#     for fname in sorted(files):
#         path = os.path.join(DATA_FOLDER, fname)
#         try: mat = scipy.io.loadmat(path)
#         except: continue
        
#         for trial_i in range(1, 16): # 15 trials
#             key = f"de_LDS{trial_i}"
#             if key not in mat: continue
            
#             # Data shape: (62, N, 5) -> Transpose to (N, 62, 5)
#             data = mat[key]
#             data = np.transpose(data, (1, 0, 2)) 
            
#             X_list.append(data)
#             # Create labels for this trial
#             y_list.append(np.full(data.shape[0], label_map[trial_labels[trial_i-1]]))
            
#     if not X_list: return None, None
    
#     return np.concatenate(X_list), np.concatenate(y_list)


# def plot_line_comparison(data_dict, title, ylabel, filename):
#     """
#     Generates a line plot comparing emotions across channels.
#     data_dict: { 'Negative': [62 values], 'Neutral': [62 values], ... }
#     """
#     plt.figure(figsize=(18, 6))
    
#     x = np.arange(len(CHANNEL_NAMES))
    
#     for emo in EMOTIONS:
#         if emo in data_dict:
#             plt.plot(x, data_dict[emo], marker='o', label=emo, color=COLORS[emo], linewidth=2)
            
#     plt.title(title, fontsize=14)
#     plt.xlabel('Channel', fontsize=12)
#     plt.ylabel(ylabel, fontsize=12)
#     plt.xticks(x, CHANNEL_NAMES, rotation=90, fontsize=8)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     save_path = os.path.join(OUTPUT_FOLDER, filename)
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved {filename}")


# def analyze_band_power(X, y, subject_id="All"):
#     """
#     Plots the average power of each band for each emotion.
#     X shape: (N, 62, 5)
#     """
#     # Average across channels -> (N, 5)
#     X_bands = np.mean(X, axis=1)
    
#     # Prepare data for plotting
#     plot_data = []
#     for i in range(len(X)):
#         for b_idx, band in enumerate(BANDS):
#             plot_data.append({
#                 'Emotion': EMOTIONS[y[i]],
#                 'Band': band,
#                 'Power': X_bands[i, b_idx]
#             })
            
#     # Convert to simple lists for plotting (faster than DataFrame for huge data)
#     # Actually, let's just compute means manually to save memory
#     means = np.zeros((3, 5)) # (Emotion, Band)
#     for emo_i in range(3):
#         mask = (y == emo_i)
#         if np.sum(mask) > 0:
#             means[emo_i] = np.mean(X_bands[mask], axis=0)
            
#     # Plot
#     plt.figure(figsize=(10, 6))
#     x = np.arange(len(BANDS))
#     width = 0.25
    
#     plt.bar(x - width, means[0], width, label='Negative', color='#e74c3c')
#     plt.bar(x,        means[1], width, label='Neutral',  color='#f1c40f')
#     plt.bar(x + width, means[2], width, label='Positive', color='#2ecc71')
    
#     plt.xlabel('Frequency Band')
#     plt.ylabel('Mean DE Feature Value')
#     plt.title(f'Band Power Analysis - Subject {subject_id}')
#     plt.xticks(x, BANDS)
#     plt.legend()
#     plt.grid(axis='y', alpha=0.3)
    
#     save_path = os.path.join(OUTPUT_FOLDER, f"Band_Power_Sub_{subject_id}.png")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved Band Power plot to {save_path}")


# def analyze_channel_means_lineplot(X, y, subject_id="All"):
#     """
#     Plots Mean Channel Amplitude (Line Plot) for each band.
#     """
#     # X shape: (N, 62, 5)
    
#     for b_idx, band in enumerate(BANDS):
#         band_data = X[:, :, b_idx] # (N, 62)
        
#         means_dict = {}
#         for i, emo in enumerate(EMOTIONS):
#             mask = (y == i)
#             if np.sum(mask) > 0:
#                 means_dict[emo] = np.mean(band_data[mask], axis=0)
        
#         plot_line_comparison(
#             means_dict, 
#             f'{band} Band: Mean Channel Amplitude - Subject {subject_id}',
#             'Mean DE Value',
#             f"LinePlot_Mean_{band}_Sub_{subject_id}.png"
#         )

# def analyze_channel_variance_lineplot(X, y, subject_id="All"):
#     """
#     Plots Channel Variance (Line Plot) for each band.
#     """
#     # X shape: (N, 62, 5)
    
#     for b_idx, band in enumerate(BANDS):
#         band_data = X[:, :, b_idx] # (N, 62)
        
#         var_dict = {}
#         for i, emo in enumerate(EMOTIONS):
#             mask = (y == i)
#             if np.sum(mask) > 0:
#                 var_dict[emo] = np.var(band_data[mask], axis=0)
        
#         plot_line_comparison(
#             var_dict, 
#             f'{band} Band: Channel Variance - Subject {subject_id}',
#             'Variance',
#             f"LinePlot_Var_{band}_Sub_{subject_id}.png"
#         )

# def analyze_distribution_overlay(X, y, subject_id="All"):
#     """
#     Plots KDE Distribution Overlay for each band (Global Power).
#     """
#     # Average across channels to get global band power -> (N, 5)
#     X_global = np.mean(X, axis=1)
    
#     for b_idx, band in enumerate(BANDS):
#         plt.figure(figsize=(10, 6))
        
#         for i, emo in enumerate(EMOTIONS):
#             mask = (y == i)
#             if np.sum(mask) > 0:
#                 sns.kdeplot(X_global[mask, b_idx], label=emo, color=COLORS[emo], fill=True, alpha=0.1)
                
#         plt.title(f'{band} Band: Global Distribution Overlay - Subject {subject_id}')
#         plt.xlabel('Mean Band Power')
#         plt.ylabel('Density')
#         plt.legend()
        
#         save_path = os.path.join(OUTPUT_FOLDER, f"Dist_Overlay_{band}_Sub_{subject_id}.png")
#         plt.savefig(save_path)
#         plt.close()
#         print(f"Saved Distribution Overlay for {band}")


# def analyze_channel_amplitude(X, y, subject_id="All"):
#     """
#     Plots a heatmap of Channel Activity for each emotion.
#     X shape: (N, 62, 5)
#     """
#     # Average across bands -> (N, 62) (Total Energy)
#     # Or we can look at Gamma specifically? Let's do Total Energy first.
#     X_channels = np.mean(X, axis=2)
    
#     # Compute mean per emotion -> (3, 62)
#     means = np.zeros((3, 62))
#     for emo_i in range(3):
#         mask = (y == emo_i)
#         if np.sum(mask) > 0:
#             means[emo_i] = np.mean(X_channels[mask], axis=0)
            
#     # Plot Heatmap
#     plt.figure(figsize=(15, 6))
#     sns.heatmap(means, xticklabels=CHANNEL_NAMES, yticklabels=EMOTIONS, cmap="viridis", cbar_kws={'label': 'Mean DE Value'})
#     plt.title(f'Channel Amplitude Heatmap (All Bands) - Subject {subject_id}')
#     plt.xlabel('Channel')
    
#     save_path = os.path.join(OUTPUT_FOLDER, f"Channel_Amp_Sub_{subject_id}.png")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved Channel Heatmap to {save_path}")


# def main():
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
#     # 1. Analyze "Hard" Subject (e.g., 2)
#     X_2, y_2 = load_subject_data(2)
#     if X_2 is not None:
#         print("\n--- Analyzing Subject 2 (Hard) ---")
#         analyze_band_power(X_2, y_2, subject_id="02_Hard")
#         analyze_channel_amplitude(X_2, y_2, subject_id="02_Hard")
#         analyze_channel_means_lineplot(X_2, y_2, subject_id="02_Hard")
#         analyze_channel_variance_lineplot(X_2, y_2, subject_id="02_Hard")
#         analyze_distribution_overlay(X_2, y_2, subject_id="02_Hard")
        
#     # 2. Analyze "Easy" Subject (e.g., 14)
#     X_14, y_14 = load_subject_data(14)
#     if X_14 is not None:
#         print("\n--- Analyzing Subject 14 (Easy) ---")
#         analyze_band_power(X_14, y_14, subject_id="14_Easy")
#         analyze_channel_amplitude(X_14, y_14, subject_id="14_Easy")
#         analyze_channel_means_lineplot(X_14, y_14, subject_id="14_Easy")
#         analyze_channel_variance_lineplot(X_14, y_14, subject_id="14_Easy")
#         analyze_distribution_overlay(X_14, y_14, subject_id="14_Easy")

#     # 3. Analyze All Subjects (Global Average)
#     print("\n--- Analyzing All Subjects ---")
#     X_all_list, y_all_list = [], []
#     for i in range(1, 16):
#         X, y = load_subject_data(i)
#         if X is not None:
#             X_all_list.append(X)
#             y_all_list.append(y)
            
#     if X_all_list:
#         X_all = np.concatenate(X_all_list)
#         y_all = np.concatenate(y_all_list)
#         analyze_band_power(X_all, y_all, subject_id="All")
#         analyze_channel_amplitude(X_all, y_all, subject_id="All")
#         analyze_channel_means_lineplot(X_all, y_all, subject_id="All")
#         analyze_channel_variance_lineplot(X_all, y_all, subject_id="All")
#         analyze_distribution_overlay(X_all, y_all, subject_id="All")

# if __name__ == "__main__":
#     main()



# filepath: c:\Users\eleko\OneDrive\Radna površina\GCN-CNN-for-EEG\utils\advanced_feature_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_FOLDER = "Data/ExtractedFeatures_1s"
OUTPUT_FOLDER = "Analysis_Results"
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
EMOTIONS = ['Negative', 'Neutral', 'Positive']
COLORS = {'Negative': '#e74c3c', 'Neutral': '#f1c40f', 'Positive': '#2ecc71'}
CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
    'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
    'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
    'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

def create_subject_structure(subject_label):
    """
    Creates the folder structure:
    Analysis_Results/Subject_{label}/
        Alpha/
        Beta/
        ...
        Summary/ (for non-band specific plots)
    Returns the base path for the subject.
    """
    base_path = os.path.join(OUTPUT_FOLDER, f"Subject_{subject_label}")
    
    # Create Summary folder
    os.makedirs(os.path.join(base_path, "Summary"), exist_ok=True)
    
    # Create Band folders
    for band in BANDS:
        os.makedirs(os.path.join(base_path, band), exist_ok=True)
        
    return base_path

def load_subject_data(subject_id):
    """Loads DE features for a specific subject."""
    print(f"Loading Subject {subject_id}...")
    label_file = os.path.join(DATA_FOLDER, "label.mat")
    try:
        label_mat = scipy.io.loadmat(label_file)
        trial_labels = label_mat['label'][0] # -1, 0, 1
    except:
        print("Label file missing.")
        return None, None

    label_map = {-1: 0, 0: 1, 1: 2} # Map to 0, 1, 2
    
    X_list = []
    y_list = []
    
    # Find files for this subject
    files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    
    for fname in sorted(files):
        path = os.path.join(DATA_FOLDER, fname)
        try: mat = scipy.io.loadmat(path)
        except: continue
        
        for trial_i in range(1, 16): # 15 trials
            key = f"de_LDS{trial_i}"
            if key not in mat: continue
            
            # Data shape: (62, N, 5) -> Transpose to (N, 62, 5)
            data = mat[key]
            data = np.transpose(data, (1, 0, 2)) 
            
            X_list.append(data)
            # Create labels for this trial
            y_list.append(np.full(data.shape[0], label_map[trial_labels[trial_i-1]]))
            
    if not X_list: return None, None
    
    return np.concatenate(X_list), np.concatenate(y_list)


def plot_line_comparison(data_dict, title, ylabel, save_path):
    """
    Generates a line plot comparing emotions across channels and saves to save_path.
    """
    plt.figure(figsize=(18, 6))
    
    x = np.arange(len(CHANNEL_NAMES))
    
    for emo in EMOTIONS:
        if emo in data_dict:
            plt.plot(x, data_dict[emo], marker='o', label=emo, color=COLORS[emo], linewidth=2)
            
    plt.title(title, fontsize=14)
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x, CHANNEL_NAMES, rotation=90, fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {os.path.basename(save_path)}")


def analyze_band_power(X, y, subject_label, base_path):
    """
    Plots the average power of each band for each emotion.
    Saves to Subject_{label}/Summary/
    """
    X_bands = np.mean(X, axis=1)
    
    means = np.zeros((3, 5)) # (Emotion, Band)
    for emo_i in range(3):
        mask = (y == emo_i)
        if np.sum(mask) > 0:
            means[emo_i] = np.mean(X_bands[mask], axis=0)
            
    plt.figure(figsize=(10, 6))
    x = np.arange(len(BANDS))
    width = 0.25
    
    plt.bar(x - width, means[0], width, label='Negative', color='#e74c3c')
    plt.bar(x,        means[1], width, label='Neutral',  color='#f1c40f')
    plt.bar(x + width, means[2], width, label='Positive', color='#2ecc71')
    
    plt.xlabel('Frequency Band')
    plt.ylabel('Mean DE Feature Value')
    plt.title(f'Band Power Analysis - Subject {subject_label}')
    plt.xticks(x, BANDS)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(base_path, "Summary", "Band_Power_Analysis.png")
    plt.savefig(save_path)
    plt.close()


def analyze_channel_means_lineplot(X, y, subject_label, base_path):
    """
    Plots Mean Channel Amplitude (Line Plot) for each band.
    Saves to Subject_{label}/{Band}/
    """
    for b_idx, band in enumerate(BANDS):
        band_data = X[:, :, b_idx] # (N, 62)
        
        means_dict = {}
        for i, emo in enumerate(EMOTIONS):
            mask = (y == i)
            if np.sum(mask) > 0:
                means_dict[emo] = np.mean(band_data[mask], axis=0)
        
        save_path = os.path.join(base_path, band, "Mean_Channel_Amplitude.png")
        plot_line_comparison(
            means_dict, 
            f'{band} Band: Mean Channel Amplitude - Subject {subject_label}',
            'Mean DE Value',
            save_path
        )

def analyze_channel_variance_lineplot(X, y, subject_label, base_path):
    """
    Plots Channel Variance (Line Plot) for each band.
    Saves to Subject_{label}/{Band}/
    """
    for b_idx, band in enumerate(BANDS):
        band_data = X[:, :, b_idx] # (N, 62)
        
        var_dict = {}
        for i, emo in enumerate(EMOTIONS):
            mask = (y == i)
            if np.sum(mask) > 0:
                var_dict[emo] = np.var(band_data[mask], axis=0)
        
        save_path = os.path.join(base_path, band, "Channel_Variance.png")
        plot_line_comparison(
            var_dict, 
            f'{band} Band: Channel Variance - Subject {subject_label}',
            'Variance',
            save_path
        )

def analyze_distribution_overlay(X, y, subject_label, base_path):
    """
    Plots KDE Distribution Overlay for each band (Global Power).
    Saves to Subject_{label}/{Band}/
    """
    X_global = np.mean(X, axis=1)
    
    for b_idx, band in enumerate(BANDS):
        plt.figure(figsize=(10, 6))
        
        for i, emo in enumerate(EMOTIONS):
            mask = (y == i)
            if np.sum(mask) > 0:
                sns.kdeplot(X_global[mask, b_idx], label=emo, color=COLORS[emo], fill=True, alpha=0.1)
                
        plt.title(f'{band} Band: Global Distribution Overlay - Subject {subject_label}')
        plt.xlabel('Mean Band Power')
        plt.ylabel('Density')
        plt.legend()
        
        save_path = os.path.join(base_path, band, "Global_Distribution_Overlay.png")
        plt.savefig(save_path)
        plt.close()


def analyze_channel_amplitude(X, y, subject_label, base_path):
    """
    Plots a heatmap of Channel Activity for each emotion (All Bands).
    Saves to Subject_{label}/Summary/
    """
    X_channels = np.mean(X, axis=2)
    
    means = np.zeros((3, 62))
    for emo_i in range(3):
        mask = (y == emo_i)
        if np.sum(mask) > 0:
            means[emo_i] = np.mean(X_channels[mask], axis=0)
            
    plt.figure(figsize=(15, 6))
    sns.heatmap(means, xticklabels=CHANNEL_NAMES, yticklabels=EMOTIONS, cmap="viridis", cbar_kws={'label': 'Mean DE Value'})
    plt.title(f'Channel Amplitude Heatmap (All Bands) - Subject {subject_label}')
    plt.xlabel('Channel')
    
    save_path = os.path.join(base_path, "Summary", "Channel_Amplitude_Heatmap.png")
    plt.savefig(save_path)
    plt.close()


def generate_subject_plots(subject_id):
    """
    Main function to generate all plots for a given subject ID.
    Can pass 'All' to aggregate all subjects.
    """
    subject_label = str(subject_id)
    
    # 1. Load Data
    if subject_id == "All":
        print("\n--- Aggregating All Subjects ---")
        X_list, y_list = [], []
        for i in range(1, 16): # Assuming 15 subjects
             X, y = load_subject_data(i)
             if X is not None:
                 X_list.append(X)
                 y_list.append(y)
        if not X_list:
            print("No data found.")
            return
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
    else:
        print(f"\n--- Analyzing Subject {subject_id} ---")
        X, y = load_subject_data(subject_id)
        if X is None:
            print(f"Skipping Subject {subject_id} (No Data)")
            return

    # 2. Create Directory Structure
    base_path = create_subject_structure(subject_label)
    print(f"Saving results to: {base_path}")

    # 3. Run Analyses
    analyze_band_power(X, y, subject_label, base_path)
    analyze_channel_amplitude(X, y, subject_label, base_path)
    analyze_channel_means_lineplot(X, y, subject_label, base_path)
    analyze_channel_variance_lineplot(X, y, subject_label, base_path)
    analyze_distribution_overlay(X, y, subject_label, base_path)
    
    print(f"Completed analysis for Subject {subject_label}")


def main():
    # Example usage:
    generate_subject_plots(8)
    generate_subject_plots(9) 
    generate_subject_plots(10) 
    generate_subject_plots(11)
    generate_subject_plots(13)



if __name__ == "__main__":
    main()