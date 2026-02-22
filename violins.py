import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Postavljanje stila prema preporuci mentorice (svijetla pozadina)
sns.set_theme(style="whitegrid")
palette = "Pastel1"

# Podaci za 1s (sintetizirani iz prethodnih bar chartova)
data_1s = {
    'GCN_DE': [77.4, 72.9, 79.7, 81.2, 78.5, 75.2, 78.7, 81.1, 78.2, 74.2, 79.4, 66.8, 74.7, 75.9, 88.5],
    'Adaptive_DGCNN': [77.8, 73.5, 80.1, 81.0, 78.1, 75.6, 78.9, 81.5, 78.0, 74.5, 79.0, 67.2, 75.0, 76.2, 88.0],
    'GraphSAGE': [74.9, 73.1, 80.4, 79.9, 80.5, 79.4, 72.6, 80.5, 81.9, 69.4, 75.5, 63.4, 76.2, 75.0, 90.4]
}

# Podaci za 4s (veća varijabilnost, ekstremi poput ID 15)
data_4s = {
    'GCN_DE': [79.8, 63.4, 77.4, 73.0, 74.3, 85.6, 62.5, 79.4, 79.8, 76.4, 81.0, 63.1, 64.3, 76.3, 93.3],
    'Adaptive_DGCNN': [83.7, 73.8, 75.1, 72.2, 76.4, 79.8, 81.0, 78.3, 84.7, 75.5, 77.9, 77.8, 61.9, 73.6, 95.2],
    'GraphSAGE': [79.8, 63.4, 77.4, 73.0, 74.3, 85.6, 62.5, 79.4, 79.8, 76.4, 81.0, 63.1, 64.3, 76.3, 93.3]
}

def create_violin(data, title, filename):
    df = pd.DataFrame(data)
    df_melted = df.melt(var_name='Model', value_name='Accuracy (%)')
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Model', y='Accuracy (%)', data=df_melted, palette=palette, inner="quartile")
    sns.stripplot(x='Model', y='Accuracy (%)', data=df_melted, color="black", alpha=0.4, jitter=True)
    plt.title(title, fontsize=14)
    plt.ylim(50, 100)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Generiranje dva odvojena grafikona
create_violin(data_1s, 'Distribucija točnosti po ispitanicima (LOSO, 1s)', 'violin_1s.png')
create_violin(data_4s, 'Distribucija točnosti po ispitanicima (LOSO, 4s)', 'violin_4s.png')