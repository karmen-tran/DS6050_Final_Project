import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_top_features(feature_names, coef_values, top_n=10, title="Top Features"):
    top_idx = np.argsort(np.abs(coef_values))[::-1][:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_values = [coef_values[i] for i in top_idx]

    plt.figure(figsize=(8,5))
    sns.barplot(x=top_values, y=top_features, palette='viridis')
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()
