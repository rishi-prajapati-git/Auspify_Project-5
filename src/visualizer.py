"""
Classification Visualization Engine.
Renders high-resolution confusion matrix heatmaps and cross-algorithm benchmark comparisons.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set_theme(style="whitegrid")
NETFLIX_RED = "#E50914"
NETFLIX_BLACK = "#221F1F"


def plot_algorithm_comparison(comparison_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, comparison_df["Accuracy"] * 100, width, label="Accuracy (%)", color=NETFLIX_RED, edgecolor="black")
    rects2 = ax.bar(x + width/2, comparison_df["F1-Score (Weighted)"] * 100, width, label="F1-Score (%)", color=NETFLIX_BLACK, edgecolor="black")
    
    ax.set_ylabel("Score (%)")
    ax.set_title("Machine Learning Classification Algorithm Benchmark Comparison", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Algorithm"], fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig(out_path / "01_algorithm_benchmark_comparison.png", dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, labels: list, best_algo_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Calculate percentages per true label
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt=".1%", 
        cmap="Reds", 
        xticklabels=labels, 
        yticklabels=labels,
        ax=ax,
        cbar=True,
        linewidths=1,
        linecolor="black"
    )
    
    ax.set_title(f"Normalized Confusion Matrix ({best_algo_name})", pad=15, fontweight="bold")
    ax.set_ylabel("True Demographic Tier", fontweight="bold")
    ax.set_xlabel("Predicted Demographic Tier", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path / "02_best_model_confusion_matrix.png", dpi=300)
    plt.close()


def generate_all_classification_figures(
    comparison_df: pd.DataFrame, 
    cm: np.ndarray, 
    labels: list, 
    best_algo_name: str, 
    figures_dir: str = "figures"
):
    out_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Rendering publication-quality classification figures...")
    plot_algorithm_comparison(comparison_df, out_path)
    plot_confusion_matrix_heatmap(cm, labels, best_algo_name, out_path)
    print(f"[SUCCESS] High-resolution classification figures saved in: {out_path.resolve()}")
