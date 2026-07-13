"""Phase 2 - EDA visualizations for every dataset.

Required plots: Histogram, KDE, Boxplot, QQ-Plot, Pair Plot.
All figures are saved under ``results/eda/<dataset>/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for scripted runs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils.io import ensure_dir, save_figure


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram + KDE per numeric column, one combined grid figure."""
    cols = df.select_dtypes("number").columns
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Histogram + KDE")
    fig.tight_layout()
    save_figure(fig, out_dir / "histograms_kde.png")


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot per numeric column (own axis, since scales differ)."""
    cols = df.select_dtypes("number").columns
    n = len(cols)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Boxplots")
    fig.tight_layout()
    save_figure(fig, out_dir / "boxplots.png")


def plot_qq(df: pd.DataFrame, out_dir: Path) -> None:
    """QQ plot per numeric column against the normal distribution."""
    cols = df.select_dtypes("number").columns
    n = len(cols)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    for ax, col in zip(axes.ravel(), cols):
        stats.probplot(df[col].dropna(), dist="norm", plot=ax)
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("QQ Plots (vs Normal)")
    fig.tight_layout()
    save_figure(fig, out_dir / "qq_plots.png")


def plot_pairplot(df: pd.DataFrame, out_dir: Path) -> None:
    """Pair plot over all numeric columns (feasible: <= 12 columns here)."""
    g = sns.pairplot(df.select_dtypes("number"), corner=True, plot_kws={"s": 18})
    g.figure.suptitle("Pair Plot", y=1.01)
    ensure_dir(out_dir)
    g.figure.savefig(out_dir / "pairplot.png", dpi=130, bbox_inches="tight")
    plt.close(g.figure)


def plot_correlation_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    """Pearson / Spearman / Kendall correlation heatmaps side by side."""
    methods = ["pearson", "spearman", "kendall"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0 * 3, 5.5))
    for ax, method in zip(axes, methods):
        corr = df.select_dtypes("number").corr(method=method)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    annot_kws={"size": 6}, cbar=False)
        ax.set_title(f"{method.capitalize()} correlation")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(fig, out_dir / "correlation_heatmaps.png")


def run_all_visualizations(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate every required EDA figure for one dataset."""
    ensure_dir(out_dir)
    plot_histograms(df, out_dir)
    plot_boxplots(df, out_dir)
    plot_qq(df, out_dir)
    plot_pairplot(df, out_dir)
    plot_correlation_heatmaps(df, out_dir)
