"""Small I/O helpers (result folders, saving figures and tables)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a results table as CSV, creating parent folders as needed."""
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def save_figure(fig, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it to free memory."""
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
