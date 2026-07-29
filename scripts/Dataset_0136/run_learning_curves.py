"""Overfitting control - learning curves for Dataset_0136, Top-3 models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import learning_curves_for_dataset, read_top3_models

DATASET_NAME = "Dataset_0136"

if __name__ == "__main__":
    top3 = read_top3_models()
    learning_curves_for_dataset(DATASET_NAME, top3)
