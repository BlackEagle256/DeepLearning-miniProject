"""Phase 2 - EDA for Dataset_0136."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import eda_for_dataset

DATASET_NAME = "Dataset_0136"

if __name__ == "__main__":
    eda_for_dataset(DATASET_NAME)
