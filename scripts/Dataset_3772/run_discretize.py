"""Phase 5 - build the Discrete-Input version of Dataset_3772."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import discretize_dataset

DATASET_NAME = "Dataset_3772"

if __name__ == "__main__":
    discretize_dataset(DATASET_NAME)
