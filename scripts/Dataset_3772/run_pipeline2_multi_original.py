"""Pipeline 2 - Multi-output regression on the ORIGINAL Dataset_3772 dataset (all models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import run_pipeline_for_dataset

DATASET_NAME = "Dataset_3772"

if __name__ == "__main__":
    run_pipeline_for_dataset(DATASET_NAME, mode="multi", discrete=False, pipeline_id=2)
