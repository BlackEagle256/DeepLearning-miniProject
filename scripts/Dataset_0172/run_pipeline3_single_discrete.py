"""Pipeline 3 - Single-output regression on the DISCRETE Dataset_0172 dataset (Top-3 only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models, run_pipeline_for_dataset

DATASET_NAME = "Dataset_0172"

if __name__ == "__main__":
    top3 = read_top3_models()
    print(f"[pipeline 3] {DATASET_NAME}: running Top-3 models on the discrete dataset: {top3}")
    run_pipeline_for_dataset(DATASET_NAME, mode="single", discrete=True, pipeline_id=3, models=top3)
