"""Phase 6 - XAI (SHAP + LIME + importances + interpretation) for Dataset_0172 only, Top-3 models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models, xai_for_dataset

DATASET_NAME = "Dataset_0172"

if __name__ == "__main__":
    top3 = read_top3_models()
    xai_for_dataset(DATASET_NAME, top3)
