"""Phase 4 - hyperparameter optimization for Dataset_0172 only.

Stage 1: Random Search for EVERY model (assignment requirement).
Stage 2: Optuna Bayesian optimization + Nested-CV honest generalization
         check, Top-3 models only. Grid Search is forbidden and unused.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models, tuning_for_dataset

DATASET_NAME = "Dataset_0172"

if __name__ == "__main__":
    top3 = read_top3_models()
    tuning_for_dataset(DATASET_NAME, top3)
