"""Phase 4 - convenience wrapper: hyperparameter optimization, ALL 3 datasets.

Stage 1: Random Search for EVERY model (assignment requirement - not just
         the Top-3), per dataset per target.
Stage 2: Optuna Bayesian optimization + Nested-CV honest generalization
         check, Top-3 models only.
Grid Search is forbidden (small data) and intentionally unavailable.

The actual per-dataset deliverable is ``scripts/<dataset>/run_tuning.py``.
Results land under ``results/tuning/<dataset>/`` and are also logged where
applicable to MLflow.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models, tuning_for_dataset
from src.config import load_config


def main() -> None:
    top3 = read_top3_models()
    for ds_name in load_config()["datasets"]:
        tuning_for_dataset(ds_name, top3)


if __name__ == "__main__":
    main()
