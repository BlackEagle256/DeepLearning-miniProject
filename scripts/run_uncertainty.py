"""Phase 7: Uncertainty Quantification, ALL 3 datasets.

For every dataset / Level-A target: GPR gets the analytical Prediction
Interval, the Top-3 models get the Bootstrap Prediction Interval.

The actual per-dataset deliverable is ``scripts/<dataset>/run_uncertainty.py``.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models, uncertainty_for_dataset
from src.config import load_config


def main() -> None:
    top3 = read_top3_models()
    for ds_name in load_config()["datasets"]:
        uncertainty_for_dataset(ds_name, top3)


if __name__ == "__main__":
    main()
