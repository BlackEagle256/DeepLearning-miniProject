"""Overfitting control: learning curves, ALL 3 datasets.

The actual per-dataset deliverable is
``scripts/<dataset>/run_learning_curves.py``.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import learning_curves_for_dataset, read_top3_models
from src.config import load_config


def main() -> None:
    top3 = read_top3_models()
    for ds_name in load_config()["datasets"]:
        learning_curves_for_dataset(ds_name, top3)


if __name__ == "__main__":
    main()
