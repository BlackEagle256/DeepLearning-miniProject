"""Phase 7: statistical comparisons, ALL 3 datasets.

1. Friedman test + Nemenyi post-hoc across models (per-fold test RMSE).
2. Wilcoxon signed-rank: Single-output vs Multi-output, paired per target.

The actual per-dataset deliverable is ``scripts/<dataset>/run_stat_tests.py``.
Requires Pipelines 1 and 2 to have been run for every dataset first.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import stat_tests_for_dataset
from src.config import load_config


def main() -> None:
    for ds_name in load_config()["datasets"]:
        stat_tests_for_dataset(ds_name)


if __name__ == "__main__":
    main()
