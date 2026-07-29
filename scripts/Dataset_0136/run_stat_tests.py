"""Phase 7 - statistical comparisons for Dataset_0136.

1. Friedman test + Nemenyi post-hoc across models (per-fold test RMSE).
2. Wilcoxon signed-rank: Single-output vs Multi-output, paired per target.
Requires Pipelines 1 and 2 to have been run for Dataset_0136 first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401

from _pipeline_common import stat_tests_for_dataset

DATASET_NAME = "Dataset_0136"

if __name__ == "__main__":
    stat_tests_for_dataset(DATASET_NAME)
