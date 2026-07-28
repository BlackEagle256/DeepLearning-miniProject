"""Phase 2 - convenience wrapper: run EDA for all 3 datasets at once.

The assignment's actual per-dataset deliverable is the independent script
under ``scripts/<dataset>/run_eda.py`` (and the matching notebook under
``notebooks/``). This wrapper just loops over them so `make eda` /
`tasks.ps1 eda` can regenerate everything in one command; it adds no logic
of its own.

Usage:
    python scripts/run_eda.py --all
    python scripts/run_eda.py --dataset Dataset_0136
"""

import argparse

import _bootstrap  # noqa: F401

from _pipeline_common import eda_for_dataset
from src.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, help="Config dataset name")
    parser.add_argument("--all", action="store_true", help="Run for all datasets")
    args = parser.parse_args()

    names = list(load_config()["datasets"]) if args.all or not args.dataset else [args.dataset]
    for n in names:
        eda_for_dataset(n)
