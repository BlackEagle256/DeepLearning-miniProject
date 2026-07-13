"""Phase 2 - run the full EDA for one or all datasets.

Usage:
    python scripts/run_eda.py --all
    python scripts/run_eda.py --dataset Dataset_0136
Outputs: figures + CSV tables under results/eda/<dataset>/.
"""

import argparse

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from src.config import get_path, load_config
from src.data.loader import load_dataset
from src.eda.outliers import outlier_report
from src.eda.statistics import (
    correlation_tests,
    descriptive_statistics,
    kendall_ci_table,
    normality_tests,
)
from src.eda.visualization import run_all_visualizations
from src.utils.io import save_table


def run_for(name: str) -> None:
    bundle = load_dataset(name)
    # EDA is done on the full table (inputs + outputs) before dropping
    # constants, so the report can DOCUMENT the constant feature too.
    import pandas as pd

    full = pd.concat([bundle.X, bundle.Y], axis=1)
    out_dir = get_path("results_dir") / "eda" / name

    run_all_visualizations(full, out_dir)
    save_table(descriptive_statistics(full), out_dir / "descriptive_statistics.csv", index=True)
    save_table(normality_tests(full), out_dir / "normality_tests.csv")
    save_table(correlation_tests(full), out_dir / "correlation_tests.csv")
    save_table(kendall_ci_table(full), out_dir / "kendall_confidence_intervals.csv")
    save_table(outlier_report(full), out_dir / "outlier_report.csv", index=True)

    if bundle.dropped_constant_features:
        print(f"[{name}] constant features dropped for modelling: "
              f"{bundle.dropped_constant_features}")
    print(f"[{name}] EDA artifacts written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, help="Config dataset name")
    parser.add_argument("--all", action="store_true", help="Run for all datasets")
    args = parser.parse_args()

    names = list(load_config()["datasets"]) if args.all or not args.dataset else [args.dataset]
    for n in names:
        run_for(n)
