"""Detailed statistical review of the Top-3 models (cross-dataset report).

Runs pandas ``.describe()`` on the Top-3 models' fold-level results (every
dataset, every target, every seed, every fold) and writes both the raw
describe() tables and a compact per-model/per-dataset summary, so the
Top-3 models can be inspected in detail rather than just ranked by a single
composite score.

Prerequisite: Pipeline 1 must have been run for all three datasets first
(this reads ``results/pipeline1/<dataset>/fold_results.csv``), and
``scripts/select_top3.py`` must have produced ``results/top3_models.csv``.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from _pipeline_common import read_top3_models
from src.config import get_path, load_config
from src.utils.io import ensure_dir, save_table

METRIC_COLS = [
    "train_r2", "test_r2", "train_rmse", "test_rmse",
    "train_nrmse", "test_nrmse", "train_mae", "test_mae",
]


def _load_all_pipeline1_folds() -> pd.DataFrame:
    cfg = load_config()
    frames = []
    for name in cfg["datasets"]:
        path = get_path("results_dir") / "pipeline1" / name / "fold_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run Pipeline 1 for '{name}' first.")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    top3 = read_top3_models()
    print(f"[describe-top3] Top-3 models: {top3}")

    folds = _load_all_pipeline1_folds()
    top3_folds = folds[folds["model"].isin(top3)].copy()
    top3_folds["gap_r2"] = (top3_folds["train_r2"] - top3_folds["test_r2"]).abs()

    out_dir = ensure_dir(get_path("results_dir") / "top3_review")

    # 1) pandas .describe() per model, across ALL datasets/targets/seeds/folds
    describe_per_model = (
        top3_folds.groupby("model")[METRIC_COLS + ["gap_r2"]]
        .describe()
    )
    describe_per_model.to_csv(out_dir / "describe_per_model.csv")
    print("\n=== .describe() per Top-3 model (all datasets/targets/folds/seeds) ===")
    print(describe_per_model)

    # 2) pandas .describe() per (model, dataset) - does behaviour hold across
    #    all 3 datasets or is it driven by one of them?
    describe_per_model_dataset = (
        top3_folds.groupby(["model", "dataset"])[METRIC_COLS + ["gap_r2"]]
        .describe()
    )
    describe_per_model_dataset.to_csv(out_dir / "describe_per_model_dataset.csv")

    # 3) Compact human-readable summary: mean +/- std of the key metrics,
    #    generalization gap, and fold stability, per model per dataset.
    summary = (
        top3_folds.groupby(["model", "dataset"])
        .agg(
            n_fold_obs=("test_r2", "size"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_nrmse_mean=("test_nrmse", "mean"),
            test_mae_mean=("test_mae", "mean"),
            gap_r2_mean=("gap_r2", "mean"),
        )
        .reset_index()
        .sort_values(["model", "dataset"])
    )
    save_table(summary, out_dir / "top3_summary_per_dataset.csv")

    # 4) Cross-dataset rollup per model (one row per Top-3 model).
    rollup = (
        top3_folds.groupby("model")
        .agg(
            n_fold_obs=("test_r2", "size"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            gap_r2_mean=("gap_r2", "mean"),
            gap_r2_std=("gap_r2", "std"),
        )
        .reset_index()
        .sort_values("test_r2_mean", ascending=False)
    )
    save_table(rollup, out_dir / "top3_rollup_all_datasets.csv")

    print("\n=== Compact per-(model, dataset) summary ===")
    print(summary.to_string(index=False))
    print("\n=== Cross-dataset rollup ===")
    print(rollup.to_string(index=False))
    print(f"\n[describe-top3] full tables saved to {out_dir}")


if __name__ == "__main__":
    main()
