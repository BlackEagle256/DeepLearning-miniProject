"""Shared logic for the four pipeline entry points."""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import get_path, load_config
from src.data.loader import load_all_datasets
from src.evaluation.metrics import aggregate_fold_results
from src.evaluation.model_selection import select_top3
from src.utils.io import save_table


def run_pipeline(mode: str, discrete: bool, pipeline_id: int, models: list[str] | None = None) -> pd.DataFrame:
    """Run one of the four project pipelines over all datasets.

    mode      : 'single' or 'multi'
    discrete  : False -> Original datasets, True -> Discrete Input datasets
    models    : restrict to a subset (e.g. Top-3 for pipelines 3 and 4)
    """
    from src.pipelines.multi_output import run_multi_output
    from src.pipelines.single_output import run_single_output

    bundles = load_all_datasets(discrete=discrete)
    results = []
    for name, bundle in bundles.items():
        exp_name = f"pipeline{pipeline_id}_{name}"
        if mode == "single":
            df = run_single_output(bundle, models=models, experiment_name=exp_name)
        else:
            df = run_multi_output(bundle, models=models, experiment_name=exp_name)
        results.append(df)

    fold_df = pd.concat(results, ignore_index=True)
    out_dir = get_path("results_dir") / f"pipeline{pipeline_id}"
    save_table(fold_df, out_dir / "fold_results.csv")

    agg = aggregate_fold_results(fold_df, group_cols=["dataset", "mode", "target", "model"])
    save_table(agg, out_dir / "aggregated_results.csv")
    print(f"[pipeline {pipeline_id}] results saved to {out_dir}")
    return agg


def read_top3_models() -> list[str]:
    """Read the Top-3 selection produced from the Original-dataset runs.

    Falls back to computing it from pipeline-1 aggregates if the dedicated
    file does not exist yet.
    """
    top3_path = get_path("results_dir") / "top3_models.csv"
    if top3_path.exists():
        return pd.read_csv(top3_path)["model"].head(3).tolist()

    agg_path = get_path("results_dir") / "pipeline1" / "aggregated_results.csv"
    if not agg_path.exists():
        raise FileNotFoundError(
            "Run pipeline 1 first (Original datasets) so the Top-3 models can be selected."
        )
    agg = pd.read_csv(agg_path)
    per_model = agg.groupby("model", as_index=False)[
        ["test_r2_mean", "gap_r2", "stability_test_r2_std"]
    ].mean()
    ranked = select_top3(per_model)
    save_table(ranked, top3_path)
    print("[top3] multi-criteria ranking:\n", ranked[["rank", "model", "composite_score"]].head(5))
    return ranked["model"].head(3).tolist()


_ = load_config  # re-exported convenience
