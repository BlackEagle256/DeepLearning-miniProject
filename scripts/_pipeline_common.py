"""Shared library for the PER-DATASET pipeline scripts.

The assignment appendix requires: "har dataset bayad pipeline jodagane
dashte bashad (script-haye mostaghel)" - each dataset must have its own
independent pipeline / standalone script. Concretely this means, for every
phase (EDA, Pipeline 1-4, tuning, XAI, uncertainty, learning curves,
statistical tests), Dataset_0136 / Dataset_0172 / Dataset_3772 each get
their OWN runnable script under ``scripts/dataset_XXXX/``.

To keep that requirement without duplicating logic three times, all the
actual work lives in the functions below (one function per phase, taking a
single ``dataset_name``); the per-dataset scripts are thin ~5-line entry
points that just call these with a hard-coded dataset name. Only a couple
of steps are legitimately CROSS-dataset by design (Top-3 model selection
needs all three datasets' Pipeline-1 results first) - those live in
top-level ``scripts/select_top3.py`` and ``scripts/describe_top3.py``.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import get_base_seed, get_path, load_config
from src.data.loader import DatasetBundle, load_dataset
from src.evaluation import aggregate_fold_results, select_top3
from src.utils.io import save_table

# Level-A "dedicated analysis" outputs used by every post-Pipeline-1/2 phase
# (tuning, XAI, uncertainty, learning curves, stat tests): the 5 outputs
# shared by all three datasets, PLUS - for Dataset_0136 only - Temperature
# and Strain, which the assignment explicitly calls out for dedicated
# per-dataset analysis even though they are not shared across datasets.
_DATASET_0136_EXTRA_TARGETS = ("Temperature (°C)", "Strain")


def analysis_targets(bundle: DatasetBundle) -> list[str]:
    """Level-A target list for tuning/XAI/uncertainty/learning-curve phases."""
    cfg = load_config()
    targets = [c for c in cfg["shared_outputs"] if c in bundle.Y.columns]
    for extra in _DATASET_0136_EXTRA_TARGETS:
        if extra in bundle.Y.columns and extra not in targets:
            targets.append(extra)
    return targets


# ---------------------------------------------------------------------------
# Phase 2 - EDA (one dataset)
# ---------------------------------------------------------------------------


def eda_for_dataset(dataset_name: str) -> None:
    from src.eda.outliers import outlier_report
    from src.eda.statistics import (
        correlation_tests,
        descriptive_statistics,
        kendall_ci_table,
        normality_tests,
    )
    from src.eda.visualization import run_all_visualizations

    bundle = load_dataset(dataset_name)
    full = pd.concat([bundle.X, bundle.Y], axis=1)
    out_dir = get_path("results_dir") / "eda" / dataset_name

    run_all_visualizations(full, out_dir)
    save_table(descriptive_statistics(full), out_dir / "descriptive_statistics.csv", index=True)
    save_table(normality_tests(full), out_dir / "normality_tests.csv")
    save_table(correlation_tests(full), out_dir / "correlation_tests.csv")
    save_table(kendall_ci_table(full), out_dir / "kendall_confidence_intervals.csv")
    save_table(outlier_report(full), out_dir / "outlier_report.csv", index=True)

    if bundle.dropped_constant_features:
        print(f"[{dataset_name}] constant features dropped for modelling: "
              f"{bundle.dropped_constant_features}")
    print(f"[{dataset_name}] EDA artifacts written to {out_dir}")


# ---------------------------------------------------------------------------
# Phase 5 - Discretization (one dataset)
# ---------------------------------------------------------------------------


def discretize_dataset(dataset_name: str) -> None:
    from src.data.discretization import build_discrete_dataset

    df = build_discrete_dataset(dataset_name)
    print(f"[discretization] {dataset_name}: saved {df.shape[0]} rows -> data/discrete/")


# ---------------------------------------------------------------------------
# Pipelines 1-4 (one dataset)
# ---------------------------------------------------------------------------


def run_pipeline_for_dataset(
    dataset_name: str,
    mode: str,
    discrete: bool,
    pipeline_id: int,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Run ONE of the four project pipelines for ONE dataset only.

    mode     : 'single' or 'multi'
    discrete : False -> Original dataset, True -> Discrete-Input dataset
    models   : restrict to a subset (Top-3 for pipelines 3 and 4)
    """
    from src.pipelines.multi_output import run_multi_output
    from src.pipelines.single_output import run_single_output

    bundle = load_dataset(dataset_name, discrete=discrete)
    exp_name = f"pipeline{pipeline_id}_{dataset_name}"
    if mode == "single":
        fold_df = run_single_output(bundle, models=models, experiment_name=exp_name)
    else:
        fold_df = run_multi_output(bundle, models=models, experiment_name=exp_name)

    out_dir = get_path("results_dir") / f"pipeline{pipeline_id}" / dataset_name
    save_table(fold_df, out_dir / "fold_results.csv")

    agg = aggregate_fold_results(fold_df, group_cols=["dataset", "mode", "target", "model"])
    save_table(agg, out_dir / "aggregated_results.csv")
    print(f"[pipeline {pipeline_id}] {dataset_name}: results saved to {out_dir}")
    return agg


# ---------------------------------------------------------------------------
# Top-3 selection - inherently CROSS-dataset (needs all 3 datasets' Pipeline-1
# results first). Lives here as a shared helper; the actual entry point is
# the standalone scripts/select_top3.py.
# ---------------------------------------------------------------------------


def compute_top3_from_pipeline1() -> pd.DataFrame:
    """Read every dataset's Pipeline-1 per-dataset results and rank models."""
    cfg = load_config()
    frames = []
    for name in cfg["datasets"]:
        path = get_path("results_dir") / "pipeline1" / name / "aggregated_results.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run pipeline 1 for '{name}' first "
                f"(scripts/{name.lower()}/run_pipeline1_single_original.py)."
            )
        frames.append(pd.read_csv(path))
    agg = pd.concat(frames, ignore_index=True)

    per_model = agg.groupby("model", as_index=False)[
        ["test_r2_mean", "gap_r2", "stability_test_r2_std"]
    ].mean()
    ranked = select_top3(per_model)
    save_table(ranked, get_path("results_dir") / "top3_models.csv")
    return ranked


def run_pipeline_all_datasets(
    mode: str, discrete: bool, pipeline_id: int, models: list[str] | None = None
) -> pd.DataFrame:
    """Convenience-only orchestrator: loop ``run_pipeline_for_dataset`` over
    every configured dataset. NOT the assignment's required deliverable by
    itself (that's the per-dataset scripts under ``scripts/dataset_XXXX/``);
    this just saves re-typing three commands when you want to run all three
    at once (e.g. from the Makefile/tasks.ps1 ``all`` target)."""
    cfg = load_config()
    aggs = [
        run_pipeline_for_dataset(name, mode, discrete, pipeline_id, models=models)
        for name in cfg["datasets"]
    ]
    return pd.concat(aggs, ignore_index=True)


def read_top3_models() -> list[str]:
    """Read the cached Top-3 selection, computing it if missing."""
    top3_path = get_path("results_dir") / "top3_models.csv"
    if top3_path.exists():
        return pd.read_csv(top3_path)["model"].head(3).tolist()
    ranked = compute_top3_from_pipeline1()
    print("[top3] multi-criteria ranking:\n", ranked[["rank", "model", "composite_score"]].head(5))
    return ranked["model"].head(3).tolist()


# ---------------------------------------------------------------------------
# Phase 4 - Hyperparameter tuning (one dataset)
#   Stage 1: Random Search for EVERY model (assignment: "Random Search baraye
#            hame model-ha").
#   Stage 2: Optuna Bayesian optimization + Nested-CV honest generalization
#            check, Top-3 models only.
#   Grid Search is forbidden and never implemented.
# ---------------------------------------------------------------------------


def tuning_for_dataset(dataset_name: str, top3: list[str]) -> None:
    from src.models.registry import available_models, build_model
    from src.pipelines.cross_validation import nested_cv_score
    from src.tuning.hyperparameter import optuna_tune, random_search

    seed = get_base_seed()
    bundle = load_dataset(dataset_name)
    targets = analysis_targets(bundle)
    all_models = [m for m in load_config()["models"] if m in set(available_models())]

    out_dir = get_path("results_dir") / "tuning" / dataset_name

    # ---- Stage 1: Random Search, every model, every target ----
    stage1_rows = []
    for target in targets:
        y = bundle.Y[target]
        for model_name in all_models:
            rs = random_search(model_name, bundle.X, y, seed=seed)
            stage1_rows.append(
                {
                    "dataset": dataset_name,
                    "target": target,
                    "model": model_name,
                    "random_search_best_rmse": -rs.best_score_,
                    "random_search_best_params": str(rs.best_params_),
                }
            )
            print(f"[tuning/random-search] {dataset_name} / {target} / {model_name} done")
    save_table(pd.DataFrame(stage1_rows), out_dir / "random_search_all_models.csv")

    # ---- Stage 2: Optuna (Top-3 only) + Nested-CV honest generalization ----
    from src.models.registry import SEARCH_SPACES

    stage2_rows = []
    for target in targets:
        y = bundle.Y[target]
        for model_name in top3:
            study = optuna_tune(model_name, bundle.X, y, seed=seed)
            nested = nested_cv_score(
                build_model(model_name, seed=seed),
                SEARCH_SPACES.get(model_name, {}),
                bundle.X,
                y,
                seed=seed,
            )
            stage2_rows.append(
                {
                    "dataset": dataset_name,
                    "target": target,
                    "model": model_name,
                    "optuna_best_rmse": -study.best_value,
                    "optuna_best_params": str(study.best_params),
                    "nested_cv_test_rmse_mean": nested["test_rmse"].mean(),
                    "nested_cv_test_rmse_std": nested["test_rmse"].std(),
                    "nested_cv_test_r2_mean": nested["test_r2"].mean(),
                }
            )
            print(f"[tuning/optuna+nested-cv] {dataset_name} / {target} / {model_name} done")
    save_table(pd.DataFrame(stage2_rows), out_dir / "top3_optuna_nested_cv.csv")
    print(f"[tuning] {dataset_name}: results saved to {out_dir}")


# ---------------------------------------------------------------------------
# Phase 6 - XAI (one dataset, Top-3 models)
# ---------------------------------------------------------------------------


def xai_for_dataset(dataset_name: str, top3: list[str]) -> None:
    from src.models.registry import build_model
    from src.xai.importance import compute_permutation_importance, tree_feature_importance
    from src.xai.interpretation import summarize_shap_interpretation
    from src.xai.lime_analysis import lime_explain_samples
    from src.xai.shap_analysis import compute_shap_values, generate_all_shap_plots

    seed = get_base_seed()
    bundle = load_dataset(dataset_name)
    targets = analysis_targets(bundle)

    interp_rows = []
    for target in targets:
        y = bundle.Y[target]
        for model_name in top3:
            pipe = build_model(model_name, seed=seed)
            pipe.fit(bundle.X.to_numpy(), y.to_numpy())

            safe_target = target.replace("/", "_").replace(" ", "_")
            out_dir = get_path("results_dir") / "xai" / dataset_name / model_name / safe_target

            generate_all_shap_plots(pipe, bundle.X, out_dir)
            lime_explain_samples(pipe, bundle.X, out_dir, sample_indices=[0, 1])

            perm = compute_permutation_importance(pipe, bundle.X, y, seed=seed)
            save_table(perm, out_dir / "permutation_importance.csv")

            try:
                tree_imp = tree_feature_importance(pipe, bundle.feature_names)
                save_table(tree_imp, out_dir / "tree_feature_importance.csv")
            except TypeError:
                pass

            # Automated scientific-interpretation summary (Phase 6 requires
            # explaining the plots, not just drawing them): strongest driver,
            # effect sign, linearity, interaction strength, physical-sanity flag.
            sv = compute_shap_values(pipe, bundle.X)
            summary = summarize_shap_interpretation(sv, bundle.X, dataset_name, model_name, target)
            save_table(summary, out_dir / "interpretation_summary.csv")
            interp_rows.append(summary)

            print(f"[xai] {dataset_name} / {target} / {model_name} done")

    all_interp = pd.concat(interp_rows, ignore_index=True)
    save_table(all_interp, get_path("results_dir") / "xai" / dataset_name / "interpretation_summary_all.csv")


# ---------------------------------------------------------------------------
# Phase 7 - Uncertainty Quantification (one dataset)
# ---------------------------------------------------------------------------


def uncertainty_for_dataset(dataset_name: str, top3: list[str]) -> None:
    import numpy as np
    from sklearn.model_selection import KFold

    from src.evaluation import bootstrap_prediction_interval, gpr_prediction_interval, interval_metrics
    from src.models.registry import build_model

    seed = get_base_seed()
    n_splits = load_config()["cross_validation"]["n_splits"]
    models = sorted(set(top3) | {"gpr"})

    bundle = load_dataset(dataset_name)
    X = bundle.X.to_numpy()
    targets = analysis_targets(bundle)

    rows = []
    for target in targets:
        y = bundle.Y[target].to_numpy()
        for model_name in models:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            y_true_all, lo_all, hi_all = [], [], []
            for tr, te in kf.split(X):
                pipe = build_model(model_name, seed=seed)
                pipe.fit(X[tr], y[tr])
                if model_name == "gpr":
                    _, lo, hi = gpr_prediction_interval(pipe, X[te])
                else:
                    _, lo, hi = bootstrap_prediction_interval(
                        build_model(model_name, seed=seed), X[tr], y[tr], X[te], seed=seed
                    )
                y_true_all.append(y[te]); lo_all.append(lo); hi_all.append(hi)
            metrics = interval_metrics(
                np.concatenate(y_true_all), np.concatenate(lo_all), np.concatenate(hi_all)
            )
            rows.append({"dataset": dataset_name, "target": target, "model": model_name, **metrics})
            print(f"[uncertainty] {dataset_name} / {target} / {model_name}: {metrics}")

    save_table(pd.DataFrame(rows), get_path("results_dir") / "uncertainty" / dataset_name / "prediction_intervals.csv")


# ---------------------------------------------------------------------------
# Overfitting control - learning curves (one dataset)
# ---------------------------------------------------------------------------


def learning_curves_for_dataset(dataset_name: str, top3: list[str]) -> None:
    from src.evaluation import compute_learning_curve, plot_learning_curve
    from src.models.registry import build_model
    from src.utils.io import save_figure

    seed = get_base_seed()
    bundle = load_dataset(dataset_name)
    targets = analysis_targets(bundle)

    for target in targets:
        for model_name in top3:
            pipe = build_model(model_name, seed=seed)
            curve = compute_learning_curve(pipe, bundle.X, bundle.Y[target], seed=seed)
            safe_target = target.replace("/", "_").replace(" ", "_")
            out_dir = get_path("results_dir") / "learning_curves" / dataset_name
            save_table(curve, out_dir / f"{model_name}__{safe_target}.csv")
            fig = plot_learning_curve(curve, f"{dataset_name} | {model_name} | {target}")
            save_figure(fig, out_dir / f"{model_name}__{safe_target}.png")
            print(f"[learning-curve] {dataset_name} / {target} / {model_name} done")


# ---------------------------------------------------------------------------
# Phase 7 - Statistical comparisons (one dataset)
#   * Friedman + Nemenyi post-hoc across models (per-fold test RMSE).
#   * Wilcoxon signed-rank: Single-output vs Multi-output, paired per target.
# ---------------------------------------------------------------------------


def stat_tests_for_dataset(dataset_name: str) -> None:
    from src.evaluation import friedman_test, nemenyi_posthoc, wilcoxon_pairwise

    results_dir = get_path("results_dir")
    p1 = pd.read_csv(results_dir / "pipeline1" / dataset_name / "fold_results.csv")
    p2 = pd.read_csv(results_dir / "pipeline2" / dataset_name / "fold_results.csv")
    out_dir = results_dir / "statistical_tests" / dataset_name

    # ---- Friedman + Nemenyi across models (single-output) ----
    matrix = p1.pivot_table(
        index=["target", "seed", "fold"], columns="model", values="test_rmse"
    ).dropna()
    friedman_res = friedman_test(matrix)
    save_table(pd.DataFrame([{"dataset": dataset_name, **friedman_res}]), out_dir / "friedman_across_models.csv")
    nemenyi = nemenyi_posthoc(matrix)
    nemenyi.index = nemenyi.columns = matrix.columns
    save_table(nemenyi, out_dir / "nemenyi.csv", index=True)

    # ---- Wilcoxon: Single vs Multi output, paired PER TARGET/seed/fold ----
    # (see module docstring in the original stats script for why the
    # comparison must use per-target RMSE columns, not the raw aggregate).
    per_target_cols = [c for c in p2.columns if c.startswith("test_") and c.endswith("__rmse")]
    multi_long = p2.melt(
        id_vars=["dataset", "model", "seed", "fold"],
        value_vars=per_target_cols,
        var_name="target",
        value_name="multi_rmse",
    )
    multi_long["target"] = multi_long["target"].str.removeprefix("test_").str.removesuffix("__rmse")

    single_long = (
        p1.groupby(["dataset", "model", "target", "seed", "fold"])["test_rmse"]
        .mean()
        .rename("single_rmse")
        .reset_index()
    )
    paired = single_long.merge(multi_long, on=["dataset", "model", "target", "seed", "fold"], how="inner")

    wilcoxon_rows = []
    for model, grp in paired.groupby("model"):
        res = wilcoxon_pairwise(grp["single_rmse"], grp["multi_rmse"])
        wilcoxon_rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "n_paired_target_fold_obs": len(grp),
                "mean_rmse_single": grp["single_rmse"].mean(),
                "mean_rmse_multi": grp["multi_rmse"].mean(),
                **res,
            }
        )
    save_table(pd.DataFrame(wilcoxon_rows), out_dir / "wilcoxon_single_vs_multi.csv")
    print(f"[stats] {dataset_name}: tables saved to {out_dir}")
