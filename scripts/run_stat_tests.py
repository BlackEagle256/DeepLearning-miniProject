"""Phase 7 - statistical comparisons.

1. Friedman test + Nemenyi post-hoc across models (per-fold test RMSE).
2. Wilcoxon signed-rank: Single-output vs Multi-output per model, compared
   PER TARGET (paired over identical target/seed/fold blocks) to answer
   "does Multi-output have a significant advantage?".

   IMPORTANT: this must compare RMSE for the SAME physical target on both
   sides. Pipeline 2's aggregate ``test_rmse`` is sqrt(mean of per-target
   MSE across all outputs); Pipeline 1's ``test_rmse`` is the RMSE of ONE
   target. By the RMS/QM-AM inequality, sqrt(mean(x_i^2)) >= mean(|x_i|)
   ALWAYS holds, so comparing those two aggregates directly makes Multi
   look uniformly worse regardless of the data (a scale artifact, not a
   finding). Instead we use the per-target columns
   (``test_{target}__rmse``) that ``cross_validate_model`` adds for
   multi-output runs, so both sides measure the same target in the same
   units.
Inputs: fold_results.csv files of pipelines 1 and 2.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import get_path
from src.evaluation import friedman_test, nemenyi_posthoc, wilcoxon_pairwise
from src.utils.io import save_table


def main() -> None:
    results_dir = get_path("results_dir")
    p1 = pd.read_csv(results_dir / "pipeline1" / "fold_results.csv")
    p2 = pd.read_csv(results_dir / "pipeline2" / "fold_results.csv")
    out_dir = results_dir / "statistical_tests"

    # ---- Friedman + Nemenyi across models (single-output, per dataset) ----
    friedman_rows = []
    for ds_name, grp in p1.groupby("dataset"):
        # blocks = (target, seed, fold); columns = models; values = test RMSE
        matrix = grp.pivot_table(
            index=["target", "seed", "fold"], columns="model", values="test_rmse"
        ).dropna()
        res = friedman_test(matrix)
        friedman_rows.append({"dataset": ds_name, **res})
        nemenyi = nemenyi_posthoc(matrix)
        nemenyi.index = nemenyi.columns = matrix.columns
        save_table(nemenyi, out_dir / f"nemenyi_{ds_name}.csv", index=True)
    save_table(pd.DataFrame(friedman_rows), out_dir / "friedman_across_models.csv")

    # ---- Wilcoxon: Single vs Multi output, paired PER TARGET/seed/fold ----
    per_target_cols = [
        c for c in p2.columns if c.startswith("test_") and c.endswith("__rmse")
    ]
    multi_long = p2.melt(
        id_vars=["dataset", "model", "seed", "fold"],
        value_vars=per_target_cols,
        var_name="target",
        value_name="multi_rmse",
    )
    multi_long["target"] = (
        multi_long["target"].str.removeprefix("test_").str.removesuffix("__rmse")
    )

    single_long = (
        p1.groupby(["dataset", "model", "target", "seed", "fold"])["test_rmse"]
        .mean()
        .rename("single_rmse")
        .reset_index()
    )
    paired = single_long.merge(
        multi_long, on=["dataset", "model", "target", "seed", "fold"], how="inner"
    )

    wilcoxon_rows = []
    for (ds_name, model), grp in paired.groupby(["dataset", "model"]):
        res = wilcoxon_pairwise(grp["single_rmse"], grp["multi_rmse"])
        wilcoxon_rows.append(
            {
                "dataset": ds_name,
                "model": model,
                "n_paired_target_fold_obs": len(grp),
                "mean_rmse_single": grp["single_rmse"].mean(),
                "mean_rmse_multi": grp["multi_rmse"].mean(),
                **res,
            }
        )
    save_table(pd.DataFrame(wilcoxon_rows), out_dir / "wilcoxon_single_vs_multi.csv")
    print(f"[stats] tables saved to {out_dir}")


if __name__ == "__main__":
    main()
