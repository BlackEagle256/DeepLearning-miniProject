"""Phase 7 - statistical comparisons.

1. Friedman test + Nemenyi post-hoc across models (per-fold test RMSE).
2. Wilcoxon signed-rank: Single-output vs Multi-output per model (paired
   over identical seed/fold blocks) to answer "does Multi-output have a
   significant advantage?".
Inputs: fold_results.csv files of pipelines 1 and 2.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import get_path
from src.evaluation.statistical_tests import friedman_test, nemenyi_posthoc, wilcoxon_pairwise
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

    # ---- Wilcoxon: Single vs Multi output (paired per seed/fold) ----
    single = (
        p1.groupby(["dataset", "model", "seed", "fold"])["test_rmse"].mean().rename("single")
    )
    multi = (
        p2.groupby(["dataset", "model", "seed", "fold"])["test_rmse"].mean().rename("multi")
    )
    paired = pd.concat([single, multi], axis=1).dropna().reset_index()

    wilcoxon_rows = []
    for (ds_name, model), grp in paired.groupby(["dataset", "model"]):
        res = wilcoxon_pairwise(grp["single"], grp["multi"])
        wilcoxon_rows.append(
            {
                "dataset": ds_name,
                "model": model,
                "mean_rmse_single": grp["single"].mean(),
                "mean_rmse_multi": grp["multi"].mean(),
                **res,
            }
        )
    save_table(pd.DataFrame(wilcoxon_rows), out_dir / "wilcoxon_single_vs_multi.csv")
    print(f"[stats] tables saved to {out_dir}")


if __name__ == "__main__":
    main()
