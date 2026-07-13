"""Phase 4 - hyperparameter optimization.

Stage 1: Random Search for ALL models (per dataset/target) - already
         covered by the search spaces; this script focuses on Stage 2.
Stage 2: Bayesian optimization (Optuna) for the Top-3 models only.
Grid Search is forbidden (small data) and intentionally unavailable.

Results (best params + best CV score) are saved to results/tuning/ and
logged to MLflow.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from _pipeline_common import read_top3_models
from src.config import get_base_seed, get_path, load_config
from src.data.loader import load_all_datasets
from src.tuning.hyperparameter import optuna_tune, random_search
from src.utils.io import save_table


def main() -> None:
    cfg = load_config()
    seed = get_base_seed()
    top3 = read_top3_models()
    bundles = load_all_datasets(discrete=False)

    rows = []
    for ds_name, bundle in bundles.items():
        shared_targets = [c for c in cfg["shared_outputs"] if c in bundle.Y.columns]
        for target in shared_targets:
            y = bundle.Y[target]
            for model_name in top3:
                # Stage 1: random search (all models get this treatment).
                rs = random_search(model_name, bundle.X, y, seed=seed)
                # Stage 2: Bayesian optimization (Top-3 only).
                study = optuna_tune(model_name, bundle.X, y, seed=seed)
                rows.append(
                    {
                        "dataset": ds_name,
                        "target": target,
                        "model": model_name,
                        "random_search_best_rmse": -rs.best_score_,
                        "random_search_best_params": str(rs.best_params_),
                        "optuna_best_rmse": -study.best_value,
                        "optuna_best_params": str(study.best_params),
                    }
                )
                print(f"[tuning] {ds_name} / {target} / {model_name} done")

    df = pd.DataFrame(rows)
    save_table(df, get_path("results_dir") / "tuning" / "top3_tuning_results.csv")


if __name__ == "__main__":
    main()
