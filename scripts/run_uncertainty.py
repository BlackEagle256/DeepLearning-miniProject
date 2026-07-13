"""Phase 7 - Uncertainty Quantification.

For every dataset / shared target:
  * GPR   -> analytical Prediction Interval (posterior std).
  * Top-3 -> Bootstrap Prediction Interval.
Reports mean PI width and empirical coverage (out-of-fold), which also
feed the multi-criteria model comparison.
"""

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from _pipeline_common import read_top3_models
from src.config import get_base_seed, get_path, load_config
from src.data.loader import load_all_datasets
from src.evaluation.uncertainty import (
    bootstrap_prediction_interval,
    gpr_prediction_interval,
    interval_metrics,
)
from src.models.registry import build_model
from src.utils.io import save_table


def main() -> None:
    cfg = load_config()
    seed = get_base_seed()
    n_splits = cfg["cross_validation"]["n_splits"]
    models = sorted(set(read_top3_models()) | {"gpr"})
    bundles = load_all_datasets(discrete=False)

    rows = []
    for ds_name, bundle in bundles.items():
        X = bundle.X.to_numpy()
        shared_targets = [c for c in cfg["shared_outputs"] if c in bundle.Y.columns]
        for target in shared_targets:
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
                rows.append({"dataset": ds_name, "target": target, "model": model_name, **metrics})
                print(f"[uncertainty] {ds_name} / {target} / {model_name}: {metrics}")

    save_table(pd.DataFrame(rows), get_path("results_dir") / "uncertainty" / "prediction_intervals.csv")


if __name__ == "__main__":
    main()
