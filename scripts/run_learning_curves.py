"""Phase 8 support - learning curves for overfitting control.

Generates learning-curve figures for the Top-3 models on every dataset /
shared target (train vs CV RMSE as data grows).
"""

import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models
from src.config import get_base_seed, get_path, load_config
from src.data.loader import load_all_datasets
from src.evaluation.overfitting import compute_learning_curve, plot_learning_curve
from src.models.registry import build_model
from src.utils.io import save_figure, save_table


def main() -> None:
    cfg = load_config()
    seed = get_base_seed()
    top3 = read_top3_models()
    bundles = load_all_datasets(discrete=False)

    for ds_name, bundle in bundles.items():
        shared_targets = [c for c in cfg["shared_outputs"] if c in bundle.Y.columns]
        for target in shared_targets:
            for model_name in top3:
                pipe = build_model(model_name, seed=seed)
                curve = compute_learning_curve(pipe, bundle.X, bundle.Y[target], seed=seed)
                safe_target = target.replace("/", "_").replace(" ", "_")
                out_dir = get_path("results_dir") / "learning_curves" / ds_name
                save_table(curve, out_dir / f"{model_name}__{safe_target}.csv")
                fig = plot_learning_curve(curve, f"{ds_name} | {model_name} | {target}")
                save_figure(fig, out_dir / f"{model_name}__{safe_target}.png")
                print(f"[learning-curve] {ds_name} / {target} / {model_name} done")


if __name__ == "__main__":
    main()
