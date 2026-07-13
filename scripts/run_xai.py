"""Phase 6 - XAI: SHAP (global + local), LIME, permutation and tree
importances for the Top-3 models on every dataset / shared target.

Figures land in results/xai/<dataset>/<model>/<target>/ and can be logged
to MLflow as artifacts. Remember: the report must INTERPRET these plots
(effect direction, linearity, interactions, physical consistency with the
friction process), not just show them.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models
from src.config import get_base_seed, get_path, load_config
from src.data.loader import load_all_datasets
from src.models.registry import build_model
from src.utils.io import save_table
from src.xai.importance import compute_permutation_importance, tree_feature_importance
from src.xai.lime_analysis import lime_explain_samples
from src.xai.shap_analysis import generate_all_shap_plots


def main() -> None:
    cfg = load_config()
    seed = get_base_seed()
    top3 = read_top3_models()
    bundles = load_all_datasets(discrete=False)

    for ds_name, bundle in bundles.items():
        shared_targets = [c for c in cfg["shared_outputs"] if c in bundle.Y.columns]
        for target in shared_targets:
            y = bundle.Y[target]
            for model_name in top3:
                pipe = build_model(model_name, seed=seed)
                pipe.fit(bundle.X.to_numpy(), y.to_numpy())

                safe_target = target.replace("/", "_").replace(" ", "_")
                out_dir = get_path("results_dir") / "xai" / ds_name / model_name / safe_target

                # SHAP: summary / beeswarm / waterfall / dependence /
                # interaction / feature importance.
                generate_all_shap_plots(pipe, bundle.X, out_dir)

                # LIME on selected local samples.
                lime_explain_samples(pipe, bundle.X, out_dir, sample_indices=[0, 1])

                # Permutation importance (model-agnostic).
                perm = compute_permutation_importance(pipe, bundle.X, y, seed=seed)
                save_table(perm, out_dir / "permutation_importance.csv")

                # Tree-based importance where applicable.
                try:
                    tree_imp = tree_feature_importance(pipe, bundle.feature_names)
                    save_table(tree_imp, out_dir / "tree_feature_importance.csv")
                except TypeError:
                    pass
                print(f"[xai] {ds_name} / {target} / {model_name} done")


if __name__ == "__main__":
    main()
