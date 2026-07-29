"""Multi-criteria Top-3 model selection.

This is the ONE legitimately cross-dataset step in the whole project: the
assignment asks to run every model on the Original datasets, THEN pick one
Top-3 list and reuse it everywhere else (discrete datasets, tuning, XAI,
uncertainty) "to reduce run volume and simulation cost". That selection
needs all three datasets' Pipeline-1 results, so unlike every other phase it
cannot be a per-dataset independent script.

Prerequisite: run Pipeline 1 for all three datasets first, e.g.
    python scripts/Dataset_0136/run_pipeline1_single_original.py
    python scripts/Dataset_0172/run_pipeline1_single_original.py
    python scripts/Dataset_3772/run_pipeline1_single_original.py

Criteria (config.yaml: top3_selection.weights): accuracy (R2), the
Train/Test generalization gap, fold stability, uncertainty (PI width) and
interpretability - accuracy alone never decides the winner.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import compute_top3_from_pipeline1


def main() -> None:
    ranked = compute_top3_from_pipeline1()
    print("[top3] multi-criteria ranking (all 3 datasets combined):")
    print(ranked[["rank", "model", "test_r2_mean", "gap_r2", "stability_test_r2_std", "composite_score"]])
    print("\nTop-3:", ranked["model"].head(3).tolist())


if __name__ == "__main__":
    main()
