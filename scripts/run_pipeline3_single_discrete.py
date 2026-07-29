"""Pipeline 3: single-output, DISCRETE datasets, ALL 3.

Only the Top-3 models (multi-criteria selection from the Original runs) are
executed, per the assignment ("to reduce runtime"). The actual per-dataset
deliverable is ``scripts/<dataset>/run_pipeline3_single_discrete.py``.
"""

from _pipeline_common import read_top3_models, run_pipeline_all_datasets

if __name__ == "__main__":
    top3 = read_top3_models()
    print(f"[pipeline 3] running Top-3 models on discrete datasets: {top3}")
    run_pipeline_all_datasets(mode="single", discrete=True, pipeline_id=3, models=top3)
