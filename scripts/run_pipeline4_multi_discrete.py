"""Pipeline 4: multi-output, DISCRETE datasets, ALL 3 (Top-3 only).

The actual per-dataset deliverable is
``scripts/<dataset>/run_pipeline4_multi_discrete.py``.
"""

from _pipeline_common import read_top3_models, run_pipeline_all_datasets

if __name__ == "__main__":
    top3 = read_top3_models()
    print(f"[pipeline 4] running Top-3 models on discrete datasets: {top3}")
    run_pipeline_all_datasets(mode="multi", discrete=True, pipeline_id=4, models=top3)
