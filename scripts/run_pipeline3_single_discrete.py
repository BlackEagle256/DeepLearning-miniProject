"""Pipeline 3 - Single-output regression on the DISCRETE datasets.

Only the Top-3 models (multi-criteria selection from the Original runs)
are executed, per the assignment ("to reduce runtime").
"""

from _pipeline_common import read_top3_models, run_pipeline

if __name__ == "__main__":
    top3 = read_top3_models()
    print(f"[pipeline 3] running Top-3 models on discrete datasets: {top3}")
    run_pipeline(mode="single", discrete=True, pipeline_id=3, models=top3)
