"""Pipeline 4 - Multi-output regression on the DISCRETE datasets (Top-3 only)."""

from _pipeline_common import read_top3_models, run_pipeline

if __name__ == "__main__":
    top3 = read_top3_models()
    print(f"[pipeline 4] running Top-3 models on discrete datasets: {top3}")
    run_pipeline(mode="multi", discrete=True, pipeline_id=4, models=top3)
