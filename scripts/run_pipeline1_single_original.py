"""Pipeline 1 - Single-output regression on the ORIGINAL datasets (all models)."""

from _pipeline_common import run_pipeline

if __name__ == "__main__":
    run_pipeline(mode="single", discrete=False, pipeline_id=1)
