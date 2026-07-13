"""Pipeline 2 - Multi-output regression on the ORIGINAL datasets (all models)."""

from _pipeline_common import run_pipeline

if __name__ == "__main__":
    run_pipeline(mode="multi", discrete=False, pipeline_id=2)
