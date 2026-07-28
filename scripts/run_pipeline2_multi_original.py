"""Pipeline 2 - convenience wrapper: multi-output regression, ALL 3 datasets.

The assignment's actual per-dataset deliverable is the independent script
under ``scripts/<dataset>/run_pipeline2_multi_original.py``; this wrapper
just loops over the three of them for `make pipeline2` / `tasks.ps1
pipeline2`.
"""

from _pipeline_common import run_pipeline_all_datasets

if __name__ == "__main__":
    run_pipeline_all_datasets(mode="multi", discrete=False, pipeline_id=2)
