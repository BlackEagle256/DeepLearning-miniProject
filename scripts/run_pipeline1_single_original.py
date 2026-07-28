"""Pipeline 1 - convenience wrapper: single-output regression, ALL 3 datasets.

The assignment's actual per-dataset deliverable is the independent script
under ``scripts/<dataset>/run_pipeline1_single_original.py``; this wrapper
just loops over the three of them for `make pipeline1` / `tasks.ps1
pipeline1`.
"""

from _pipeline_common import run_pipeline_all_datasets

if __name__ == "__main__":
    run_pipeline_all_datasets(mode="single", discrete=False, pipeline_id=1)
