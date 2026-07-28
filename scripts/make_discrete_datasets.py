"""Phase 5 - convenience wrapper: build all three Discrete-Input datasets.

The actual per-dataset deliverable is
``scripts/<dataset>/run_discretize.py``.
"""

import _bootstrap  # noqa: F401

from src.data.discretization import build_all_discrete_datasets

if __name__ == "__main__":
    build_all_discrete_datasets()
