"""Phase 5 - build the three Discrete Input datasets (CSV + mapping JSON)."""

import _bootstrap  # noqa: F401

from src.data.discretization import build_all_discrete_datasets

if __name__ == "__main__":
    build_all_discrete_datasets()
