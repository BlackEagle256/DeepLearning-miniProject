"""Phase 5 - Input discretization (Discrete Input datasets).

The assignment requires converting the continuous input variables into
discrete integer values (1, 2, 3, ...) to build three new secondary
datasets. Experimental inputs here take a small number of distinct set
points (e.g. Plunging Speed in {25, 31.5, 40}), so the natural encoding
is ORDINAL RANK MAPPING: for each input column the sorted unique values
are mapped to consecutive integers 1..k. This is monotone, lossless for
level-type variables and produces exactly the required integer codes.

Outputs are NEVER discretized - only inputs.
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import get_path, load_config
from src.data.loader import load_dataset
from src.utils.io import ensure_dir


def ordinal_discretize(series: pd.Series) -> tuple[pd.Series, dict]:
    """Map the sorted unique values of a column to integers 1..k.

    Returns the encoded series and the value->code mapping (kept for
    documentation / reproducibility).
    """
    unique_sorted = sorted(series.dropna().unique())
    mapping = {val: code for code, val in enumerate(unique_sorted, start=1)}
    return series.map(mapping).astype(int), {str(k): v for k, v in mapping.items()}


def build_discrete_dataset(name: str) -> pd.DataFrame:
    """Create the Discrete-Input version of one dataset and save it as CSV.

    The saved CSV keeps the same column layout (inputs + outputs) so that
    the same loader / pipelines can be reused via ``discrete=True``.
    A JSON file with the per-column mappings is saved next to it.
    """
    cfg = load_config()
    ds_cfg = cfg["datasets"][name]
    bundle = load_dataset(name, discrete=False)

    X_disc = pd.DataFrame(index=bundle.X.index)
    mappings: dict[str, dict] = {}
    for col in bundle.X.columns:
        X_disc[col], mappings[col] = ordinal_discretize(bundle.X[col])

    out = pd.concat([X_disc, bundle.Y], axis=1)

    out_dir = ensure_dir(get_path("discrete_data_dir"))
    csv_path = out_dir / f"{name}_discrete.csv"
    out.to_csv(csv_path, index=False)
    with open(out_dir / f"{name}_discrete_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    # Note: constant input columns were already removed by the loader for
    # this dataset; ds_cfg still documents the full original input list.
    _ = ds_cfg
    return out


def build_all_discrete_datasets() -> None:
    """Build the three Discrete Input datasets required by Phase 5."""
    for name in load_config()["datasets"]:
        df = build_discrete_dataset(name)
        print(f"[discretization] {name}: saved {df.shape[0]} rows -> data/discrete/")
