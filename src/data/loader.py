"""Dataset loading utilities.

Key rules encoded here (from the assignment):
  * ``No.`` is only a sample index (extra column) and is always dropped.
  * ``Composite Volume Fraction (%)`` is binary (0/1). It is constant in
    Dataset_0136 (all 0) and Dataset_3772 (all 1); constant features carry
    no predictive information, must not show up as meaningful in feature
    importance, and are dropped to avoid numerical issues. It varies only
    in Dataset_0172, where it is kept.
  * NO ROW IS EVER REMOVED. Outliers are only flagged/reported (EDA phase),
    because in mechanics they may reflect true physical behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import get_path, load_config


@dataclass
class DatasetBundle:
    """Container holding one loaded dataset and its metadata."""

    name: str
    X: pd.DataFrame                     # input features (constants removed)
    Y: pd.DataFrame                     # all Level-A outputs of this dataset
    dropped_constant_features: list[str] = field(default_factory=list)
    is_discrete: bool = False

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def output_names(self) -> list[str]:
        return list(self.Y.columns)

    def shared_Y(self) -> pd.DataFrame:
        """Return only the outputs shared by all datasets (Level-B comparison)."""
        shared = load_config()["shared_outputs"]
        return self.Y[[c for c in shared if c in self.Y.columns]]


def _drop_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns with a single unique value and report them."""
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    return X.drop(columns=constant), constant


def load_dataset(name: str, discrete: bool = False) -> DatasetBundle:
    """Load one dataset by config name (e.g. ``Dataset_0136``).

    Parameters
    ----------
    name:
        Key inside the ``datasets`` section of ``configs/config.yaml``.
    discrete:
        If True, load the generated Discrete-Input version from
        ``data/discrete`` instead of the original Excel file.
    """
    cfg = load_config()
    if name not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(cfg['datasets'])}")
    ds_cfg = cfg["datasets"][name]

    if discrete:
        path = get_path("discrete_data_dir") / f"{name}_discrete.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python scripts/make_discrete_datasets.py' first."
            )
        df = pd.read_csv(path)
    else:
        path = get_path("raw_data_dir") / ds_cfg["file"]
        df = pd.read_excel(path)

    # Drop the pure index column ("No.") - it is an extra, non-predictive column.
    index_col = ds_cfg.get("index_column")
    if index_col and index_col in df.columns:
        df = df.drop(columns=[index_col])

    X = df[[c for c in ds_cfg["inputs"] if c in df.columns]].copy()
    Y = df[[c for c in ds_cfg["outputs"] if c in df.columns]].copy()

    X, dropped = _drop_constant_columns(X)

    return DatasetBundle(
        name=name, X=X, Y=Y, dropped_constant_features=dropped, is_discrete=discrete
    )


def load_all_datasets(discrete: bool = False) -> dict[str, DatasetBundle]:
    """Load every dataset declared in the configuration."""
    cfg = load_config()
    return {name: load_dataset(name, discrete=discrete) for name in cfg["datasets"]}
