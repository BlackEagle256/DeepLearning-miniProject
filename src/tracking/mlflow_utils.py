"""MLflow helpers - Phase 1 (MLOps) requirement.

Every experiment (parameters, per-seed metrics, aggregated metrics and
artifact tables) is logged to a local SQLite-backed MLflow tracking store
(``mlflow.db`` at the repo root). Launch the UI with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

(and take the screenshots required for the final report).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - keep repo importable without mlflow
    _MLFLOW_AVAILABLE = False


def _setup(experiment_name: str) -> bool:
    """Point MLflow at the local store and select/create the experiment.

    Recent MLflow versions (2.2x+) put the plain folder-based file store
    ("./mlruns") into maintenance mode and refuse to start unless
    MLFLOW_ALLOW_FILE_STORE=true is set. To avoid that entirely, tracking
    uses a local SQLite database (mlflow.db at the repo root) as the
    backend store. This works out of the box on Windows, macOS and Linux
    with no extra environment variables.
    """
    if not _MLFLOW_AVAILABLE:
        print("[mlflow] mlflow not installed - skipping tracking.")
        return False
    db_path = PROJECT_ROOT / "mlflow.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return True


def log_cv_results(
    experiment_name: str,
    run_name: str,
    fold_df: pd.DataFrame,
    params: dict,
) -> None:
    """Log one model run: params, aggregated metrics, and the raw fold table."""
    if not _setup(experiment_name):
        return

    metric_cols = [c for c in fold_df.columns if c.startswith(("train_", "test_"))]
    means = fold_df[metric_cols].mean()
    stds = fold_df[metric_cols].std()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for col in metric_cols:
            mlflow.log_metric(f"{col}_mean", float(means[col]))
            mlflow.log_metric(f"{col}_std", float(stds[col]))
        # Generalization gap = the project's primary overfitting signal.
        mlflow.log_metric("gap_r2", float(abs(means["train_r2"] - means["test_r2"])))
        mlflow.log_metric("gap_rmse", float(abs(means["train_rmse"] - means["test_rmse"])))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold_results.csv"
            fold_df.to_csv(path, index=False)
            mlflow.log_artifact(str(path))


def log_artifact_file(experiment_name: str, run_name: str, file_path: Path, params: dict | None = None) -> None:
    """Log a standalone artifact (figure, table, ...) as its own run."""
    if not _setup(experiment_name):
        return
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_artifact(str(file_path))
