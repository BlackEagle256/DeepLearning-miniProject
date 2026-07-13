# Surrogate Models for Friction-Processed Composites (Small Data)

**Deep Learning course project — Dr. Bahaghighat, Imam Khomeini International University**

Development and validation of reliable, interpretable, and **overfitting-resistant**
surrogate models (XAI-based) predicting mechanical/functional properties of
friction-processed composites from three small experimental datasets
(36 / 72 / 36 samples).

> Design principle of the whole project: **Overfitting is the main enemy.**
> Accuracy alone never decides anything — generalization gap, fold stability,
> uncertainty, and interpretability are first-class citizens everywhere.

---

## Repository structure

```
surrogate-composites/
├── configs/
│   └── config.yaml              # datasets, seeds (42 + 5 seeds), CV, models, Top-3 weights
├── data/
│   ├── raw/                     # the 3 original Excel datasets
│   └── discrete/                # generated Discrete-Input datasets (Phase 5)
├── notebooks/
│   ├── 01_eda_dataset_0136.ipynb    # Phase 2 — one EDA notebook per dataset
│   ├── 02_eda_dataset_0172.ipynb
│   └── 03_eda_dataset_3772.ipynb
├── scripts/                     # CLI entry points (one per phase / pipeline)
│   ├── run_eda.py                   # Phase 2
│   ├── run_pipeline1_single_original.py   # Pipeline 1
│   ├── run_pipeline2_multi_original.py    # Pipeline 2
│   ├── make_discrete_datasets.py          # Phase 5 (discretization)
│   ├── run_pipeline3_single_discrete.py   # Pipeline 3 (Top-3 only)
│   ├── run_pipeline4_multi_discrete.py    # Pipeline 4 (Top-3 only)
│   ├── run_tuning_top3.py                 # Phase 4 (Random Search + Optuna)
│   ├── run_uncertainty.py                 # Phase 7 (GPR PI + Bootstrap PI)
│   ├── run_xai.py                         # Phase 6 (SHAP / LIME / importances)
│   ├── run_learning_curves.py             # Overfitting control curves
│   └── run_stat_tests.py                  # Phase 7 (Friedman+Nemenyi, Wilcoxon)
├── src/
│   ├── config.py                # YAML config loader
│   ├── data/                    # loader (constant-feature handling) + discretization
│   ├── eda/                     # visualization, statistics, outlier detection
│   ├── models/registry.py       # 11 models + Random-Search spaces (single source of truth)
│   ├── pipelines/               # CV engine, single-output, multi-output runners
│   ├── evaluation/              # metrics, uncertainty, overfitting, stats tests, Top-3
│   ├── tuning/                  # Random Search (all) + Optuna (Top-3 only; NO Grid Search)
│   ├── xai/                     # SHAP, LIME, permutation / tree importance
│   ├── tracking/                # MLflow logging helpers
│   └── utils/                   # seeding, I/O
├── results/                     # generated figures / tables (gitignored)
├── mlruns/                      # MLflow local tracking store (gitignored)
├── requirements.txt
├── environment.yml
└── Makefile
```

## Setup (Phase 1 — MLOps)

**macOS / Linux:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)** — `make` and `source` are Unix-only tools and don't
exist on Windows by default; use these instead:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If activation is blocked by the execution policy, run once in that terminal
(session-only): `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Or with conda (any OS): `conda env create -f environment.yml`

Reproducibility is enforced everywhere: base seed **42**, every experiment
repeated over **5 seeds** `[42, 123, 7, 2024, 99]` (see `configs/config.yaml`),
scaling only inside sklearn Pipelines (no leakage), and all runs logged to
**MLflow** through a local **SQLite** backend (`mlflow.db`) rather than the
plain folder store, because recent MLflow versions block that folder store
by default. Launch the UI for the required screenshots:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## How to run (phase by phase)

**macOS / Linux** (`make` is preinstalled or `sudo apt install make`):
```bash
make eda          # Phase 2 : EDA figures + tables for all 3 datasets
make pipeline1    # Pipeline 1: single-output, original datasets, all models
make pipeline2    # Pipeline 2: multi-output, original datasets, all models
make discretize   # Phase 5 : build the 3 Discrete-Input datasets
make pipeline3    # Pipeline 3: single-output, discrete datasets, Top-3 models
make pipeline4    # Pipeline 4: multi-output, discrete datasets, Top-3 models
make tune         # Phase 4 : Random Search + Optuna (Top-3). Grid Search is forbidden.
make xai          # Phase 6 : SHAP global/local, LIME, permutation & tree importance
make stats        # Phase 7 : Friedman + Nemenyi, Wilcoxon (Single vs Multi)
python scripts/run_uncertainty.py       # Phase 7: prediction intervals (GPR + Bootstrap)
python scripts/run_learning_curves.py   # Overfitting control (learning curves)
```

**Windows (PowerShell)** — use `tasks.ps1` (mirrors the Makefile) or call the
scripts directly, either works:
```powershell
.\tasks.ps1 eda
.\tasks.ps1 pipeline1
.\tasks.ps1 pipeline2
.\tasks.ps1 discretize
.\tasks.ps1 pipeline3
.\tasks.ps1 pipeline4
.\tasks.ps1 tune
.\tasks.ps1 xai
.\tasks.ps1 stats
.\tasks.ps1 uncertainty
.\tasks.ps1 learning-curves
# or run everything in order:
.\tasks.ps1 all

# equivalently, direct calls work everywhere (no make/tasks.ps1 needed):
python scripts\run_eda.py --all
python scripts\run_pipeline1_single_original.py
```

The **Top-3 models** are selected automatically after Pipeline 1 by the
multi-criteria score (accuracy, generalization gap, fold stability,
uncertainty, interpretability — weights in `config.yaml`) and cached in
`results/top3_models.csv`.

## Assignment rules hard-coded into this repo

| Rule | Where it is enforced |
|---|---|
| `No.` column is an extra index → dropped | `src/data/loader.py` |
| Constant `Composite Volume Fraction (%)` (0136: all 0, 3772: all 1) → dropped, never in feature importance | `src/data/loader.py` |
| No data row is ever removed; outliers only flagged (IQR + LOF + Isolation Forest) | `src/eda/outliers.py` |
| Scaling inside Pipeline, fit on train fold only → no leakage | `src/models/registry.py` |
| Fixed seed 42 + repetition over 5 seeds | `configs/config.yaml`, `src/pipelines/cross_validation.py` |
| Shallow ANN only (1 hidden layer, 8–32 neurons, ReLU, Adam, early stopping) | `src/models/registry.py` |
| Overfitting management (L1/L2, early stopping, tree depth / min_samples_leaf, kernel regularization, k-Fold/LOOCV/Nested CV) | `src/models/registry.py`, `src/pipelines/cross_validation.py` |
| Metrics: R², RMSE, NRMSE, MAE + Train/Test gap + fold stability | `src/evaluation/metrics.py` |
| PI for GPR + Bootstrap PI for the rest | `src/evaluation/uncertainty.py` |
| Random Search for all, Optuna for Top-3, **no Grid Search** | `src/tuning/hyperparameter.py` |
| Level A (per-dataset outputs) vs Level B (shared outputs only) | `configs/config.yaml` (`shared_outputs`), `DatasetBundle.shared_Y()` |
| Temperature / Strain analysed only in Dataset_0136 (Level A) | `configs/config.yaml` |

## Team workflow notes

- `configs/config.yaml` is the single source of truth — change seeds, CV,
  model lists, or Top-3 weights there, never inside code.
- Add a new model in **one** place: `src/models/registry.py`
  (factory + search space); every pipeline picks it up automatically.
- `results/` and `mlruns/` are gitignored — regenerate them with the
  scripts; keep MLflow screenshots for the final report.
- Final report checklist (Phase 8): intro & Small-Data Surrogate Modeling
  motivation, full EDA, pipeline implementation details, comparison
  tables/plots with error bars, learning/validation-curve analysis,
  XAI + mechanical interpretation, Original vs Discrete comparison,
  final trusted-surrogate recommendation, limitations & future work.
