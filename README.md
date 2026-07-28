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
│   ├── 01_eda_dataset_0136.ipynb    # Phase 2 deliverable: one EDA notebook per dataset
│   ├── 02_eda_dataset_0172.ipynb
│   └── 03_eda_dataset_3772.ipynb
│   # No other notebooks: the assignment only requires a notebook for EDA
│   # (Phase 2). Every other phase is delivered as independent scripts (below).
├── scripts/
│   ├── Dataset_0136/             # <-- the assignment's required independent
│   │   ├── run_eda.py            #     per-dataset pipeline/scripts: every
│   │   ├── run_discretize.py     #     phase gets its OWN standalone script
│   │   ├── run_pipeline1_single_original.py    #     PER dataset (no shared
│   │   ├── run_pipeline2_multi_original.py     #     "loop over all 3
│   │   ├── run_pipeline3_single_discrete.py     #    datasets" scripts).
│   │   ├── run_pipeline4_multi_discrete.py
│   │   ├── run_tuning.py                # Phase 4: Random Search (ALL models)
│   │   │                                 #          + Optuna/Nested-CV (Top-3)
│   │   ├── run_xai.py                   # Phase 6: SHAP/LIME/importances
│   │   │                                 #          + auto interpretation summary
│   │   ├── run_uncertainty.py           # Phase 7: GPR PI + Bootstrap PI
│   │   ├── run_learning_curves.py       # Overfitting control curves
│   │   └── run_stat_tests.py            # Phase 7: Friedman+Nemenyi, Wilcoxon
│   ├── Dataset_0172/              # (same 11 scripts)
│   ├── Dataset_3772/              # (same 11 scripts)
│   ├── select_top3.py            # cross-dataset (needs all 3 Pipeline-1 results)
│   ├── describe_top3.py          # cross-dataset: pandas .describe() + detailed
│   │                              # review of the Top-3 models (all datasets)
│   ├── _pipeline_common.py       # shared library used by every script above
│   │                              # (no duplicated logic between datasets)
│   ├── _bootstrap.py             # sys.path / warnings setup
│   └── run_*.py                  # flat "all 3 datasets" convenience wrappers
│                                  # (thin loops over the per-dataset scripts;
│                                  # NOT the required deliverable by themselves)
├── src/
│   ├── config.py                # YAML config loader
│   ├── data/                    # loader (constant-feature handling) + discretization
│   ├── eda/                     # visualization, statistics, outlier detection
│   ├── models/registry.py       # 11 models + Random-Search spaces (single source of truth)
│   ├── pipelines/               # CV engine, single-output, multi-output runners
│   ├── evaluation.py            # metrics, uncertainty, overfitting, stats tests, Top-3
│   ├── tuning/                  # Random Search (all) + Optuna (Top-3 only; NO Grid Search)
│   ├── xai/                     # SHAP, LIME, permutation / tree importance, interpretation
│   ├── tracking/                # MLflow logging helpers
│   └── utils/                   # seeding, I/O
├── results/                     # generated figures / tables (gitignored)
│   ├── pipeline{1,2,3,4}/<dataset>/     # per-dataset fold + aggregated results
│   ├── tuning/<dataset>/                # Stage-1 (all models) + Stage-2 (Top-3) tuning
│   ├── xai/<dataset>/<model>/<target>/  # plots + interpretation_summary.csv
│   ├── uncertainty/<dataset>/
│   ├── statistical_tests/<dataset>/
│   ├── top3_models.csv                  # cross-dataset Top-3 ranking
│   └── top3_review/                     # describe_top3.py output
├── mlruns/                      # legacy MLflow file-store (gitignored, stale/unused)
├── mlflow.db                    # MLflow SQLite tracking store (gitignored)
├── requirements.txt
├── environment.yml
├── Makefile
└── tasks.ps1
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

## How to run

### Per dataset (the assignment's required "independent scripts")

```bash
python scripts/Dataset_0136/run_eda.py
python scripts/Dataset_0136/run_pipeline1_single_original.py
python scripts/Dataset_0136/run_pipeline2_multi_original.py
python scripts/Dataset_0136/run_discretize.py
# ... repeat for Dataset_0172 and Dataset_3772 ...

# once Pipeline 1 has run for ALL THREE datasets:
python scripts/select_top3.py

python scripts/Dataset_0136/run_pipeline3_single_discrete.py
python scripts/Dataset_0136/run_pipeline4_multi_discrete.py
python scripts/Dataset_0136/run_tuning.py
python scripts/Dataset_0136/run_xai.py
python scripts/Dataset_0136/run_uncertainty.py
python scripts/Dataset_0136/run_learning_curves.py
python scripts/Dataset_0136/run_stat_tests.py

python scripts/describe_top3.py
```

`make <phase>-Dataset_0136` (e.g. `make eda-Dataset_0136`) and
`.\tasks.ps1 <phase>-Dataset_0136` are shorthands for the same calls.

### All 3 datasets at once (convenience wrappers)

**macOS / Linux** (`make` is preinstalled or `sudo apt install make`):
```bash
make eda          # Phase 2 : EDA figures + tables, all 3 datasets
make pipeline1    # Pipeline 1: single-output, original datasets, all models
make pipeline2    # Pipeline 2: multi-output, original datasets, all models
make discretize   # Phase 5 : build the 3 Discrete-Input datasets
make select-top3  # cross-dataset Top-3 selection (needs pipeline1 for all 3 first)
make pipeline3    # Pipeline 3: single-output, discrete datasets, Top-3 models
make pipeline4    # Pipeline 4: multi-output, discrete datasets, Top-3 models
make tune         # Phase 4 : Random Search (ALL models) + Optuna/Nested-CV (Top-3)
make xai          # Phase 6 : SHAP global/local, LIME, permutation & tree importance
make uncertainty  # Phase 7 : prediction intervals (GPR + Bootstrap)
make learning-curves   # Overfitting control (learning curves)
make stats        # Phase 7 : Friedman + Nemenyi, Wilcoxon (Single vs Multi)
make describe-top3     # pandas .describe() + detailed Top-3 review
make all          # everything, in the correct order
```

**Windows (PowerShell)**:
```powershell
.\tasks.ps1 eda
.\tasks.ps1 pipeline1
.\tasks.ps1 pipeline2
.\tasks.ps1 discretize
.\tasks.ps1 select-top3
.\tasks.ps1 pipeline3
.\tasks.ps1 pipeline4
.\tasks.ps1 tune
.\tasks.ps1 xai
.\tasks.ps1 uncertainty
.\tasks.ps1 learning-curves
.\tasks.ps1 stats
.\tasks.ps1 describe-top3
.\tasks.ps1 all
```

The **Top-3 models** are selected after Pipeline 1 has run for all three
datasets, by the multi-criteria score (accuracy, generalization gap, fold
stability, uncertainty, interpretability — weights in `config.yaml`) and
cached in `results/top3_models.csv`. Every later phase (tuning stage 2, XAI,
uncertainty, learning curves) reads that cache via `read_top3_models()`.

## Assignment rules hard-coded into this repo

| Rule | Where it is enforced |
|---|---|
| `No.` column is an extra index → dropped | `src/data/loader.py` |
| Constant `Composite Volume Fraction (%)` (0136: all 0, 3772: all 1) → dropped, never in feature importance | `src/data/loader.py` |
| No data row is ever removed; outliers only flagged (IQR + LOF + Isolation Forest) | `src/eda/outliers.py` |
| Scaling inside Pipeline, fit on train fold only → no leakage | `src/models/registry.py` |
| Fixed seed 42 + repetition over 5 seeds | `configs/config.yaml`, `src/pipelines/cross_validation.py` |
| Shallow ANN only (1 hidden layer, 8–32 neurons, ReLU, Adam, early stopping) | `src/models/registry.py` |
| Overfitting management (L1/L2, early stopping, tree depth / min_samples_leaf, kernel regularization, k-Fold/Nested CV) | `src/models/registry.py`, `src/pipelines/cross_validation.py` |
| Metrics: R², RMSE, NRMSE, MAE + Train/Test gap + fold stability | `src/evaluation.py` |
| PI for GPR + Bootstrap PI for the rest | `src/evaluation.py` |
| Random Search for **all** models, Optuna + Nested-CV honest check for Top-3, **no Grid Search** | `scripts/_pipeline_common.py: tuning_for_dataset`, `src/tuning/hyperparameter.py` |
| Level A (per-dataset outputs) vs Level B (shared outputs only) | `configs/config.yaml` (`shared_outputs`), `DatasetBundle.shared_Y()` |
| Temperature / Strain analysed as dedicated Level-A outputs, Dataset_0136 only, in every downstream phase (tuning/XAI/uncertainty/learning-curves) | `scripts/_pipeline_common.py: analysis_targets()` |
| Each dataset has its own independent pipeline/scripts | `scripts/Dataset_0136/`, `scripts/Dataset_0172/`, `scripts/Dataset_3772/` |

### Known, unavoidable limitation

Table 2 of the assignment lists **Dropout** as an overfitting technique for
the ANN. The assignment also mandates a plain `MLPRegressor` (scikit-learn),
which has **no dropout API at all** — dropout is a training-time technique
that doesn't exist in sklearn's MLP implementation. The ANN instead uses L2
regularization (`alpha`) + `early_stopping` (both in Table 2's own row for
ANN), which is the closest available substitute within the mandated
scikit-learn stack. Note this explicitly in the final report.

## Team workflow notes

- `configs/config.yaml` is the single source of truth — change seeds, CV,
  model lists, or Top-3 weights there, never inside code.
- Add a new model in **one** place: `src/models/registry.py`
  (factory + search space); every pipeline picks it up automatically.
- `results/` and `mlruns/` are gitignored — regenerate them with the
  scripts; keep MLflow screenshots for the final report.
- `results/xai/<dataset>/<model>/<target>/interpretation_summary.csv` is an
  **automated, numbers-only** starting point for the required scientific
  interpretation (strongest driver, effect sign, linearity, interaction
  proxy) — it is not a substitute for confirming physical plausibility
  against friction-process mechanics by hand in the final report.
- Final report checklist (Phase 8, see `report/FINAL_REPORT_TEMPLATE.md`):
  intro & Small-Data Surrogate Modeling motivation, full EDA, pipeline
  implementation details, comparison tables/plots with error bars,
  learning/validation-curve analysis, XAI + mechanical interpretation,
  Original vs Discrete comparison, final trusted-surrogate recommendation,
  limitations & future work.
