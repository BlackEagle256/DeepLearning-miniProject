<!--
Phase 8 deliverable skeleton - mirrors the assignment's required report
structure EXACTLY (9 sections, page 10-11 of the assignment PDF). This file
is scaffolding only: every section below must be filled in with the real
numbers/figures produced by the scripts once they have been run for all
three datasets. Pointers to the exact CSV/PNG files that back each section
are given as HTML comments - replace them with the actual tables/plots and
your written analysis before submission.
-->

# Development and Validation of Reliable, Interpretable, and Overfitting-Resistant Surrogate Models for Friction-Processed Composites — Final Report

Course: Deep Learning — Dr. Bahaghighat, Imam Khomeini International University

## 1. Introduction — Small-Data Surrogate Modeling in Mechanical Engineering

<!--
Explain: why friction-processed composite testing is expensive/slow (hence
only 36/72/36 samples exist), what a "surrogate model" is and why it must
replace expensive physical experiments here, and why THIS project treats
overfitting control (not accuracy) as the central design constraint.
-->

## 2. Full EDA for Every Dataset

<!--
One subsection per dataset (Dataset_0136, Dataset_0172, Dataset_3772).
Source data: results/eda/<dataset>/*.png and *.csv
  - histograms_kde.png, boxplots.png, qq_plots.png, pairplot.png, correlation_heatmaps.png
  - descriptive_statistics.csv, normality_tests.csv, correlation_tests.csv,
    kendall_confidence_intervals.csv, outlier_report.csv
For each dataset report: descriptive stats, normality conclusions
(Shapiro/Anderson-Darling/KS), correlation structure (Pearson/Spearman/
Kendall + bootstrap CI), and outlier flags (IQR+LOF+IsolationForest) with an
explicit note that NO row was removed.
-->

### 2.1 Dataset_0136 (36 samples)
### 2.2 Dataset_0172 (72 samples)
### 2.3 Dataset_3772 (36 samples)

## 3. Pipeline Implementation Details

<!--
Describe: the 4 pipelines (single/multi-output x original/discrete),
5-fold CV x 5 seeds, scaling-inside-pipeline (no leakage), the 11 models
(10 required + LightGBM as an optional extra explicitly listed in the
Phase-3 model list), and how overfitting management maps onto Table 2
(L1/L2, early stopping, tree-depth/min_samples_leaf limits, kernel
regularization, k-Fold as the general framework). Note the ANN/Dropout
limitation explicitly (see README "Known, unavoidable limitation").
Source: src/models/registry.py, src/pipelines/, scripts/Dataset_XXXX/.
-->

## 4. Comparative Tables and Plots (with error bars / std)

<!--
Source: results/pipeline{1,2,3,4}/<dataset>/aggregated_results.csv
  (mean +/- std of R2/RMSE/NRMSE/MAE, train/test gap, fold stability)
Also: results/top3_models.csv (multi-criteria ranking) and
results/top3_review/{describe_per_model.csv, describe_per_model_dataset.csv,
top3_summary_per_dataset.csv, top3_rollup_all_datasets.csv} (Top-3 detailed
.describe() review).
Include: per-dataset model leaderboards, and the Level-B cross-dataset
comparison restricted to results/pipeline*/*/aggregated_results.csv rows
whose target is in configs/config.yaml: shared_outputs.
-->

## 5. Learning / Validation Curve Analysis (Overfitting Control)

<!--
Source: results/learning_curves/<dataset>/<model>__<target>.png/.csv
Discuss, per Top-3 model: does the Train/CV RMSE gap close as training size
grows? Does it plateau (data-starved) or converge (well-specified model)?
Cross-reference with results/pipeline1/<dataset>/aggregated_results.csv
gap_r2/gap_rmse columns and results/tuning/<dataset>/top3_optuna_nested_cv.csv
(nested_cv_test_rmse_mean vs the non-nested tuned score) as an HONEST
generalization check that tuning itself didn't overfit the small dataset.
-->

## 6. Results + XAI Mechanical Interpretation

<!--
Source: results/xai/<dataset>/<model>/<target>/
  - shap_summary.png, shap_beeswarm.png, shap_feature_importance.png,
    shap_dependence_<feature>.png, shap_waterfall_sample_{0,1}.png,
    shap_feature_interaction.png (tree models), lime_sample_{0,1}.png/.txt
  - permutation_importance.csv, tree_feature_importance.csv
  - interpretation_summary.csv  <-- AUTOMATED starting point: strongest
    driver, effect sign, linearity (linear/non-linear/non-monotonic),
    and a feature-interaction proxy, computed directly from the SHAP values.

For EACH dataset / Top-3 model / target, answer explicitly (this is
mandatory per the assignment - plots alone are not sufficient):
  1. Which input has the strongest effect?
  2. Is that effect positive or negative?
  3. Is it linear or non-linear?
  4. Is there interaction between inputs?
  5. Is the result physically consistent with friction-process /
     composite mechanics? (confirm or override the automated summary here
     using mechanical-engineering domain knowledge)
-->

## 7. Original vs Discrete Inputs Comparison

<!--
Source: results/pipeline{1,3}/<dataset>/aggregated_results.csv (single-output,
original vs discrete) and results/pipeline{2,4}/<dataset>/aggregated_results.csv
(multi-output). Also results/statistical_tests/<dataset>/ for formal
significance tests where applicable.
Answer: did discretizing the inputs help, hurt, or not matter for the Top-3
models? Report R2/RMSE deltas with std, not just point estimates.
-->

## 8. Scientific Conclusion — Which Models Are Trustworthy Surrogates?

<!--
Source: results/top3_models.csv (composite ranking + criteria breakdown),
results/top3_review/ (detailed describe()-based review),
results/uncertainty/<dataset>/prediction_intervals.csv (PI width/coverage),
results/statistical_tests/<dataset>/{friedman_across_models.csv,nemenyi.csv,
wilcoxon_single_vs_multi.csv}.
State explicitly, with justification: which model(s) are recommended as
trustworthy surrogates for this small-data problem, and why (not just
accuracy - cite generalization gap, fold stability, uncertainty width,
interpretability, and statistical significance of any claimed advantage).
-->

## 9. Limitations and Future Work

<!--
Cover at least: small-sample-size limits on statistical power (Friedman/
Wilcoxon), the ANN-Dropout/MLPRegressor limitation, outliers that were
flagged but intentionally kept (mechanical justification or lack thereof),
and what more data / more physics-informed features could improve.
-->
