# Convenience targets for the project.
#
# Every phase has an INDEPENDENT script per dataset under
# scripts/Dataset_0136/, scripts/Dataset_0172/, scripts/Dataset_3772/
# (the assignment's "har dataset bayad pipeline jodagane / script-haye
# mostaghel dashte bashad" requirement). The flat targets below (eda,
# pipeline1, ...) are pure convenience wrappers that just loop over the
# three per-dataset scripts - use the -0136/-0172/-3772 targets to run a
# single dataset only.
PY=python

# ---- per-dataset targets (e.g. make eda-Dataset_0136) ------------------
eda-%:
	$(PY) scripts/$*/run_eda.py

discretize-%:
	$(PY) scripts/$*/run_discretize.py

pipeline1-%:
	$(PY) scripts/$*/run_pipeline1_single_original.py

pipeline2-%:
	$(PY) scripts/$*/run_pipeline2_multi_original.py

pipeline3-%:
	$(PY) scripts/$*/run_pipeline3_single_discrete.py

pipeline4-%:
	$(PY) scripts/$*/run_pipeline4_multi_discrete.py

tune-%:
	$(PY) scripts/$*/run_tuning.py

xai-%:
	$(PY) scripts/$*/run_xai.py

uncertainty-%:
	$(PY) scripts/$*/run_uncertainty.py

learning-curves-%:
	$(PY) scripts/$*/run_learning_curves.py

stats-%:
	$(PY) scripts/$*/run_stat_tests.py

# ---- cross-dataset steps (the only two that are legitimately not
#      per-dataset: Top-3 selection needs all 3 datasets' Pipeline-1
#      results first, and its detailed review reads all 3 back) --------
select-top3:
	$(PY) scripts/select_top3.py

describe-top3:
	$(PY) scripts/describe_top3.py

# ---- flat "all 3 datasets" convenience wrappers ------------------------
eda:
	$(PY) scripts/run_eda.py --all

discretize:
	$(PY) scripts/make_discrete_datasets.py

pipeline1:
	$(PY) scripts/run_pipeline1_single_original.py

pipeline2:
	$(PY) scripts/run_pipeline2_multi_original.py

pipeline3:
	$(PY) scripts/run_pipeline3_single_discrete.py

pipeline4:
	$(PY) scripts/run_pipeline4_multi_discrete.py

tune:
	$(PY) scripts/run_tuning_top3.py

xai:
	$(PY) scripts/run_xai.py

stats:
	$(PY) scripts/run_stat_tests.py

uncertainty:
	$(PY) scripts/run_uncertainty.py

learning-curves:
	$(PY) scripts/run_learning_curves.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

all: eda pipeline1 pipeline2 discretize select-top3 pipeline3 pipeline4 tune xai uncertainty learning-curves stats describe-top3
