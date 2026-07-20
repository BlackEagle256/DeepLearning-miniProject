# Convenience targets for the four project pipelines and supporting phases.
PY=python

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

all: eda pipeline1 pipeline2 discretize pipeline3 pipeline4 tune xai uncertainty learning-curves stats
