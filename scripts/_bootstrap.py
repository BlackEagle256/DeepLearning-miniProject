"""Make ``src`` importable when scripts are run from anywhere."""

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Harmless sklearn warning triggered when a model trained on a DataFrame
# (or vice versa) is later called with a plain numpy array (or vice versa).
# It never affects correctness here since scaling/estimators are consistent
# within each pipeline call - it only adds noise to script logs.
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="X has feature names, but.*")
