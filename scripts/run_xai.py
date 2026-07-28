"""Phase 6 - convenience wrapper: XAI for the Top-3 models, ALL 3 datasets.

SHAP (global + local), LIME, permutation and tree importances, plus an
automated scientific-interpretation summary (strongest driver, effect
sign, linearity, interaction strength) computed directly from the SHAP
values - remember the assignment requires INTERPRETING these plots
(effect direction, linearity, interactions, physical consistency with the
friction process), not just showing them; the auto-generated summary is a
numerical starting point, not a substitute for that final mechanical check.

The actual per-dataset deliverable is ``scripts/<dataset>/run_xai.py``.
"""

import _bootstrap  # noqa: F401

from _pipeline_common import read_top3_models, xai_for_dataset
from src.config import load_config


def main() -> None:
    top3 = read_top3_models()
    for ds_name in load_config()["datasets"]:
        xai_for_dataset(ds_name, top3)


if __name__ == "__main__":
    main()
