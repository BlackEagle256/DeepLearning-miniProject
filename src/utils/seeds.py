"""Global seeding utilities for reproducibility.

Small datasets make results extremely seed-sensitive, therefore every
experiment must (a) fix a base seed and (b) be repeated over 5 seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
