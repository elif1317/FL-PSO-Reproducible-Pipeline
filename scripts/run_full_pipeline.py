# =========================
# PATH & REPO SETUP
# =========================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# =========================
# STANDARD LIBRARY IMPORTS
# =========================
import os
import time
import json
import math
import hashlib
import traceback
from dataclasses import dataclass

# =========================
# THIRD-PARTY IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy.stats import wilcoxon, friedmanchisquare, binomtest
from scipy.linalg import cholesky, solve_triangular

try:
    from opfunu.cec_based import cec2017, cec2022
except Exception:
    cec2017 = None
    cec2022 = None

# =========================
# PROJECT IMPORTS
# =========================
from src.flpso.optimizers import *
from src.flpso.config import get_algorithms
# from src.flpso.cec import *
# from src.flpso.benchmarks import *

# =========================
# MATPLOTLIB SETTINGS
# =========================
plt.ioff()
plt.close("all")

# =========================
# OUTPUT PATHS
# =========================
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DATA PATHS
# =========================
DATA_DIR = ROOT / "data"
CEC2017_DIR = DATA_DIR / "cec2017"
CEC2022_DIR = DATA_DIR / "cec2022"
WRAPPER_DIR = DATA_DIR / "wrappers"

# =========================
# GLOBAL SETTINGS
# =========================
GLOBAL_SEED = 2025
FE_MULT_DEFAULT = 10_000
TAU_REL = 1e-6
TAU_ABS = 1e-12

# =========================
# ALGORITHM CONFIG
# =========================
ALGORITHMS = get_algorithms()


# =========================
# MAIN ENTRY POINT
# =========================
def main():
    print("=" * 50)
    print("FL-PSO REPRODUCIBLE PIPELINE")
    print("=" * 50)

    # basic system info
    print("\n[INFO] Paths")
    print(f"ROOT: {ROOT}")
    print(f"OUTPUT: {OUT_DIR}")
    print(f"CEC2017_DIR: {CEC2017_DIR}")
    print(f"CEC2022_DIR: {CEC2022_DIR}")

    # algorithms
    print("\n[INFO] Algorithms loaded:")
    for name in ALGORITHMS.keys():
        print(f" - {name}")

    # external packages
    print("\n[INFO] External dependencies:")
    print(f"opfunu available: {cec2017 is not None and cec2022 is not None}")

    # seed test
    np.random.seed(GLOBAL_SEED)
    test_rand = np.random.rand()
    print(f"\n[INFO] Seed test value: {test_rand:.6f}")

    print("\n[SUCCESS] Pipeline initialized successfully.")


if __name__ == "__main__":
    main()
