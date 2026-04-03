# =========================
# PATH & REPO SETUP
# =========================
import sys
from pathlib import Path

# repo root path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# =========================
# PROJECT IMPORTS
# =========================
from src.flpso.optimizers import *
from src.flpso.config import get_algorithms
# from src.flpso.cec import *   # CEC modülünü tam entegre edince bunu aç
# from src.flpso.benchmarks import *   # benchmarks.py'yi en son entegre edeceğiz

# =========================
# STANDARD LIBRARY / THIRD-PARTY IMPORTS
# =========================
import os
import time
import json
import math
import hashlib
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy.stats import wilcoxon, friedmanchisquare, binomtest
from scipy.linalg import cholesky, solve_triangular

# Optional external package
from opfunu.cec_based import cec2017, cec2022

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

# Optional wrapper directory
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
