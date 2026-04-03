# =========================
# PATH & REPO SETUP
# =========================
import sys
from pathlib import Path

# repo root path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# =========================
# IMPORTS
# =========================
from src.flpso.optimizers import *

import os
import time
import json
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

from scipy.special import gamma as Gamma
from scipy.integrate import quad
from scipy.stats import wilcoxon, friedmanchisquare, binomtest
from scipy.linalg import cholesky, solve_triangular

# =========================
# OUTPUT PATH (FIXED)
# =========================
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DATA PATHS (FIXED)
# =========================
CEC2017_DIR = ROOT / "data" / "cec2017"
CEC2022_DIR = ROOT / "data" / "cec2022"

# =========================
# GLOBAL SETTINGS
# =========================
GLOBAL_SEED = 2025
FE_MULT_DEFAULT = 10_000
TAU_REL = 1e-6
TAU_ABS = 1e-12
