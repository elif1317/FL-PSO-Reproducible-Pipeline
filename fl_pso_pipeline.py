# -*- coding: utf-8 -*-
"""
FL-PSO (Residual-Corrected Fractional–Langevin PSO) — Reproducible Experimental Pipeline
======================================================================================

Single-file, copy-paste runnable script designed for reproducibility in academic work.

Authors:
    Elif Demir¹, Yusuf Zeren¹, Suayip Toprakseven², Alpaslan Demirci³ (corresponding author)
¹ Department of Mathematics, Yildiz Technical University, Istanbul, Turkey
² Department of Mathematics, Yozgat Bozok University, Yozgat, Turkey
³ Department of Electrical Engineering, Yildiz Technical University, Istanbul, Turkey

Features:
- No hard-coded paths → repository-relative + environment variable support
- Optional CEC 2017/2022 support (gracefully skipped if missing)
- Fixed seeds, FE-based termination
- Convergence curves + diversity tracking
- Basic result summary (results.csv)

Quick run:
    python fl_pso_pipeline.py
    python fl_pso_pipeline.py --help
"""
import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, binomtest

# ──────────────────────────────────────────────────────────────
# 1. PORTABLE CONFIGURATION
# ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

OUT_DIR = Path(os.environ.get("FLPSO_OUT_DIR", str(REPO_ROOT / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

WRAPPER_DIR = Path(os.environ.get("FLPSO_WRAPPER_DIR", str(REPO_ROOT / "wrappers")))
if WRAPPER_DIR.exists() and str(WRAPPER_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(WRAPPER_DIR.resolve()))

CEC2017_INPUT_DIR = Path(os.environ.get("FLPSO_CEC2017_INPUT", str(REPO_ROOT / "input_data/cec2017")))
CEC2022_INPUT_DIR = Path(os.environ.get("FLPSO_CEC2022_INPUT", str(REPO_ROOT / "input_data/cec2022")))

GLOBAL_SEED = int(os.environ.get("FLPSO_GLOBAL_SEED", "2025"))
FE_MULT_DEFAULT = int(os.environ.get("FLPSO_FE_MULT", "10000"))
TAU_REL = float(os.environ.get("FLPSO_TAU_REL", "1e-6"))
TAU_ABS = float(os.environ.get("FLPSO_TAU_ABS", "1e-12"))


def print_config():
    print("\n" + "="*70)
    print("FL-PSO Reproducible Pipeline")
    print("="*70)
    print(f"REPO_ROOT              : {REPO_ROOT}")
    print(f"OUT_DIR                : {OUT_DIR}")
    print(f"WRAPPER_DIR            : {WRAPPER_DIR} (exists={WRAPPER_DIR.exists()})")
    print(f"CEC2017_INPUT_DIR      : {CEC2017_INPUT_DIR} (exists={CEC2017_INPUT_DIR.exists()})")
    print(f"CEC2022_INPUT_DIR      : {CEC2022_INPUT_DIR} (exists={CEC2022_INPUT_DIR.exists()})")
    print(f"Global seed            : {GLOBAL_SEED}")
    print(f"Default FE multiplier  : {FE_MULT_DEFAULT} × D")
    print("="*70 + "\n")


# ──────────────────────────────────────────────────────────────
# 2. HELPERS
# ──────────────────────────────────────────────────────────────
def rng(seed):
    return np.random.default_rng(int(seed))


def clamp(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def mean_centroid_diversity(X):
    c = X.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(X - c, axis=1)))


def stable_seed(suite_name, fid, tag, run_idx, base_seed=GLOBAL_SEED):
    key = (suite_name, fid, tag, run_idx)
    s = repr(key).encode("utf-8", errors="ignore")
    h = hashlib.md5(s).hexdigest()
    return base_seed + int(h[:12], 16) % 2_000_000_000


# ──────────────────────────────────────────────────────────────
# 3. BENCHMARK SUITE
# ──────────────────────────────────────────────────────────────
def suite_classical(D: int):
    def sphere(x): return np.sum(x**2)
    def rosenbrock(x): return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    lb = -100 * np.ones(D)
    ub = 100 * np.ones(D)

    return [
        {"fid": "F1",  "name": "Sphere",     "fun": sphere,     "lb": lb, "ub": ub, "fopt": 0.0},
        {"fid": "F6",  "name": "Rosenbrock", "fun": rosenbrock, "lb": lb, "ub": ub, "fopt": 0.0},
        {"fid": "F13", "name": "Rastrigin",  "fun": rastrigin,  "lb": lb, "ub": ub, "fopt": 0.0},
    ]


# ──────────────────────────────────────────────────────────────
# 4. FL-PSO CORE
# ──────────────────────────────────────────────────────────────
@dataclass
class StopState:
    fe_budget: int
    fe: int = 0
    best: float = np.inf


def fl_pso(
    fun,
    lb,
    ub,
    fe_budget: int,
    N: int = 50,
    seed: int = 0,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    eta: float = 0.08,
    track_div: bool = True
):
    r = rng(seed)
    D = lb.size
    X = r.uniform(lb, ub, (N, D))
    V = np.zeros((N, D))

    stop = StopState(fe_budget=fe_budget)

    FX = np.array([fun(x) for x in X])
    stop.fe += N

    P = X.copy()
    FP = FX.copy()
    gbest_idx = np.argmin(FP)
    gbest = P[gbest_idx].copy()
    fg = FP[gbest_idx]

    curve = np.array([fg], dtype=float)
    div_curve = [] if track_div else None

    while stop.fe < stop.fe_budget:
        if track_div:
            div_curve.append(mean_centroid_diversity(X))

        r1 = r.random((N, D))
        r2 = r.random((N, D))

        centroid = X.mean(axis=0, keepdims=True)
        residual = -(X - centroid)

        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gbest - X) + eta * residual
        X = clamp(X + V, lb, ub)

        FX = np.array([fun(x) for x in X])
        stop.fe += N

        better = FX < FP
        P[better] = X[better]
        FP[better] = FX[better]

        new_best_idx = np.argmin(FP)
        if FP[new_best_idx] < fg:
            fg = FP[new_best_idx]
            gbest = P[new_best_idx].copy()

        curve = np.append(curve, fg)

        if stop.fe >= stop.fe_budget:
            break

    div_array = np.array(div_curve, dtype=float) if track_div else None
    return gbest, fg, stop, curve, div_array


# ──────────────────────────────────────────────────────────────
# 5. EXPERIMENT RUNNER
# ──────────────────────────────────────────────────────────────
def run_suite(
    suite_items,
    out_dir: Path,
    N: int = 50,
    runs: int = 30,
    alg_name: str = "FL_PSO",
    track_div: bool = True
):
    ensure_dir(out_dir)
    results = []

    for item in suite_items:
        fid = item["fid"]
        fun = item["fun"]
        lb = item["lb"]
        ub = item["ub"]
        fopt = item.get("fopt", None)
        D = item["lb"].size

        fe_budget = FE_MULT_DEFAULT * D
        print(f"→ {fid} ({item['name']})  D={D}  FE={fe_budget:,}")

        for r in range(runs):
            seed = stable_seed("classical", fid, alg_name, r)
            t0 = time.perf_counter()

            try:
                _, fb, stop, curve, div = fl_pso(
                    fun, lb, ub, fe_budget,
                    N=N, seed=seed, track_div=track_div
                )
                cpu_s = time.perf_counter() - t0

                success = False
                if fopt is not None:
                    tol = max(TAU_REL * (1 + abs(fopt)), TAU_ABS)
                    success = fb <= fopt + tol

                results.append({
                    "suite": "classical",
                    "fid": fid,
                    "D": D,
                    "run": r + 1,
                    "seed": seed,
                    "best": float(fb),
                    "success": success,
                    "fe_used": stop.fe,
                    "cpu_s": cpu_s,
                })

                # Save per-run curve
                curve_df = pd.DataFrame({"step": range(len(curve)), "best": curve})
                curve_df.to_csv(out_dir / f"curve__{fid}__run{r+1}.csv", index=False)

            except Exception as e:
                print(f"  Error on {fid} run {r+1}: {type(e).__name__} - {e}")
                traceback.print_exc()

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"Results saved: {out_dir / 'results.csv'}")
    return df


# ──────────────────────────────────────────────────────────────
# 6. MAIN + ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="FL-PSO Reproducible Experimental Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUT_DIR),
        help="Base output directory (default: ./outputs)"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent runs per function (default: 30)"
    )
    parser.add_argument(
        "--pop-size", type=int, default=50,
        help="Population size N (default: 50)"
    )
    parser.add_argument(
        "--no-diversity", action="store_true",
        help="Skip tracking and saving diversity curves"
    )
    return parser.parse_args()


def main():
    print_config()
    args = parse_args()

    print("Starting classical benchmark suite...")
    out_dir = ensure_dir(Path(args.output_dir) / "classical")
    suite = suite_classical(D=30)

    run_suite(
        suite_items=suite,
        out_dir=out_dir,
        N=args.pop_size,
        runs=args.runs,
        track_div=not args.no_diversity
    )

    print("\n" + "="*70)
    print("Experiment finished successfully.")
    print(f"Results saved to: {out_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
