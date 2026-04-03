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
import json
import traceback
from datetime import datetime

# =========================
# THIRD-PARTY IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from scipy.stats import wilcoxon

try:
    from opfunu.cec_based import cec2017, cec2022
except Exception:
    cec2017 = None
    cec2022 = None

# =========================
# PROJECT IMPORTS
# =========================
from src.flpso.config import get_algorithms

# =========================
# MATPLOTLIB SETTINGS
# =========================
plt.ioff()
plt.close("all")

# =========================
# CONFIG
# =========================
CONFIG_PATH = ROOT / "configs" / "default.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# =========================
# OUTPUT PATHS
# =========================
OUT_DIR = ROOT / CONFIG.get("output_dir", "results")
CLASSICAL_DIR = OUT_DIR / "classical"
ABLATION_DIR = OUT_DIR / "ablation"
STAT_DIR = OUT_DIR / "statistics"
PLOT_DIR = OUT_DIR / "plots"

for d in [OUT_DIR, CLASSICAL_DIR, ABLATION_DIR, STAT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
GLOBAL_SEED = int(CONFIG.get("seed", 2025))
FE_MULT_DEFAULT = int(CONFIG.get("fe_multiplier", 10_000))
DIMENSIONS = list(CONFIG.get("dimensions", [10, 30, 50]))
RUNS_SMALL = int(CONFIG.get("runs_small_dim", 30))
RUNS_LARGE = int(CONFIG.get("runs_large_dim", 50))
BENCHMARK_SUITE = str(CONFIG.get("benchmark_suite", "classical24")).lower()

# =========================
# ALGORITHM CONFIG
# =========================
ALGORITHMS = get_algorithms()

# =========================
# BENCHMARKS
# =========================
def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

def ackley(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(s1 / n))
    term2 = -np.exp(s2 / n)
    return float(term1 + term2 + 20.0 + np.e)

def griewank(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)

def zakharov(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1)
    s1 = np.sum(x**2)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2**2 + s2**4)

def schwefel_12(x):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(x)
    return float(np.sum(c**2))

def levy(x):
    x = np.asarray(x, dtype=float)
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    return float(term1 + term2 + term3)

def get_classical_benchmarks():
    return [
        {"name": "Sphere", "fun": sphere, "lb": -100.0, "ub": 100.0},
        {"name": "Rastrigin", "fun": rastrigin, "lb": -5.12, "ub": 5.12},
        {"name": "Rosenbrock", "fun": rosenbrock, "lb": -30.0, "ub": 30.0},
        {"name": "Ackley", "fun": ackley, "lb": -32.768, "ub": 32.768},
        {"name": "Griewank", "fun": griewank, "lb": -600.0, "ub": 600.0},
        {"name": "Zakharov", "fun": zakharov, "lb": -10.0, "ub": 10.0},
        {"name": "Schwefel12", "fun": schwefel_12, "lb": -100.0, "ub": 100.0},
        {"name": "Levy", "fun": levy, "lb": -10.0, "ub": 10.0},
    ]

# =========================
# HELPERS
# =========================
def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))

def get_num_runs(dim: int) -> int:
    return RUNS_SMALL if dim <= 30 else RUNS_LARGE

def fe_budget_from_dim(dim: int) -> int:
    return FE_MULT_DEFAULT * int(dim)

def make_bounds(lb_scalar: float, ub_scalar: float, dim: int):
    lb = np.full(dim, lb_scalar, dtype=float)
    ub = np.full(dim, ub_scalar, dtype=float)
    return lb, ub

def safe_curve(curve, target_len=None):
    curve = np.asarray(curve, dtype=float).ravel()
    if curve.size == 0:
        if target_len is None:
            return np.array([], dtype=float)
        return np.full(target_len, np.nan, dtype=float)
    if target_len is None:
        return curve
    if curve.size >= target_len:
        return curve[:target_len]
    out = np.full(target_len, curve[-1], dtype=float)
    out[:curve.size] = curve
    return out

def run_one_algorithm(algo_name, algo_callable, fun, lb, ub, fe_budget, seed):
    try:
        best_x, best_f, stop, curve, div_curve = algo_callable(
            fun,
            lb,
            ub,
            fe_budget,
            seed=int(seed),
            track_div=True
        )
        return {
            "ok": True,
            "best_x": best_x,
            "best_f": float(best_f),
            "fe_used": int(getattr(stop, "fe", fe_budget)),
            "curve": np.asarray(curve, dtype=float),
            "div_curve": np.asarray(div_curve, dtype=float),
            "error": ""
        }
    except Exception as e:
        return {
            "ok": False,
            "best_x": None,
            "best_f": np.nan,
            "fe_used": np.nan,
            "curve": np.array([], dtype=float),
            "div_curve": np.array([], dtype=float),
            "error": f"{type(e).__name__}: {e}"
        }

def save_convergence_plot(curves_by_algo, out_path, title):
    plt.figure(figsize=(8, 5))
    plotted = False
    for algo_name, curves in curves_by_algo.items():
        valid = [np.asarray(c, dtype=float) for c in curves if len(c) > 0]
        if not valid:
            continue
        max_len = max(len(c) for c in valid)
        padded = np.vstack([safe_curve(c, target_len=max_len) for c in valid])
        mean_curve = np.nanmean(padded, axis=0)
        plt.plot(mean_curve, label=algo_name)
        plotted = True

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far objective")
    if plotted:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def compute_pairwise_wilcoxon(df_runs, baseline="PSO"):
    rows = []
    grouped = df_runs.groupby(["benchmark", "dimension"])
    for (bench, dim), sub in grouped:
        pivot = sub.pivot_table(
            index="run",
            columns="algorithm",
            values="best_f",
            aggfunc="first"
        )
        if baseline not in pivot.columns:
            continue

        base_vals = pivot[baseline].dropna()
        for algo in pivot.columns:
            if algo == baseline:
                continue
            pair = pd.concat([pivot[baseline], pivot[algo]], axis=1).dropna()
            if len(pair) < 3:
                rows.append({
                    "benchmark": bench,
                    "dimension": dim,
                    "baseline": baseline,
                    "algorithm": algo,
                    "n": len(pair),
                    "p_value": np.nan,
                    "median_diff": np.nan,
                    "better_than_baseline": np.nan
                })
                continue

            try:
                stat = wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1], zero_method="wilcox", alternative="two-sided")
                median_diff = float(np.median(pair.iloc[:, 1] - pair.iloc[:, 0]))
                rows.append({
                    "benchmark": bench,
                    "dimension": dim,
                    "baseline": baseline,
                    "algorithm": algo,
                    "n": len(pair),
                    "p_value": float(stat.pvalue),
                    "median_diff": median_diff,
                    "better_than_baseline": bool(median_diff < 0.0)
                })
            except Exception:
                rows.append({
                    "benchmark": bench,
                    "dimension": dim,
                    "baseline": baseline,
                    "algorithm": algo,
                    "n": len(pair),
                    "p_value": np.nan,
                    "median_diff": np.nan,
                    "better_than_baseline": np.nan
                })

    return pd.DataFrame(rows)

def dump_run_metadata():
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root": str(ROOT),
        "output_dir": str(OUT_DIR),
        "data_dir": str(DATA_DIR),
        "benchmark_suite": BENCHMARK_SUITE,
        "dimensions": DIMENSIONS,
        "runs_small_dim": RUNS_SMALL,
        "runs_large_dim": RUNS_LARGE,
        "fe_multiplier": FE_MULT_DEFAULT,
        "global_seed": GLOBAL_SEED,
        "algorithms": list(ALGORITHMS.keys()),
        "opfunu_available": bool(cec2017 is not None and cec2022 is not None),
    }
    with open(OUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# =========================
# MAIN PIPELINE
# =========================
def run_classical_suite():
    benchmarks = get_classical_benchmarks()
    all_rows = []

    for bench in benchmarks:
        bench_name = bench["name"]
        bench_fun = bench["fun"]
        lb_scalar = bench["lb"]
        ub_scalar = bench["ub"]

        print(f"\n{'=' * 70}")
        print(f"[BENCHMARK] {bench_name}")
        print(f"{'=' * 70}")

        for dim in DIMENSIONS:
            n_runs = get_num_runs(dim)
            fe_budget = fe_budget_from_dim(dim)
            lb, ub = make_bounds(lb_scalar, ub_scalar, dim)

            print(f"[INFO] Dimension={dim}, Runs={n_runs}, FE budget={fe_budget}")

            curves_by_algo = {algo_name: [] for algo_name in ALGORITHMS.keys()}

            for run_id in range(n_runs):
                run_seed = GLOBAL_SEED + 1000 * dim + run_id

                for algo_name, algo_callable in ALGORITHMS.items():
                    print(f"  -> {algo_name:15s} | run {run_id + 1:02d}/{n_runs}", end="")

                    result = run_one_algorithm(
                        algo_name=algo_name,
                        algo_callable=algo_callable,
                        fun=bench_fun,
                        lb=lb,
                        ub=ub,
                        fe_budget=fe_budget,
                        seed=run_seed
                    )

                    row = {
                        "benchmark": bench_name,
                        "dimension": dim,
                        "run": run_id,
                        "seed": run_seed,
                        "algorithm": algo_name,
                        "best_f": result["best_f"],
                        "fe_used": result["fe_used"],
                        "status": "ok" if result["ok"] else "fail",
                        "error": result["error"],
                    }
                    all_rows.append(row)

                    if result["ok"]:
                        curves_by_algo[algo_name].append(result["curve"])
                        print(f" | best={result['best_f']:.6e}")
                    else:
                        print(f" | FAILED ({result['error']})")

            df_dim = pd.DataFrame([
                r for r in all_rows
                if r["benchmark"] == bench_name and r["dimension"] == dim
            ])

            dim_csv = CLASSICAL_DIR / f"{bench_name}_D{dim}_runs.csv"
            df_dim.to_csv(dim_csv, index=False)

            plot_path = PLOT_DIR / f"{bench_name}_D{dim}_convergence.png"
            save_convergence_plot(
                curves_by_algo=curves_by_algo,
                out_path=plot_path,
                title=f"{bench_name} (D={dim})"
            )

    return pd.DataFrame(all_rows)

def build_summary(df_runs):
    ok = df_runs[df_runs["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=[
            "benchmark", "dimension", "algorithm",
            "n_runs", "mean_best", "std_best", "median_best", "min_best", "max_best"
        ])

    summary = (
        ok.groupby(["benchmark", "dimension", "algorithm"], as_index=False)
        .agg(
            n_runs=("best_f", "count"),
            mean_best=("best_f", "mean"),
            std_best=("best_f", "std"),
            median_best=("best_f", "median"),
            min_best=("best_f", "min"),
            max_best=("best_f", "max"),
        )
        .sort_values(["benchmark", "dimension", "mean_best"], ascending=[True, True, True])
    )
    return summary

def main():
    print("=" * 70)
    print("FL-PSO REPRODUCIBLE PIPELINE")
    print("=" * 70)

    print("\n[INFO] Paths")
    print(f"ROOT: {ROOT}")
    print(f"OUTPUT: {OUT_DIR}")
    print(f"CONFIG: {CONFIG_PATH}")
    print(f"CEC2017_DIR: {CEC2017_DIR}")
    print(f"CEC2022_DIR: {CEC2022_DIR}")

    print("\n[INFO] Algorithms loaded:")
    for name in ALGORITHMS.keys():
        print(f" - {name}")

    print("\n[INFO] External dependencies:")
    print(f"opfunu available: {cec2017 is not None and cec2022 is not None}")

    set_global_seed(GLOBAL_SEED)
    test_rand = np.random.rand()
    print(f"\n[INFO] Seed test value: {test_rand:.6f}")

    print("\n[INFO] Loaded config:")
    print(json.dumps(CONFIG, indent=2))

    dump_run_metadata()

    try:
        if BENCHMARK_SUITE.startswith("classical"):
            print("\n[INFO] Running classical benchmark suite...")
            df_runs = run_classical_suite()
        else:
            print(f"\n[WARNING] Unsupported benchmark_suite='{BENCHMARK_SUITE}'. Falling back to classical suite.")
            df_runs = run_classical_suite()

        runs_csv = CLASSICAL_DIR / "all_runs.csv"
        df_runs.to_csv(runs_csv, index=False)

        df_summary = build_summary(df_runs)
        summary_csv = CLASSICAL_DIR / "summary.csv"
        df_summary.to_csv(summary_csv, index=False)

        df_wil = compute_pairwise_wilcoxon(df_runs, baseline="PSO")
        wil_csv = STAT_DIR / "wilcoxon_vs_pso.csv"
        df_wil.to_csv(wil_csv, index=False)

        print("\n[SUCCESS] Pipeline completed successfully.")
        print(f"[OUTPUT] Per-run results : {runs_csv}")
        print(f"[OUTPUT] Summary         : {summary_csv}")
        print(f"[OUTPUT] Statistics      : {wil_csv}")
        print(f"[OUTPUT] Plots folder     : {PLOT_DIR}")

    except Exception as e:
        print("\n[ERROR] Pipeline failed.")
        print(f"{type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
