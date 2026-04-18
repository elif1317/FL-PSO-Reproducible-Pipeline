# -*- coding: utf-8 -*-
"""
FL-PSO GUI - Enhanced standalone version.

- All benchmark functions and algorithms are defined directly within the code.
- No external benchmark files are required.

Enhancements:
    * Progress bar and status display
    * Dedicated FL-PSO parameter panel
    * Export of results to CSV format
    * Convergence plots for selected functions/algorithms
    * Statistical summary and ranking table
    * Fix for ablation closure bug
    * Proper recording of diversity traces
"""

import sys
import math
import time
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.special import gamma as Gamma
from scipy.stats import friedmanchisquare, wilcoxon

# =============================================================================
# 1. BENCHMARK FONKSIYONLARI
# =============================================================================

def f_sphere(z): return np.sum(z**2)
def f_schwefel_222(z): return np.sum(np.abs(z)) + np.prod(np.abs(z))
def f_powell_sum(z):
    i = np.arange(1, z.size + 1, dtype=float)
    return np.sum(np.abs(z) ** (i + 1))
def f_schwefel_12(z): return np.sum(np.cumsum(z) ** 2)
def f_schwefel_221(z): return np.max(np.abs(z))
def f_rosenbrock(z): return np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)
def f_step(z): return np.sum((np.floor(z + 0.5)) ** 2)
def f_quartic_core(z):
    i = np.arange(1, z.size + 1, dtype=float)
    return np.sum(i * (z ** 4))
def f_zakharov(z):
    i = np.arange(1, z.size + 1)
    s1 = np.sum(z ** 2)
    s2 = 0.5 * np.sum(i * z)
    return s1 + s2**2 + s2**4
def f_schwefel_226(z):
    return 418.9829 * z.size - np.sum(z * np.sin(np.sqrt(np.abs(z))))
def f_periodic(z): return 1.0 + np.sum(np.sin(z) ** 2) - np.exp(-np.sum(z ** 2))
def f_styblinski_tang(z): return 0.5 * np.sum(z ** 4 - 16 * z ** 2 + 5 * z)
def f_rastrigin(z): return 10.0 * z.size + np.sum(z ** 2 - 10.0 * np.cos(2 * np.pi * z))
def f_ackley(z):
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.sum(z ** 2)
    s2 = np.sum(np.cos(c * z))
    return -a * np.exp(-b * np.sqrt(s1 / z.size)) - np.exp(s2 / z.size) + a + np.e
def f_griewank(z):
    i = np.arange(1, z.size + 1)
    return np.sum(z ** 2) / 4000.0 - np.prod(np.cos(z / np.sqrt(i))) + 1.0
def f_xin_she_yang4(z):
    return (np.sum(np.sin(z) ** 2) - np.exp(-np.sum(z ** 2))) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(z))) ** 2))
def f_penalized_1(x):
    D = x.size
    term1 = (np.pi / D) * (
        10 * np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2
        + np.sum((((x[:-1] + 1) / 4) ** 2) * (1 + 10 * np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2))
        + ((x[-1] + 1) / 4) ** 2
    )
    u = np.where(x > 10, 100 * (x - 10) ** 4, 0) + np.where(x < -10, 100 * (-x - 10) ** 4, 0)
    return term1 + np.sum(u)
def f_penalized_2(x):
    term = 0.1 * (
        np.sin(3 * np.pi * x[0]) ** 2
        + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
        + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
    )
    u = np.where(x > 5, 100 * (x - 5) ** 4, 0) + np.where(x < -5, 100 * (-x - 5) ** 4, 0)
    return term + np.sum(u)
def f_foxholes(x):
    x = np.asarray(x, float); assert x.size == 2
    a = np.array([-32, -16, 0, 16, 32], dtype=float)
    A = np.array([(ai, aj) for ai in a for aj in a], dtype=float)
    j = np.arange(1, 26, dtype=float)
    denom = j + (x[0] - A[:, 0]) ** 6 + (x[1] - A[:, 1]) ** 6
    return 1.0 / (1 / 500.0 + np.sum(1.0 / denom))
def f_kowalik(x):
    x = np.asarray(x, float); assert x.size == 4
    a = np.array([0.1957,0.1947,0.1735,0.1600,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246], float)
    b = 1.0 / np.array([0.25,0.5,1.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0], float)
    yhat = (x[0] * (b**2 + b * x[1])) / (b**2 + b * x[2] + x[3])
    return np.sum((a - yhat) ** 2)
def f_six_hump_camel(x):
    x = np.asarray(x, float); assert x.size == 2
    x1, x2 = x[0], x[1]
    return (4 - 2.1*x1**2 + (x1**4)/3.0)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2

A_shekel = np.array([
    [4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
    [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]
], float)
c_shekel = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5], float)

def shekel_core(x, m):
    s = 0.0
    for i in range(m):
        s += 1.0 / (c_shekel[i] + np.sum((x - A_shekel[i]) ** 2))
    return -s

def classical_fmin(fid, D):
    if fid in {"F1","F2","F3","F4","F5","F6","F7","F8","F9","F11","F13","F14","F15","F16","F17","F18"}:
        return 0.0
    if fid == "F10": return 0.0
    if fid == "F12": return -39.166165703771 * D
    if fid == "F19": return 0.998003837794
    if fid == "F20": return 3.075e-4
    if fid == "F21": return -1.031628453
    if fid == "F22": return -10.1532
    if fid == "F23": return -10.4028
    if fid == "F24": return -10.5364
    return None

def suite_classical24(D):
    funs = []
    def wrap(core, lb, ub, fid, name, Dfix=None):
        size = Dfix if Dfix is not None else D
        lbv = lb * np.ones(size)
        ubv = ub * np.ones(size)
        def f(x): return float(core(np.asarray(x, float)))
        return dict(fid=fid, name=name, fun=f, lb=lbv, ub=ubv, fopt=classical_fmin(fid, size), suite="CLASSICAL24", D=size)
    funs += [wrap(f_sphere, -100, 100, "F1", "Sphere")]
    funs += [wrap(f_schwefel_222, -10, 10, "F2", "Schwefel 2.22")]
    funs += [wrap(f_powell_sum, -1, 1, "F3", "Powell Sum")]
    funs += [wrap(f_schwefel_12, -100, 100, "F4", "Schwefel 1.2")]
    funs += [wrap(f_schwefel_221, -100, 100, "F5", "Schwefel 2.21")]
    funs += [wrap(f_rosenbrock, -30, 30, "F6", "Rosenbrock")]
    funs += [wrap(f_step, -100, 100, "F7", "Step")]
    funs += [wrap(f_quartic_core, -1.28, 1.28, "F8", "Quartic (core)")]
    funs += [wrap(f_zakharov, -5, 10, "F9", "Zakharov")]
    funs += [wrap(f_schwefel_226, -500, 500, "F10", "Schwefel 2.26")]
    funs += [wrap(f_periodic, -10, 10, "F11", "Periodic")]
    funs += [wrap(f_styblinski_tang, -5, 5, "F12", "Styblinski–Tang")]
    funs += [wrap(f_rastrigin, -5.12, 5.12, "F13", "Rastrigin")]
    funs += [wrap(f_ackley, -32, 32, "F14", "Ackley")]
    funs += [wrap(f_griewank, -600, 600, "F15", "Griewank")]
    funs += [wrap(f_xin_she_yang4, -10, 10, "F16", "Xin-She Yang N.4")]
    funs += [wrap(f_penalized_1, -50, 50, "F17", "Penalized 1")]
    funs += [wrap(f_penalized_2, -50, 50, "F18", "Penalized 2")]
    funs += [wrap(f_foxholes, -65, 65, "F19", "Shekel's Foxholes", Dfix=2)]
    funs += [wrap(f_kowalik, -5, 5, "F20", "Kowalik", Dfix=4)]
    funs += [wrap(f_six_hump_camel, -5, 5, "F21", "Six-Hump Camel", Dfix=2)]
    funs += [wrap(lambda x: shekel_core(np.asarray(x, float), 5), 0, 10, "F22", "Shekel-5", Dfix=4)]
    funs += [wrap(lambda x: shekel_core(np.asarray(x, float), 7), 0, 10, "F23", "Shekel-7", Dfix=4)]
    funs += [wrap(lambda x: shekel_core(np.asarray(x, float), 10), 0, 10, "F24", "Shekel-10", Dfix=4)]
    return funs

def penalty(f, g_ineq, rho=1e6):
    def F(x):
        x = np.asarray(x, float)
        viol = np.maximum(np.asarray(g_ineq(x), float), 0.0)
        return float(f(x)) + rho * np.sum(viol**2)
    return F

def suite_engineering():
    funs = []
    def f_spring(x): x1,x2,x3 = x; return (x3+2.0)*x2*(x1**2)
    def g_spring(x):
        x1,x2,x3 = x
        return [1 - (x2**3*x3)/(71785*(x1**4)),
                (4*x2**2 - x1*x2)/(12566*(x2*x1**3 - x1**4)) + 1/(5108*x1**2) - 1,
                1 - (140.45*x1)/(x2**2*x3),
                (x2+x1)/1.5 - 1]
    funs.append(dict(fid="ENG1", name="Spring Design", fun=penalty(f_spring, g_spring),
                     lb=np.array([0.05,0.25,2.0]), ub=np.array([2.0,1.3,15.0]), fopt=None, suite="ENGINEERING", D=3))

    def f_pv(x): x1,x2,x3,x4 = x; return 0.6224*x1*x3*x4 + 1.7781*x2*x3**2 + 3.1661*x1**2*x4 + 19.84*x1**2*x3
    def g_pv(x):
        x1,x2,x3,x4 = x
        return [0.0193*x3 - x1, 0.00954*x3 - x2, (math.pi*x3**2*x4 + (4/3)*math.pi*x3**3) - 1296000, x4-240]
    funs.append(dict(fid="ENG2", name="Pressure Vessel", fun=penalty(f_pv, g_pv),
                     lb=np.array([0.0625,0.0625,10.0,10.0]), ub=np.array([5.0,5.0,200.0,240.0]), fopt=None, suite="ENGINEERING", D=4))

    def f_wb(x): h,l,t,b = x; return 1.10471*h*h*l + 0.04811*t*b*(14.0+l)
    def g_wb(x):
        h,l,t,b = x
        P=6000; L=14; E=30e6; G=12e6; tau_max=13600; sigma_max=30000; delta_max=0.25
        M = P*(L + l/2.0)
        R = math.sqrt(l**2/4.0 + ((h+t)/2.0)**2)
        J = 2*math.sqrt(2)*h*l*(l**2/12.0 + ((h+t)/2.0)**2)
        tau_p = P/(math.sqrt(2)*h*l)
        tau_pp = M*R/J
        tau = math.sqrt(max(0.0, tau_p**2 + 2*tau_p*tau_pp*l/(2*R) + tau_pp**2))
        sigma = 6*P*L/(b*t**2)
        delta = 4*P*L**3/(E*b*t**3)
        Pc = (4.013*E*math.sqrt(t**2 * b**6 /36.0)/L**2)*(1 - t/(2*L)*math.sqrt(E/(4*G)))
        return [tau - tau_max, sigma - sigma_max, h - b,
                0.10471*h*h + 0.04811*t*b*(14+l) - 5.0, 0.125 - h, delta - delta_max, P - Pc]
    funs.append(dict(fid="ENG3", name="Welded Beam", fun=penalty(f_wb, g_wb),
                     lb=np.array([0.1,0.1,0.1,0.1]), ub=np.array([2.0,10.0,10.0,2.0]), fopt=None, suite="ENGINEERING", D=4))

    def f_sr(x):
        x1,x2,x3,x4,x5,x6,x7 = x
        return (0.7854*x1*x2**2*(3.3333*x3**2+14.9334*x3-43.0934) -1.508*x1*(x6**2+x7**2)
                + 7.4777*(x6**3+x7**3) + 0.7854*(x4*x6**2+x5*x7**2))
    def g_sr(x):
        x1,x2,x3,x4,x5,x6,x7 = x
        return [27/(x1*x2**2*x3)-1, 397.5/(x1*x2**2*x3**2)-1, 1.93*x4**3/(x2*x3*x6**4)-1,
                1.93*x5**3/(x2*x3*x7**4)-1, (1/(110*x6**3))*math.sqrt((745*x4/(x2*x3))**2+16.9e6)-1,
                (1/(85*x7**3))*math.sqrt((745*x5/(x2*x3))**2+157.5e6)-1, x2*x3/40-1,
                5*x2/x1-1, x1/(12*x2)-1, (1.5*x6+1.9)/x4-1, (1.1*x7+1.9)/x5-1]
    funs.append(dict(fid="ENG4", name="Speed Reducer", fun=penalty(f_sr, g_sr),
                     lb=np.array([2.6,0.7,17.0,7.3,7.3,2.9,5.0]), ub=np.array([3.6,0.8,28.0,8.3,8.3,3.9,5.5]), fopt=None, suite="ENGINEERING", D=7))
    return funs



# =============================================================================
# BENCHMARK LABELS AND FORMULAS (GUI)
# =============================================================================

BENCHMARK_INFO = {
    "F1": {"label": "F1 (Sphere)", "name": "Sphere", "formula": "f(x) = Σ_{i=1}^n x_i^2", "notes": "Separable, unimodal, classical baseline benchmark."},
    "F2": {"label": "F2 (Schwefel 2.22)", "name": "Schwefel 2.22", "formula": "f(x) = Σ |x_i| + Π |x_i|", "notes": "Contains both sum and product terms of absolute values."},
    "F3": {"label": "F3 (Powell Sum)", "name": "Powell Sum", "formula": "f(x) = Σ_{i=1}^n |x_i|^{i+1}", "notes": "Component weights increase with the index."},
    "F4": {"label": "F4 (Schwefel 1.2)", "name": "Schwefel 1.2", "formula": "f(x) = Σ_{i=1}^n (Σ_{j=1}^i x_j)^2", "notes": "Strong dependence through cumulative sums."},
    "F5": {"label": "F5 (Schwefel 2.21)", "name": "Schwefel 2.21", "formula": "f(x) = max_i |x_i|", "notes": "Maximum-norm type benchmark."},
    "F6": {"label": "F6 (Rosenbrock)", "name": "Rosenbrock", "formula": "f(x) = Σ [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]", "notes": "Narrow curved valley; difficult for many optimizers."},
    "F7": {"label": "F7 (Step)", "name": "Step", "formula": "f(x) = Σ floor(x_i + 0.5)^2", "notes": "Piecewise-constant landscape with flat regions."},
    "F8": {"label": "F8 (Quartic)", "name": "Quartic", "formula": "f(x) = Σ i x_i^4", "notes": "Higher-order growth penalizes large deviations strongly."},
    "F9": {"label": "F9 (Zakharov)", "name": "Zakharov", "formula": "f(x) = Σ x_i^2 + (0.5 Σ i x_i)^2 + (0.5 Σ i x_i)^4", "notes": "Combines quadratic and coupled polynomial terms."},
    "F10": {"label": "F10 (Schwefel 2.26)", "name": "Schwefel 2.26", "formula": "f(x) = 418.9829 n - Σ x_i sin(sqrt(|x_i|))", "notes": "Highly multimodal with many local minima."},
    "F11": {"label": "F11 (Periodic)", "name": "Periodic", "formula": "f(x) = 1 + Σ sin^2(x_i) - exp(-Σ x_i^2)", "notes": "Periodic oscillation plus exponential damping."},
    "F12": {"label": "F12 (Styblinski-Tang)", "name": "Styblinski-Tang", "formula": "f(x) = 0.5 Σ (x_i^4 - 16x_i^2 + 5x_i)", "notes": "May have a negative global minimum."},
    "F13": {"label": "F13 (Rastrigin)", "name": "Rastrigin", "formula": "f(x) = 10n + Σ [x_i^2 - 10 cos(2πx_i)]", "notes": "Highly multimodal classical benchmark."},
    "F14": {"label": "F14 (Ackley)", "name": "Ackley", "formula": "f(x) = -20 exp(-0.2 sqrt((1/n)Σx_i^2)) - exp((1/n)Σcos(2πx_i)) + 20 + e", "notes": "Flat outer region and rippled center."},
    "F15": {"label": "F15 (Griewank)", "name": "Griewank", "formula": "f(x) = (1/4000)Σx_i^2 - Π cos(x_i/sqrt(i)) + 1", "notes": "Product term induces many local structures."},
    "F16": {"label": "F16 (Xin-She Yang N.4)", "name": "Xin-She Yang N.4", "formula": "f(x) = (Σ sin^2 x_i - exp(-Σ x_i^2)) exp(-Σ sin^2(sqrt(|x_i|)))", "notes": "Oscillatory and exponentially damped composite structure."},
    "F17": {"label": "F17 (Penalized 1)", "name": "Penalized 1", "formula": "Penalized benchmark with sinusoidal core and boundary penalties.", "notes": "Solutions outside the main box are strongly penalized."},
    "F18": {"label": "F18 (Penalized 2)", "name": "Penalized 2", "formula": "Penalized benchmark with trigonometric core and penalty terms.", "notes": "Boundary violations are handled via penalty terms."},
    "F19": {"label": "F19 (Shekel's Foxholes)", "name": "Shekel's Foxholes", "formula": "f(x) = [1/500 + Σ 1/(j + (x_1-a_{j1})^6 + (x_2-a_{j2})^6)]^{-1}", "notes": "Two-dimensional function with many trap regions."},
    "F20": {"label": "F20 (Kowalik)", "name": "Kowalik", "formula": "f(x) = Σ (a_j - (x_1(b_j^2+b_jx_2))/(b_j^2+b_jx_3+x_4))^2", "notes": "A classic four-parameter fitting benchmark."},
    "F21": {"label": "F21 (Six-Hump Camel)", "name": "Six-Hump Camel", "formula": "f(x) = (4-2.1x_1^2+x_1^4/3)x_1^2 + x_1x_2 + (-4+4x_2^2)x_2^2", "notes": "Two-dimensional with multiple local minima."},
    "F22": {"label": "F22 (Shekel-5)", "name": "Shekel-5", "formula": "f(x) = -Σ_{i=1}^5 1/(c_i + ||x-a_i||^2)", "notes": "Four-dimensional Shekel-family benchmark."},
    "F23": {"label": "F23 (Shekel-7)", "name": "Shekel-7", "formula": "f(x) = -Σ_{i=1}^7 1/(c_i + ||x-a_i||^2)", "notes": "Harder Shekel variant with more attractors."},
    "F24": {"label": "F24 (Shekel-10)", "name": "Shekel-10", "formula": "f(x) = -Σ_{i=1}^{10} 1/(c_i + ||x-a_i||^2)", "notes": "The densest Shekel-family version in this suite."},
    "ENG1": {"label": "ENG1 (Spring Design)", "name": "Spring Design", "formula": "min f(x) = (x_3 + 2)x_2x_1^2, subject to design constraints handled by penalty.", "notes": "Compression spring design problem."},
    "ENG2": {"label": "ENG2 (Pressure Vessel)", "name": "Pressure Vessel", "formula": "min f(x) = 0.6224x_1x_3x_4 + 1.7781x_2x_3^2 + 3.1661x_1^2x_4 + 19.84x_1^2x_3", "notes": "Classical pressure vessel design benchmark."},
    "ENG3": {"label": "ENG3 (Welded Beam)", "name": "Welded Beam", "formula": "min f(x) = 1.10471 h^2 l + 0.04811tb(14+l)", "notes": "Welded beam design with stress and deflection constraints."},
    "ENG4": {"label": "ENG4 (Speed Reducer)", "name": "Speed Reducer", "formula": "Multivariable reducer design objective with multiple nonlinear constraints.", "notes": "Mechanical design benchmark with many constraints."},
}

# =============================================================================
# 2. OPTIMIZATION ALGORITHMS
# =============================================================================

def rng(seed): return np.random.default_rng(int(seed))
def clamp(x, lb, ub): return np.minimum(np.maximum(x, lb), ub)
def mean_centroid_diversity(X):
    c = X.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(X - c, axis=1).mean())

class StopState:
    def __init__(self, fe_budget):
        self.fe_budget = fe_budget
        self.fe = 0
        self.best = np.inf

def eval_pop(fun, X, stop):
    vals = np.array([fun(x) for x in X])
    stop.fe += len(X)
    return vals

def _pack_result(best_x, best_f, stop, best_hist, div_hist=None):
    stop.best = float(best_f)
    if div_hist is None:
        div_hist = np.array([])
    return best_x, float(best_f), stop, np.asarray(best_hist, float), np.asarray(div_hist, float)

# ---------- PSO ----------
def PSO(fun, lb, ub, fe_budget, N=50, seed=0, w=0.72, c1=1.49, c2=1.49):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    V = r.uniform(-np.abs(ub-lb), np.abs(ub-lb), (N, D)) * 0.1
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    P, FP = X.copy(), FX.copy()
    g = P[np.argmin(FP)].copy(); fg = float(FP.min())
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        hist_best.append(fg)
        hist_div.append(mean_centroid_diversity(X))
        r1, r2 = r.random((N, D)), r.random((N, D))
        V = w*V + c1*r1*(P-X) + c2*r2*(g-X)
        X = clamp(X+V, lb, ub)
        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        P[upd], FP[upd] = X[upd], FX[upd]
        j = np.argmin(FP)
        if FP[j] < fg:
            fg, g = float(FP[j]), P[j].copy()
    return _pack_result(g, fg, stop, hist_best, hist_div)

# ---------- DMS-PSO ----------
def DMSPSO(fun, lb, ub, fe_budget, N=50, seed=0, n_swarms=5, regroup_iter=20):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    V = r.uniform(-np.abs(ub-lb), np.abs(ub-lb), (N, D)) * 0.1
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    P, FP = X.copy(), FX.copy()
    def assign_groups(): return np.array_split(r.permutation(N), n_swarms)
    groups = assign_groups()
    w, c1, c2 = 0.72, 1.49, 1.49
    hist_best, hist_div = [], []
    it = 0
    while stop.fe < stop.fe_budget:
        hist_best.append(float(FP.min()))
        hist_div.append(mean_centroid_diversity(X))
        if it % regroup_iter == 0 and it > 0:
            groups = assign_groups()
        for G in groups:
            if len(G) == 0:
                continue
            lidx = G[np.argmin(FP[G])]
            lbest = P[lidx]
            r1, r2 = r.random((len(G), D)), r.random((len(G), D))
            V[G] = w*V[G] + c1*r1*(P[G]-X[G]) + c2*r2*(lbest - X[G])
            X[G] = clamp(X[G]+V[G], lb, ub)
        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        P[upd], FP[upd] = X[upd], FX[upd]
        it += 1
    j = np.argmin(FP)
    return _pack_result(P[j], FP[j], stop, hist_best, hist_div)

# ---------- CLPSO ----------
def CLPSO(fun, lb, ub, fe_budget, N=50, seed=0, w=0.729, c=1.494, Pc_min=0.05, Pc_max=0.5, refresh=7):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    V = r.uniform(-np.abs(ub-lb), np.abs(ub-lb), (N, D)) * 0.1
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    P, FP = X.copy(), FX.copy()
    g = P[np.argmin(FP)].copy(); fg = float(FP.min())
    i = np.arange(N)
    Pc = Pc_min + (Pc_max-Pc_min)*(np.exp(10*i/(N-1))-1)/(np.exp(10)-1)
    exemplar = np.zeros((N, D), int)
    stay = np.zeros(N, int)
    def pick_exemplar(_):
        ex = np.empty(D, int)
        for d in range(D):
            a, b = r.integers(0, N), r.integers(0, N)
            ex[d] = a if FP[a] < FP[b] else b
        return ex
    for i0 in range(N):
        exemplar[i0] = pick_exemplar(i0)
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        hist_best.append(fg)
        hist_div.append(mean_centroid_diversity(X))
        stay += 1
        for i0 in range(N):
            if stay[i0] >= refresh:
                stay[i0] = 0
                exemplar[i0] = pick_exemplar(i0)
        for i0 in range(N):
            learn_mask = r.random(D) < Pc[i0]
            if not np.any(learn_mask):
                learn_mask[r.integers(0, D)] = True
            p_ex = P[exemplar[i0], np.arange(D)]
            V[i0, learn_mask] = w*V[i0, learn_mask] + c*r.random(np.sum(learn_mask))*(p_ex[learn_mask] - X[i0, learn_mask])
            X[i0] = clamp(X[i0] + V[i0], lb, ub)
        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        P[upd], FP[upd] = X[upd], FX[upd]
        j = np.argmin(FP)
        if FP[j] < fg:
            fg, g = float(FP[j]), P[j].copy()
    return _pack_result(g, fg, stop, hist_best, hist_div)

# ---------- HCLPSO ----------
def HCLPSO(fun, lb, ub, fe_budget, N=50, seed=0):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    V = r.uniform(-np.abs(ub-lb), np.abs(ub-lb), (N, D)) * 0.1
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    P, FP = X.copy(), FX.copy()
    g = P[np.argmin(FP)].copy(); fg = float(FP.min())
    def pick_exemplar():
        ex = np.empty((N//2, D), int)
        for i0 in range(N//2):
            for d in range(D):
                a, b = r.integers(0, N), r.integers(0, N)
                ex[i0, d] = a if FP[a] < FP[b] else b
        return ex
    ex = pick_exemplar()
    w, c, c2 = 0.729, 1.494, 1.3
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        hist_best.append(fg)
        hist_div.append(mean_centroid_diversity(X))
        if (stop.fe // N) % 10 == 0:
            ex = pick_exemplar()
        for i0 in range(N//2):
            r1 = r.random(D)
            p_ex = P[ex[i0], np.arange(D)]
            V[i0] = w*V[i0] + c*r1*(p_ex - X[i0])
            X[i0] = clamp(X[i0] + V[i0], lb, ub)
        for i0 in range(N//2, N):
            r1, r2 = r.random(D), r.random(D)
            V[i0] = w*V[i0] + c*r1*(P[i0]-X[i0]) + c2*r2*(g - X[i0])
            X[i0] = clamp(X[i0] + V[i0], lb, ub)
        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        P[upd], FP[upd] = X[upd], FX[upd]
        j = np.argmin(FP)
        if FP[j] < fg:
            fg, g = float(FP[j]), P[j].copy()
    return _pack_result(g, fg, stop, hist_best, hist_div)

# ---------- SL-PSO ----------
def SLPSO(fun, lb, ub, fe_budget, N=50, seed=0):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        idx = np.argsort(FX)
        hist_best.append(float(FX[idx[0]]))
        hist_div.append(mean_centroid_diversity(X))
        for rank_pos in range(1, N):
            i = idx[rank_pos]
            j = idx[r.integers(0, rank_pos)]
            phi = r.random(D)
            X[i] = clamp(X[i] + phi*(X[j]-X[i]), lb, ub)
        FX = eval_pop(fun, X, stop)
    j = np.argmin(FX)
    return _pack_result(X[j], FX[j], stop, hist_best, hist_div)

# ---------- ALC-PSO ----------
def ALCPSO(fun, lb, ub, fe_budget, N=50, seed=0):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    V = r.uniform(-np.abs(ub-lb), np.abs(ub-lb), (N, D)) * 0.1
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    P, FP = X.copy(), FX.copy()
    g = P[np.argmin(FP)].copy(); fg = float(FP.min())
    w_max, w_min, c1, c2 = 0.9, 0.3, 1.7, 1.7
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        frac = stop.fe / max(1, stop.fe_budget)
        w = w_max - (w_max - w_min) * frac
        hist_best.append(fg)
        hist_div.append(mean_centroid_diversity(X))
        idx = np.argsort(FP)
        p = max(2, N//5)
        elite = idx[:p]
        L1, L2 = P[elite[r.integers(0, p)]], P[elite[r.integers(0, p)]]
        leader = L1 if r.random() < 0.5 else L2
        r1, r2 = r.random((N, D)), r.random((N, D))
        V = w*V + c1*r1*(P-X) + c2*r2*(leader - X)
        X = clamp(X + V, lb, ub)
        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        P[upd], FP[upd] = X[upd], FX[upd]
        j = np.argmin(FP)
        if FP[j] < fg:
            fg, g = float(FP[j]), P[j].copy()
    return _pack_result(g, fg, stop, hist_best, hist_div)

# ---------- CSO ----------
def CSO(fun, lb, ub, fe_budget, N=50, seed=0, phi=0.1):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        hist_best.append(float(FX.min()))
        hist_div.append(mean_centroid_diversity(X))
        idx = r.permutation(N)
        for k in range(0, N-1, 2):
            i, j = idx[k], idx[k+1]
            winner, loser = (i, j) if FX[i] < FX[j] else (j, i)
            meanX = X.mean(axis=0)
            r1, r2, r3 = r.random(D), r.random(D), r.random(D)
            X[loser] = clamp(X[loser] + r1*(X[winner]-X[loser]) + phi*(r2*(meanX-X[loser])+r3*(X[winner]-meanX)), lb, ub)
        FX = eval_pop(fun, X, stop)
    j = np.argmin(FX)
    return _pack_result(X[j], FX[j], stop, hist_best, hist_div)

# ---------- DE engine ----------
def DE_engine(fun, lb, ub, fe_budget, N=80, seed=0, variant="JADE"):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N, D))
    stop = StopState(fe_budget)
    FX = eval_pop(fun, X, stop)
    hist_best, hist_div = [], []
    archive = np.empty((0, D))
    muF, muCR = 0.5, 0.5
    H = 20
    MF, MCR = np.full(H, 0.5), np.full(H, 0.5)
    kH = 0
    CRm, Fm = 0.5, 0.5
    N_init = N; N_min = 4
    def cnEpSin_F(t): return 0.5 + 0.3 * math.sin(2 * math.pi * t)
    while stop.fe < stop.fe_budget and X.shape[0] >= 4:
        Np = X.shape[0]
        hist_best.append(float(FX.min()))
        hist_div.append(mean_centroid_diversity(X))
        p = max(2, int(0.1 * Np))
        pbest_pool = np.argsort(FX)[:p]
        if variant in {"SHADE", "LSHADE", "LSHADE_cnEpSin", "jSO"}:
            rH = r.integers(0, H, size=Np)
            CR = np.clip(r.normal(MCR[rH], 0.1), 0, 1)
            F = np.clip(r.standard_cauchy(Np) * 0.1 + MF[rH], 0, 1)
            F[F <= 0] = 0.5
            if variant == "LSHADE_cnEpSin":
                t = stop.fe / max(1, stop.fe_budget)
                F = np.clip(0.5 * F + 0.5 * cnEpSin_F(t), 0, 1)
            if variant == "jSO":
                t = stop.fe / max(1, stop.fe_budget)
                F = np.clip(F + 0.2 * (1 - t), 0, 1)
                CR = np.clip(CR + 0.1 * (1 - t), 0, 1)
        elif variant == "JADE":
            CR = np.clip(r.normal(muCR, 0.1, size=Np), 0, 1)
            F = np.clip(r.standard_cauchy(Np) * 0.1 + muF, 0, 1)
            F[F <= 0] = 0.5
        elif variant == "SaDE":
            CR = np.clip(r.normal(CRm, 0.1, size=Np), 0, 1)
            F = np.clip(r.normal(Fm, 0.1, size=Np), 0, 1)
        else:
            CR = np.full(Np, 0.9); F = np.full(Np, 0.5)
        U = np.empty_like(X)
        for i in range(Np):
            xi = X[i]
            if variant in {"JADE", "SHADE", "LSHADE", "LSHADE_cnEpSin", "jSO"}:
                pbest = X[pbest_pool[r.integers(0, p)]]
                a, b = r.choice(Np, 2, replace=False)
                pool = np.vstack([X, archive]) if archive.size else X
                c = r.integers(0, pool.shape[0])
                vi = xi + F[i] * (pbest - xi) + F[i] * (X[a] - pool[c])
            else:
                idx = np.arange(Np); idx = idx[idx != i]
                a, b, c = r.choice(idx, 3, replace=False)
                vi = X[a] + F[i] * (X[b] - X[c])
            jrand = r.integers(0, D)
            cross = r.random(D) < CR[i]
            cross[jrand] = True
            ui = np.where(cross, vi, xi)
            U[i] = clamp(ui, lb, ub)
        FU = eval_pop(fun, U, stop)
        improved = FU <= FX
        if variant in {"JADE", "SHADE", "LSHADE", "LSHADE_cnEpSin", "jSO"} and np.any(improved):
            archive = np.vstack([archive, X[improved]])
            if archive.shape[0] > 2 * Np:
                archive = archive[r.permutation(archive.shape[0])[:2 * Np]]
        X_new, FX_new = X.copy(), FX.copy()
        X_new[improved], FX_new[improved] = U[improved], FU[improved]
        if variant == "JADE" and np.any(improved):
            muCR = (1 - 0.1) * muCR + 0.1 * np.mean(CR[improved])
            muF = (1 - 0.1) * muF + 0.1 * (np.sum(F[improved] ** 2) / np.sum(F[improved]))
        if variant in {"SHADE", "LSHADE", "LSHADE_cnEpSin", "jSO"} and np.any(improved):
            mF = np.sum(F[improved] ** 2) / max(1e-12, np.sum(F[improved]))
            mCR = np.mean(CR[improved])
            MF[kH], MCR[kH] = mF, mCR
            kH = (kH + 1) % H
        if variant == "SaDE" and np.any(improved):
            CRm = 0.9 * CRm + 0.1 * np.mean(CR[improved])
            Fm = 0.9 * Fm + 0.1 * np.mean(F[improved])
        X, FX = X_new, FX_new
        if variant in {"LSHADE", "LSHADE_cnEpSin", "jSO"}:
            t = stop.fe / max(1, stop.fe_budget)
            N_target = int(round(N_init - (N_init - N_min) * t))
            N_target = max(N_min, min(N_init, N_target))
            if X.shape[0] > N_target:
                keep = np.argsort(FX)[:N_target]
                X, FX = X[keep], FX[keep]
                if archive.shape[0] > 2 * X.shape[0]:
                    archive = archive[r.permutation(archive.shape[0])[:2 * X.shape[0]]]
    best_idx = np.argmin(FX)
    return _pack_result(X[best_idx], FX[best_idx], stop, hist_best, hist_div)

# ---------- CMA-ES ----------
def CMAES(fun, lb, ub, fe_budget, seed=0, sigma0=0.3):
    r = rng(seed); D = lb.size
    lam = 4 + int(3 * np.log(D + 1))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
    cs = (mueff + 2) / (D + mueff + 5)
    c1 = 2 / ((D + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
    xmean = (lb + ub) / 2.0
    sigma = sigma0 * np.mean(ub - lb)
    C = np.eye(D)
    pc = np.zeros(D); ps = np.zeros(D)
    B = np.eye(D); Ddiag = np.ones(D)
    invsqrtC = np.eye(D)
    chiN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))
    stop = StopState(fe_budget)
    best = np.inf
    hist_best, hist_div = [], []
    while stop.fe < stop.fe_budget:
        arz = r.standard_normal((lam, D))
        ary = arz @ (B * Ddiag).T
        arx = clamp(xmean + sigma * ary, lb, ub)
        fx = eval_pop(fun, arx, stop)
        idx = np.argsort(fx)
        arx, ary, arz, fx = arx[idx], ary[idx], arz[idx], fx[idx]
        best = float(fx[0])
        hist_best.append(best)
        hist_div.append(mean_centroid_diversity(arx))
        xold = xmean.copy()
        xmean = np.sum(arx[:mu] * w[:, None], axis=0)
        ymean = (xmean - xold) / sigma
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ ymean)
        hsig = (np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(stop.fe/lam+1))) / chiN) < (1.4+2/(D+1))
        pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*ymean
        artmp = (ary[:mu].T * w)
        C = (1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + cmu*(artmp @ artmp.T)
        sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
        if (stop.fe // lam) % max(1, (D // 10) if D > 0 else 1) == 0:
            C = (C + C.T) / 2
            evals, B = np.linalg.eigh(C)
            Ddiag = np.sqrt(np.maximum(evals, 1e-30))
            invsqrtC = B @ np.diag(1.0 / Ddiag) @ B.T
    return _pack_result(xmean, best, stop, hist_best, hist_div)

# ---------- FL-PSO ----------
def gl12_caputo_katugampola(y_vals, t_vals, alpha, p):
    N = len(y_vals)
    h = max(1e-12, (t_vals[-1] ** p) / max(1, N - 1))
    Dv = np.zeros(N)
    def a_coeff(kj): return (kj+1)**(1-alpha) - kj**(1-alpha)
    def b_coeff(kj):
        return ((kj+1)**(2-alpha)-kj**(2-alpha))/(2-alpha) - 0.5*((kj+1)**(1-alpha)+kj**(1-alpha))
    for k in range(2, N):
        sum_a = 0.0
        sum_b = 0.0
        for j in range(1, k+1):
            sum_a += a_coeff(k-j) * (y_vals[j] - y_vals[j-1]) / h
        for j in range(2, k+1):
            sum_b += b_coeff(k-j) * (y_vals[j] - 2*y_vals[j-1] + y_vals[j-2]) / (h**2)
        Dv[k] = (p**alpha) * (h**(1-alpha)) / Gamma(2-alpha) * sum_a + (p**alpha) * (h**(2-alpha)) / Gamma(2-alpha) * sum_b
    return Dv

def FL_PSO_NUMERIC(fun, lb, ub, N=50, T=2000, seed=0, w=0.7, c1=1.5, c2=1.5, eta=0.08, mu=1.0,
                   use_frac=True, M_mem=10, alpha_mem=0.6, p_frac=1.0, lambda_frac=1.0,
                   use_ou=True, kappa=0.05, use_noise=True, sigma0=0.3, beta=1.0, decay="exp",
                   inject_terms=True, vmax_ratio=0.2):
    rng_ = rng(seed)
    D = lb.size
    x = rng_.uniform(lb, ub, (N, D))
    v = np.zeros_like(x)
    V_hist = [np.zeros_like(v) for _ in range(max(3, M_mem))]
    t_vals = np.linspace(0, 1, max(3, M_mem))
    vmax = vmax_ratio * (ub - lb)
    fx = np.array([fun(xx) for xx in x])
    pbest, fp = x.copy(), fx.copy()
    gidx = np.argmin(fp)
    gbest, fg = pbest[gidx].copy(), float(fp[gidx])
    curve = np.zeros(T)
    div_curve = np.zeros(T)
    for t in range(T):
        r1, r2 = rng_.random((N, D)), rng_.random((N, D))
        residual = -x
        if use_frac:
            frac_term = np.zeros_like(v)
            for i in range(N):
                for d_ in range(D):
                    y_vals = np.array([V_hist[m][i, d_] for m in range(len(V_hist))])
                    Dgl = gl12_caputo_katugampola(y_vals, t_vals, alpha_mem, p_frac)
                    frac_term[i, d_] = Dgl[-1]
            frac_term *= lambda_frac
        else:
            frac_term = 0.0
        drift = -kappa * (x - gbest) if use_ou else 0.0
        if use_noise:
            frac_decay = max(0.0, 1 - t / max(1, T)) if decay == "exp" else 0.5 * (1 + np.cos(np.pi * t / max(1, T)))
            sigma_t = sigma0 * (frac_decay ** beta)
            noise = sigma_t * rng_.standard_normal(v.shape)
        else:
            noise = 0.0
        base = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x) + (eta*mu)*residual
        v = base + frac_term + drift + noise if inject_terms else base
        v = np.clip(v, -vmax, vmax)
        x = clamp(x + v, lb, ub)
        fx = np.array([fun(xx) for xx in x])
        upd = fx < fp
        if np.any(upd):
            pbest[upd], fp[upd] = x[upd], fx[upd]
        gidx = np.argmin(fp)
        if fp[gidx] < fg - 1e-12:
            gbest, fg = pbest[gidx].copy(), float(fp[gidx])
        if use_frac:
            V_hist = V_hist[1:] + [v.copy()]
        curve[t] = fg
        div_curve[t] = mean_centroid_diversity(x)
    return gbest, fg, curve, div_curve

def FL_PSO(fun, lb, ub, fe_budget, N=50, seed=0, use_frac=True, use_ou=True, use_noise=True, mu=1.0, **kwargs):
    T = max(1, int(fe_budget // N) - 1)
    gbest, fg, curve, div_curve = FL_PSO_NUMERIC(
        fun, lb, ub, N=N, T=T, seed=seed, use_frac=use_frac, use_ou=use_ou, use_noise=use_noise, mu=mu, **kwargs
    )
    stop = StopState(fe_budget)
    stop.fe = (1 + len(curve)) * N
    stop.best = fg
    return _pack_result(gbest, fg, stop, curve, div_curve)

DISPATCH = {
    "PSO": PSO, "DMS_PSO": DMSPSO, "CLPSO": CLPSO, "HCLPSO": HCLPSO,
    "SL_PSO": SLPSO, "ALC_PSO": ALCPSO, "CSO": CSO,
    "JADE": lambda f,lb,ub,fe,**kw: DE_engine(f,lb,ub,fe,variant="JADE",**kw),
    "SaDE": lambda f,lb,ub,fe,**kw: DE_engine(f,lb,ub,fe,variant="SaDE",**kw),
    "SHADE": lambda f,lb,ub,fe,**kw: DE_engine(f,lb,ub,fe,variant="SHADE",**kw),
    "L_SHADE": lambda f,lb,ub,fe,**kw: DE_engine(f,lb,ub,fe,variant="LSHADE",**kw),
    "L_SHADE_cnEpSin": lambda f,lb,ub,fe,**kw: DE_engine(f,lb,ub,fe,variant="LSHADE_cnEpSin",**kw),
    "jSO": lambda f,lb,ub,fe,**kw: DE_engine(f,lb,ub,fe,variant="jSO",**kw),
    "CMA_ES": CMAES,
}

ABLATIONS = {
    "FL_PSO_FULL": {"use_frac": True, "use_ou": True, "use_noise": True, "mu": 1.0},
    "FL_PSO_noFrac": {"use_frac": False, "use_ou": True, "use_noise": True, "mu": 1.0},
    "FL_PSO_noOU": {"use_frac": True, "use_ou": False, "use_noise": True, "mu": 1.0},
    "FL_PSO_noNoise": {"use_frac": True, "use_ou": True, "use_noise": False, "mu": 1.0},
    "FL_PSO_noResidual": {"use_frac": True, "use_ou": True, "use_noise": True, "mu": 0.0},
    "FL_PSO_residual_only": {"use_frac": False, "use_ou": False, "use_noise": False, "mu": 1.0},
}

# =============================================================================
# 3. GUI
# =============================================================================

class FLPSO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FL-PSO Benchmark GUI (Enhanced Standalone)")
        self.root.geometry("1540x920")

        self.current_df_stats = None
        self.current_df_runs = None
        self.current_out_dir = None
        self.progress_queue = queue.Queue()
        self.worker_thread = None

        self.create_widgets()
        self.root.after(150, self.poll_queue)

    def create_widgets(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=470)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=3)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_benchmark_tab(notebook)
        self._build_algorithm_tab(notebook)
        self._build_parameter_tab(notebook)
        self._build_flpso_tab(notebook)

        runbar = ttk.Frame(left)
        runbar.pack(fill=tk.X, padx=8, pady=(0,8))
        self.run_btn = ttk.Button(runbar, text="▶ RUN", command=self.run_benchmarks)
        self.run_btn.pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(runbar, text="Export stats CSV", command=self.export_stats_csv).pack(side=tk.LEFT, padx=6)
        ttk.Button(runbar, text="Export runs CSV", command=self.export_runs_csv).pack(side=tk.LEFT, padx=6)
        self.progress = ttk.Progressbar(runbar, orient=tk.HORIZONTAL, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=6)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var).pack(fill=tk.X, padx=10, pady=(0,8))

        result_notebook = ttk.Notebook(right)
        result_notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.table_frame = ttk.Frame(result_notebook)
        self.plot_frame = ttk.Frame(result_notebook)
        self.div_frame = ttk.Frame(result_notebook)
        self.stats_frame = ttk.Frame(result_notebook)
        self.summary_frame = ttk.Frame(result_notebook)
        result_notebook.add(self.table_frame, text="Results")
        result_notebook.add(self.plot_frame, text="Convergence")
        result_notebook.add(self.div_frame, text="Diversity")
        result_notebook.add(self.stats_frame, text="Statistics")
        result_notebook.add(self.summary_frame, text="Summary")

        self._build_results_tab()
        self._build_plot_tab(self.plot_frame, kind="best")
        self._build_plot_tab(self.div_frame, kind="div")

        self.stats_text = ScrolledText(self.stats_frame, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text = ScrolledText(self.summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        console_frame = ttk.LabelFrame(right, text="Console")
        console_frame.pack(fill=tk.BOTH, expand=False, pady=(0,6), padx=6)
        self.console = ScrolledText(console_frame, height=10, bg="black", fg="white")
        self.console.pack(fill=tk.BOTH, expand=True)

        sys.stdout = self.TextRedirector(self.console)
        sys.stderr = self.TextRedirector(self.console)
        self.on_suite_change()

    def _build_benchmark_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Benchmark")
        ttk.Label(frame, text="Suite:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.suite_var = tk.StringVar(value="Classical24")
        suite_combo = ttk.Combobox(frame, textvariable=self.suite_var, values=["Classical24", "Engineering"], state="readonly")
        suite_combo.grid(row=0, column=1, sticky="ew", padx=5)
        suite_combo.bind("<<ComboboxSelected>>", self.on_suite_change)

        ttk.Label(frame, text="Dimension:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.dim_var = tk.StringVar(value="30")
        self.dim_combo = ttk.Combobox(frame, textvariable=self.dim_var, state="readonly")
        self.dim_combo.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(frame, text="Functions / Problems:").grid(row=2, column=0, sticky="nw", padx=5, pady=5)
        self.fid_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=14, exportselection=False)
        self.fid_listbox.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.fid_listbox.yview)
        scroll.grid(row=2, column=2, sticky="ns")
        self.fid_listbox.configure(yscrollcommand=scroll.set)
        self.fid_listbox.bind("<<ListboxSelect>>", self.on_benchmark_select)

        bf = ttk.Frame(frame)
        bf.grid(row=3, column=1, pady=5)
        ttk.Button(bf, text="Select All", command=self.select_all_fids).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Clear All", command=self.clear_all_fids).pack(side=tk.LEFT, padx=5)

        info_frame = ttk.LabelFrame(frame, text="Selected benchmark information")
        info_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=(5, 5))
        self.bench_title_var = tk.StringVar(value="No benchmark selected")
        ttk.Label(info_frame, textvariable=self.bench_title_var, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        self.bench_formula_text = ScrolledText(info_frame, height=7, wrap=tk.WORD)
        self.bench_formula_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.bench_formula_text.insert(tk.END, "Select a function from the list to see its formula and notes.")
        self.bench_formula_text.configure(state=tk.DISABLED)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(4, weight=1)

    def _build_algorithm_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Algorithms")
        ttk.Label(frame, text="Standard Algorithms:").pack(anchor="w", padx=5, pady=(5,0))
        self.alg_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=11, exportselection=False)
        self.alg_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for a in sorted(DISPATCH.keys()):
            self.alg_listbox.insert(tk.END, a)
        ttk.Label(frame, text="FL-PSO Ablations:").pack(anchor="w", padx=5, pady=(5,0))
        self.abl_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=8, exportselection=False)
        self.abl_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for a in ABLATIONS.keys():
            self.abl_listbox.insert(tk.END, a)
        bf = ttk.Frame(frame)
        bf.pack(pady=5)
        ttk.Button(bf, text="Select All Algs", command=lambda: self.select_all_listbox(self.alg_listbox)).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Select All Ablations", command=lambda: self.select_all_listbox(self.abl_listbox)).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Clear All", command=self.clear_all_algs).pack(side=tk.LEFT, padx=5)

    def _build_parameter_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Parameters")
        self.N_var = tk.IntVar(value=50)
        self.FE_var = tk.IntVar(value=50000)
        self.runs_var = tk.IntVar(value=10)
        self.seed_var = tk.IntVar(value=2025)
        self.outdir_var = tk.StringVar(value="./GUI_Results")
        items = [
            ("Population size (N):", self.N_var),
            ("FE budget:", self.FE_var),
            ("Runs per config:", self.runs_var),
            ("Seed base:", self.seed_var),
        ]
        for row, (label, var) in enumerate(items):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=5)
        ttk.Label(frame, text="Output dir:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.outdir_var, width=34).grid(row=4, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_outdir).grid(row=4, column=2, padx=5)
        frame.columnconfigure(1, weight=1)

    def _build_flpso_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="FL-PSO Params")
        self.fl_vars = {
            "w": tk.DoubleVar(value=0.7),
            "c1": tk.DoubleVar(value=1.5),
            "c2": tk.DoubleVar(value=1.5),
            "eta": tk.DoubleVar(value=0.08),
            "M_mem": tk.IntVar(value=10),
            "alpha_mem": tk.DoubleVar(value=0.6),
            "p_frac": tk.DoubleVar(value=1.0),
            "lambda_frac": tk.DoubleVar(value=1.0),
            "kappa": tk.DoubleVar(value=0.05),
            "sigma0": tk.DoubleVar(value=0.3),
            "beta": tk.DoubleVar(value=1.0),
            "vmax_ratio": tk.DoubleVar(value=0.2),
        }
        labels = [
            ("Inertia w", "w"), ("c1", "c1"), ("c2", "c2"), ("Residual η", "eta"),
            ("Memory length", "M_mem"), ("α_mem", "alpha_mem"), ("p_frac", "p_frac"),
            ("λ_frac", "lambda_frac"), ("κ (OU)", "kappa"), ("σ0 (noise)", "sigma0"),
            ("β decay", "beta"), ("vmax ratio", "vmax_ratio"),
        ]
        for i, (label, key) in enumerate(labels):
            ttk.Label(frame, text=label + ":").grid(row=i, column=0, sticky="w", padx=5, pady=4)
            ttk.Entry(frame, textvariable=self.fl_vars[key], width=12).grid(row=i, column=1, sticky="w", padx=5)

    def _build_results_tab(self):
        top = ttk.Frame(self.table_frame)
        top.pack(fill=tk.X, padx=6, pady=6)
        self.filter_suite_var = tk.StringVar(value="All")
        self.filter_fid_var = tk.StringVar(value="All")
        self.filter_alg_var = tk.StringVar(value="All")
        ttk.Label(top, text="Suite:").pack(side=tk.LEFT)
        self.filter_suite_combo = ttk.Combobox(top, textvariable=self.filter_suite_var, width=12, state="readonly")
        self.filter_suite_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(top, text="FID:").pack(side=tk.LEFT)
        self.filter_fid_combo = ttk.Combobox(top, textvariable=self.filter_fid_var, width=12, state="readonly")
        self.filter_fid_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(top, text="Algorithm:").pack(side=tk.LEFT)
        self.filter_alg_combo = ttk.Combobox(top, textvariable=self.filter_alg_var, width=18, state="readonly")
        self.filter_alg_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Apply Filter", command=self.refresh_results_table).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Reset", command=self.reset_filters).pack(side=tk.LEFT, padx=6)

        tree_frame = ttk.Frame(self.table_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))
        self.tree = ttk.Treeview(tree_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        xsb.pack(fill=tk.X, padx=6)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_result_row_select)

    def _build_plot_tab(self, parent, kind="best"):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=6, pady=6)
        controls = {}
        controls["fid_var"] = tk.StringVar(value="All")
        controls["alg_var"] = tk.StringVar(value="All")
        ttk.Label(top, text="FID:").pack(side=tk.LEFT)
        controls["fid_combo"] = ttk.Combobox(top, textvariable=controls["fid_var"], state="readonly", width=12)
        controls["fid_combo"].pack(side=tk.LEFT, padx=5)
        ttk.Label(top, text="Algorithm:").pack(side=tk.LEFT)
        controls["alg_combo"] = ttk.Combobox(top, textvariable=controls["alg_var"], state="readonly", width=18)
        controls["alg_combo"].pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Plot", command=(lambda k=kind: self.redraw_plot(k))).pack(side=tk.LEFT, padx=6)

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        if kind == "best":
            self.best_plot_controls = controls
            self.best_fig = fig
            self.best_ax = ax
            self.best_canvas = canvas
        else:
            self.div_plot_controls = controls
            self.div_fig = fig
            self.div_ax = ax
            self.div_canvas = canvas

    def on_suite_change(self, event=None):
        suite = self.suite_var.get()
        self.fid_listbox.delete(0, tk.END)
        if suite == "Classical24":
            self.dim_combo["values"] = [10, 30, 50]
            self.dim_var.set("30")
            keys = [f"F{i}" for i in range(1, 25)]
        else:
            self.dim_combo["values"] = []
            self.dim_var.set("")
            keys = [f"ENG{i}" for i in range(1, 5)]
        for key in keys:
            self.fid_listbox.insert(tk.END, BENCHMARK_INFO.get(key, {}).get("label", key))
        self.select_all_fids()
        self.on_benchmark_select()

    def select_all_fids(self): self.fid_listbox.select_set(0, tk.END)
    def clear_all_fids(self): self.fid_listbox.selection_clear(0, tk.END)
    def select_all_listbox(self, lb): lb.select_set(0, tk.END)
    def clear_all_algs(self):
        self.alg_listbox.selection_clear(0, tk.END)
        self.abl_listbox.selection_clear(0, tk.END)

    def parse_fid_label(self, value):
        return value.split(" ", 1)[0].strip() if value else value

    def on_benchmark_select(self, event=None):
        sel = self.fid_listbox.curselection()
        if not sel:
            self.bench_title_var.set("No benchmark selected")
            content = "Select a function from the list to see its formula and notes."
        else:
            label = self.fid_listbox.get(sel[0])
            fid = self.parse_fid_label(label)
            info = BENCHMARK_INFO.get(fid, {"label": fid, "name": fid, "formula": "No formula available.", "notes": ""})
            self.bench_title_var.set(info["label"])
            content = f"Name: {info['name']}\n\nFormula / objective:\n{info['formula']}\n\nNotes:\n{info['notes']}"
        self.bench_formula_text.configure(state=tk.NORMAL)
        self.bench_formula_text.delete("1.0", tk.END)
        self.bench_formula_text.insert(tk.END, content)
        self.bench_formula_text.configure(state=tk.DISABLED)

    def browse_outdir(self):
        d = filedialog.askdirectory(initialdir=self.outdir_var.get())
        if d:
            self.outdir_var.set(d)

    def get_flpso_kwargs(self):
        return {k: v.get() for k, v in self.fl_vars.items()}

    def run_benchmarks(self):
        try:
            suite = self.suite_var.get()
            D = int(self.dim_var.get()) if suite == "Classical24" else None
            selected_fids = [self.parse_fid_label(self.fid_listbox.get(i)) for i in self.fid_listbox.curselection()]
            selected_algs = [self.alg_listbox.get(i) for i in self.alg_listbox.curselection()]
            selected_abl = [self.abl_listbox.get(i) for i in self.abl_listbox.curselection()]
            all_algs = selected_algs + selected_abl
            if not selected_fids:
                messagebox.showerror("Error", "No functions selected")
                return
            if not all_algs:
                messagebox.showerror("Error", "No algorithms selected")
                return

            N = int(self.N_var.get())
            FE = int(self.FE_var.get())
            runs = int(self.runs_var.get())
            seed = int(self.seed_var.get())
            out_dir = Path(self.outdir_var.get())
            if suite == "Classical24":
                items = [it for it in suite_classical24(D) if it["fid"] in selected_fids]
            else:
                items = [it for it in suite_engineering() if it["fid"] in selected_fids]
            if not items:
                messagebox.showerror("Error", "No benchmark items")
                return
            total_jobs = len(items) * len(all_algs) * runs
            self.progress.configure(maximum=max(1, total_jobs), value=0)
            self.status_var.set(f"Running {total_jobs} jobs...")
            self.console.delete(1.0, tk.END)
            self.run_btn.configure(state=tk.DISABLED, text="Running...")
            self.worker_thread = threading.Thread(
                target=self._run_worker,
                args=(items, all_algs, N, FE, runs, seed, out_dir, self.get_flpso_kwargs()),
                daemon=True,
            )
            self.worker_thread.start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run_worker(self, items, algs, N, FE, runs, seed, out_dir, fl_kwargs):
        try:
            temp_dispatch = DISPATCH.copy()
            for abl_name, params in ABLATIONS.items():
                p = dict(params)
                temp_dispatch[abl_name] = lambda f, lb, ub, fe, p=p, **kw: FL_PSO(
                    f, lb, ub, fe, N=kw.get("N", 50), seed=kw.get("seed", 0),
                    use_frac=p["use_frac"], use_ou=p["use_ou"], use_noise=p["use_noise"], mu=p["mu"], **fl_kwargs
                )

            run_logs, fun_stats = [], []
            curves_dir = out_dir / "curves"
            div_dir = out_dir / "diversity"
            curves_dir.mkdir(parents=True, exist_ok=True)
            div_dir.mkdir(parents=True, exist_ok=True)

            job_counter = 0
            total_jobs = len(items) * len(algs) * runs
            print(f"Started benchmark run: {len(items)} functions × {len(algs)} algorithms × {runs} runs = {total_jobs} jobs")

            for it in items:
                fid, name, fun = it["fid"], it["name"], it["fun"]
                lb, ub, fopt, suite_name, D = it["lb"], it["ub"], it["fopt"], it["suite"], it["D"]
                print(f"\n=== {suite_name} | {fid} - {name} | D={D} ===")
                for alg in algs:
                    print(f"Running {alg} ...")
                    bests, succ, cpu_s, fe_used = [], [], [], []
                    best_traces, div_traces = [], []
                    for r_i in range(runs):
                        t0 = time.time()
                        try:
                            result = temp_dispatch[alg](fun, lb, ub, FE, N=N, seed=seed + r_i)
                            if len(result) == 5:
                                _, fbest, stop, curve, div_curve = result
                            else:
                                _, fbest, stop, curve = result
                                div_curve = np.array([])
                        except Exception as e:
                            print(f"Error {alg} {fid} run {r_i+1}: {e}")
                            fbest, stop, curve, div_curve = np.inf, StopState(FE), np.array([]), np.array([])
                        t1 = time.time()
                        bests.append(float(fbest))
                        succ.append(1 if (fopt is not None and fbest <= fopt + 1e-6) else 0)
                        cpu_s.append(t1 - t0)
                        fe_used.append(stop.fe)
                        if len(curve) > 0:
                            best_traces.append(np.asarray(curve, float))
                        if len(div_curve) > 0:
                            div_traces.append(np.asarray(div_curve, float))
                        run_logs.append({
                            "suite": suite_name, "D": D, "fid": fid, "function_name": name, "alg": alg,
                            "run": r_i + 1, "best": float(fbest), "success": succ[-1],
                            "fe_used": stop.fe, "cpu_s": t1 - t0,
                        })
                        job_counter += 1
                        self.progress_queue.put({
                            "type": "progress",
                            "value": job_counter,
                            "message": f"{job_counter}/{total_jobs} | {fid} | {alg} | run {r_i+1}/{runs}",
                        })

                    def _avg_curve(curves):
                        if not curves:
                            return None
                        L = max(len(c) for c in curves)
                        def pad(c):
                            if len(c) < L:
                                return np.concatenate([c, np.full(L - len(c), c[-1])])
                            return c[:L]
                        return np.mean(np.vstack([pad(c) for c in curves]), axis=0)

                    avg_best = _avg_curve(best_traces)
                    avg_div = _avg_curve(div_traces)
                    if avg_best is not None:
                        pd.DataFrame({"step": np.arange(1, len(avg_best) + 1), "best": avg_best}).to_csv(
                            curves_dir / f"{suite_name}__{fid}__{alg}__avgcurve.csv", index=False
                        )
                    if avg_div is not None:
                        pd.DataFrame({"step": np.arange(1, len(avg_div) + 1), "div": avg_div}).to_csv(
                            div_dir / f"{suite_name}__{fid}__{alg}__avgdiv.csv", index=False
                        )
                    fun_stats.append({
                        "suite": suite_name, "D": D, "fid": fid, "function_name": name, "alg": alg,
                        "best": np.min(bests), "mean": np.mean(bests), "median": np.median(bests),
                        "std": np.std(bests), "worst": np.max(bests), "avg_cpu_s": np.mean(cpu_s),
                        "avg_fe_used": np.mean(fe_used), "SR": np.mean(succ),
                    })

            self.current_df_stats = pd.DataFrame(fun_stats)
            self.current_df_runs = pd.DataFrame(run_logs)
            self.current_out_dir = out_dir
            self.progress_queue.put({"type": "done"})
        except Exception as e:
            self.progress_queue.put({"type": "error", "message": str(e)})

    def poll_queue(self):
        try:
            while True:
                item = self.progress_queue.get_nowait()
                if item["type"] == "progress":
                    self.progress["value"] = item["value"]
                    self.status_var.set(item["message"])
                elif item["type"] == "done":
                    self.status_var.set("Completed")
                    self.run_btn.configure(state=tk.NORMAL, text="▶ RUN")
                    self.display_results()
                elif item["type"] == "error":
                    self.run_btn.configure(state=tk.NORMAL, text="▶ RUN")
                    self.status_var.set("Error")
                    messagebox.showerror("Error", item["message"])
        except queue.Empty:
            pass
        self.root.after(150, self.poll_queue)

    def display_results(self):
        if self.current_df_stats is None or self.current_df_stats.empty:
            return
        self.populate_filter_controls()
        self.refresh_results_table()
        self.populate_plot_controls()
        self.redraw_plot("best")
        self.redraw_plot("div")
        self._compute_stats()
        self._compute_summary()

    def populate_filter_controls(self):
        df = self.current_df_stats
        suites = ["All"] + sorted(df["suite"].astype(str).unique().tolist())
        fids = ["All"] + sorted(df["fid"].astype(str).unique().tolist())
        algs = ["All"] + sorted(df["alg"].astype(str).unique().tolist())
        self.filter_suite_combo["values"] = suites
        self.filter_fid_combo["values"] = fids
        self.filter_alg_combo["values"] = algs
        self.filter_suite_var.set("All")
        self.filter_fid_var.set("All")
        self.filter_alg_var.set("All")

    def filtered_stats_df(self):
        if self.current_df_stats is None:
            return pd.DataFrame()
        df = self.current_df_stats.copy()
        if self.filter_suite_var.get() != "All":
            df = df[df["suite"] == self.filter_suite_var.get()]
        if self.filter_fid_var.get() != "All":
            df = df[df["fid"] == self.filter_fid_var.get()]
        if self.filter_alg_var.get() != "All":
            df = df[df["alg"] == self.filter_alg_var.get()]
        return df

    def refresh_results_table(self):
        df = self.filtered_stats_df()
        for row in self.tree.get_children():
            self.tree.delete(row)
        if df.empty:
            return
        cols = list(df.columns)
        self.tree["columns"] = cols
        self.tree["show"] = "headings"
        for col in cols:
            self.tree.heading(col, text=col)
            width = 110 if col not in {"function_name", "alg"} else 150
            self.tree.column(col, width=width, stretch=True)
        for _, row in df.iterrows():
            values = []
            for c in cols:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    values.append(f"{v:.6g}")
                else:
                    values.append(v)
            self.tree.insert("", tk.END, values=values)

    def reset_filters(self):
        self.filter_suite_var.set("All")
        self.filter_fid_var.set("All")
        self.filter_alg_var.set("All")
        self.refresh_results_table()

    def populate_plot_controls(self):
        if self.current_df_stats is None or self.current_df_stats.empty:
            return
        fids = ["All"] + sorted(self.current_df_stats["fid"].astype(str).unique().tolist())
        algs = ["All"] + sorted(self.current_df_stats["alg"].astype(str).unique().tolist())
        for controls in [self.best_plot_controls, self.div_plot_controls]:
            controls["fid_combo"]["values"] = fids
            controls["alg_combo"]["values"] = algs
            controls["fid_var"].set(fids[1] if len(fids) > 1 else "All")
            controls["alg_var"].set("All")

    def on_result_row_select(self, event=None):
        item_ids = self.tree.selection()
        if not item_ids:
            return
        vals = self.tree.item(item_ids[0], "values")
        if not vals:
            return
        cols = self.tree["columns"]
        data = dict(zip(cols, vals))
        fid = data.get("fid", "All")
        alg = data.get("alg", "All")
        self.best_plot_controls["fid_var"].set(fid)
        self.best_plot_controls["alg_var"].set(alg)
        self.div_plot_controls["fid_var"].set(fid)
        self.div_plot_controls["alg_var"].set(alg)
        self.redraw_plot("best")
        self.redraw_plot("div")

    def redraw_plot(self, kind="best"):
        if self.current_out_dir is None:
            return
        controls = self.best_plot_controls if kind == "best" else self.div_plot_controls
        ax = self.best_ax if kind == "best" else self.div_ax
        fig = self.best_fig if kind == "best" else self.div_fig
        canvas = self.best_canvas if kind == "best" else self.div_canvas
        ax.clear()
        target_fid = controls["fid_var"].get()
        target_alg = controls["alg_var"].get()
        curve_dir = self.current_out_dir / ("curves" if kind == "best" else "diversity")
        pattern = "*__avgcurve.csv" if kind == "best" else "*__avgdiv.csv"
        files = sorted(curve_dir.glob(pattern)) if curve_dir.exists() else []
        plotted = 0
        for f in files:
            try:
                parts = f.stem.split("__")
                if len(parts) < 4:
                    continue
                _, fid, alg, _ = parts[0], parts[1], parts[2], parts[3]
                if target_fid != "All" and fid != target_fid:
                    continue
                if target_alg != "All" and alg != target_alg:
                    continue
                data = pd.read_csv(f)
                ycol = "best" if kind == "best" else "div"
                if ycol not in data.columns:
                    continue
                if kind == "best":
                    ax.semilogy(data[ycol].values, label=f"{fid} | {alg}")
                else:
                    ax.plot(data[ycol].values, label=f"{fid} | {alg}")
                plotted += 1
            except Exception:
                pass
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness (log)" if kind == "best" else "Mean distance to centroid")
        ax.set_title("Convergence" if kind == "best" else "Diversity")
        ax.grid(True, alpha=0.3)
        if plotted > 0:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No plot available for current selection", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        canvas.draw()

    def _compute_stats(self):
        self.stats_text.delete(1.0, tk.END)
        if self.current_df_stats is None or self.current_df_stats.empty:
            return
        df = self.current_df_stats.copy()
        for (suite, D), g in df.groupby(["suite", "D"]):
            self.stats_text.insert(tk.END, f"\n--- {suite} | D={D} ---\n")
            piv = g.pivot_table(index="fid", columns="alg", values="mean")
            piv = piv.dropna(axis=1, how="all")
            algs = list(piv.columns)
            complete = piv.dropna()
            if complete.shape[0] >= 2 and len(algs) >= 3:
                arrays = [complete[a].values for a in algs]
                stat, p = friedmanchisquare(*arrays)
                self.stats_text.insert(tk.END, f"Friedman test: chi2={stat:.6f}, p={p:.4e}\n")
            else:
                self.stats_text.insert(tk.END, "Friedman test: insufficient common rows.\n")

            if "FL_PSO_FULL" in piv.columns:
                ref = piv["FL_PSO_FULL"]
                for alg in algs:
                    if alg == "FL_PSO_FULL":
                        continue
                    pair = pd.concat([ref, piv[alg]], axis=1, join="inner").dropna()
                    if pair.shape[0] >= 2:
                        try:
                            _, p = wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1])
                            self.stats_text.insert(tk.END, f"Wilcoxon (FL_PSO_FULL vs {alg}): p={p:.4e}\n")
                        except Exception as e:
                            self.stats_text.insert(tk.END, f"Wilcoxon (FL_PSO_FULL vs {alg}): error -> {e}\n")
                    else:
                        self.stats_text.insert(tk.END, f"Wilcoxon (FL_PSO_FULL vs {alg}): insufficient common rows.\n")

    def _compute_summary(self):
        self.summary_text.delete(1.0, tk.END)
        if self.current_df_stats is None or self.current_df_stats.empty:
            return
        df = self.current_df_stats.copy()
        self.summary_text.insert(tk.END, "Overall ranking by mean performance across selected functions\n")
        self.summary_text.insert(tk.END, "=" * 70 + "\n")
        rank_df = df.groupby("alg")["mean"].mean().sort_values().reset_index()
        for i, row in rank_df.iterrows():
            self.summary_text.insert(tk.END, f"{i+1:2d}. {row['alg']:<20} mean={row['mean']:.6g}\n")
        self.summary_text.insert(tk.END, "\nBest algorithm per function\n")
        self.summary_text.insert(tk.END, "=" * 70 + "\n")
        for (suite, fid), g in df.groupby(["suite", "fid"]):
            best_row = g.sort_values("mean").iloc[0]
            self.summary_text.insert(tk.END, f"{suite} | {fid}: {best_row['alg']} (mean={best_row['mean']:.6g}, std={best_row['std']:.6g})\n")

    def export_stats_csv(self):
        if self.current_df_stats is None or self.current_df_stats.empty:
            messagebox.showinfo("Info", "No statistics available.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="stats_summary.csv")
        if path:
            self.current_df_stats.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Saved: {path}")

    def export_runs_csv(self):
        if self.current_df_runs is None or self.current_df_runs.empty:
            messagebox.showinfo("Info", "No run log available.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="run_logs.csv")
        if path:
            self.current_df_runs.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Saved: {path}")

    class TextRedirector:
        def __init__(self, widget):
            self.widget = widget
        def write(self, s):
            self.widget.insert(tk.END, s)
            self.widget.see(tk.END)
        def flush(self):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = FLPSO_GUI(root)
    root.mainloop()
