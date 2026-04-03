import math
import numpy as np

# =========================
# CLASSICAL 24 BENCHMARKS
# =========================

def f_sphere(z): return np.sum(z**2)

def f_schwefel_222(z): return np.sum(np.abs(z)) + np.prod(np.abs(z))

def f_powell_sum(z):
    i = np.arange(1, z.size + 1, dtype=float)
    return np.sum(np.abs(z) ** (i + 1))

def f_schwefel_12(z):
    return np.sum(np.cumsum(z) ** 2)

def f_schwefel_221(z):
    return np.max(np.abs(z))

def f_rosenbrock(z):
    return np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)

def f_step(z):
    return np.sum((z + 0.5) ** 2)

def f_quartic_core(z):
    i = np.arange(1, z.size + 1, dtype=float)
    return np.sum(i * (z ** 4))

def f_zakharov(z):
    i = np.arange(1, z.size + 1)
    s1 = np.sum(z ** 2)
    s2 = 0.5 * np.sum(i * z)
    return s1 + s2 ** 2 + s2 ** 4

def f_schwefel_226(z):
    y = z + 420.9687462275036
    return 418.9829 * z.size - np.sum(y * np.sin(np.sqrt(np.abs(y))))

def f_periodic(z):
    return 1.0 + np.sum(np.sin(z) ** 2) - np.exp(-np.sum(z ** 2))

def f_styblinski_tang(z):
    return 0.5 * np.sum(z ** 4 - 16 * z ** 2 + 5 * z)

def f_rastrigin(z):
    return 10.0 * z.size + np.sum(z ** 2 - 10.0 * np.cos(2 * np.pi * z))

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
    x = np.asarray(x, float)
    assert x.size == 2
    a = np.array([-32, -16, 0, 16, 32], dtype=float)
    A = np.array([(ai, aj) for ai in a for aj in a], dtype=float)
    j = np.arange(1, 26, dtype=float)
    denom = j + (x[0] - A[:, 0]) ** 6 + (x[1] - A[:, 1]) ** 6
    return 1.0 / (1 / 500.0 + np.sum(1.0 / denom))

def f_kowalik(x):
    x = np.asarray(x, float)
    assert x.size == 4
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246], float)
    b = 1.0 / np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], float)
    yhat = (x[0] * (b ** 2 + b * x[1])) / (b ** 2 + b * x[2] + x[3])
    return np.sum((a - yhat) ** 2)

def f_six_hump_camel(x):
    x = np.asarray(x, float)
    assert x.size == 2
    x1, x2 = x[0], x[1]
    return (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3.0) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

A_shekel = np.array([
    [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]
], float)

c_shekel = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], float)

def shekel_core(x, m):
    s = 0.0
    for i in range(m):
        s += 1.0 / (c_shekel[i] + np.sum((x - A_shekel[i]) ** 2))
    return -s

def classical_fmin(fid, D):
    if fid in {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F11", "F13", "F14", "F15", "F16", "F17", "F18"}:
        return 0.0
    if fid == "F10":
        return -418.9829 * D
    if fid == "F12":
        return -39.166165703771 * D
    if fid == "F19":
        return 0.998003837794
    if fid == "F20":
        return 3.075e-4
    if fid == "F21":
        return -1.031628453
    if fid == "F22":
        return -10.1532
    if fid == "F23":
        return -10.4028
    if fid == "F24":
        return -10.5364
    return None

def suite_classical24(D):
    funs = []

    def wrap(core, lb, ub, fid, name, Dfix=None):
        if Dfix is not None:
            lbv = lb * np.ones(Dfix)
            ubv = ub * np.ones(Dfix)
        else:
            lbv = lb * np.ones(D)
            ubv = ub * np.ones(D)

        def f(x, core=core):
            return float(core(np.asarray(x, float)))

        fopt = classical_fmin(fid, lbv.size)
        return dict(fid=fid, name=name, fun=f, lb=lbv, ub=ubv, fopt=fopt, suite="CLASSICAL24", D=lbv.size)

    funs += [wrap(f_sphere,          -100, 100,  "F1",  "Sphere")]
    funs += [wrap(f_schwefel_222,    -10,  10,   "F2",  "Schwefel 2.22")]
    funs += [wrap(f_powell_sum,      -1,   1,    "F3",  "Powell Sum")]
    funs += [wrap(f_schwefel_12,     -100, 100,  "F4",  "Schwefel 1.2")]
    funs += [wrap(f_schwefel_221,    -100, 100,  "F5",  "Schwefel 2.21")]
    funs += [wrap(f_rosenbrock,      -30,  30,   "F6",  "Rosenbrock")]
    funs += [wrap(f_step,            -100, 100,  "F7",  "Step")]
    funs += [wrap(f_quartic_core,    -1.28, 1.28,"F8",  "Quartic (core)")]
    funs += [wrap(f_zakharov,        -5,   10,   "F9",  "Zakharov")]
    funs += [wrap(f_schwefel_226,    -500, 500,  "F10", "Schwefel 2.26")]
    funs += [wrap(f_periodic,        -10,  10,   "F11", "Periodic")]
    funs += [wrap(f_styblinski_tang, -5,   5,    "F12", "Styblinski–Tang")]
    funs += [wrap(f_rastrigin,       -5.12,5.12, "F13", "Rastrigin")]
    funs += [wrap(f_ackley,          -32,  32,   "F14", "Ackley")]
    funs += [wrap(f_griewank,        -600, 600,  "F15", "Griewank")]
    funs += [wrap(f_xin_she_yang4,   -10,  10,   "F16", "Xin-She Yang N.4")]
    funs += [wrap(f_penalized_1,     -50,  50,   "F17", "Penalized 1")]
    funs += [wrap(f_penalized_2,     -50,  50,   "F18", "Penalized 2")]
    funs += [wrap(f_foxholes,        -65,  65,   "F19", "Shekel's Foxholes", Dfix=2)]
    funs += [wrap(f_kowalik,         -5,   5,    "F20", "Kowalik", Dfix=4)]
    funs += [wrap(f_six_hump_camel,  -5,   5,    "F21", "Six-Hump Camel", Dfix=2)]
    funs += [wrap(lambda x: shekel_core(np.asarray(x, float), 5),  0, 10, "F22", "Shekel-5", Dfix=4)]
    funs += [wrap(lambda x: shekel_core(np.asarray(x, float), 7),  0, 10, "F23", "Shekel-7", Dfix=4)]
    funs += [wrap(lambda x: shekel_core(np.asarray(x, float), 10), 0, 10, "F24", "Shekel-10", Dfix=4)]
    return funs


# =========================
# ENGINEERING / CONSTRAINED
# =========================

def penalty(f, g_ineq, h_eq=None, rho=1e6):
    """
    f: objective
    g_ineq(x) <= 0 constraints list/array
    h_eq(x) == 0 constraints list/array (optional)
    """
    def F(x):
        x = np.asarray(x, float)
        fx = float(f(x))
        g = np.asarray(g_ineq(x), float)
        viol = np.maximum(g, 0.0)
        pen = rho * np.sum(viol ** 2)
        if h_eq is not None:
            h = np.asarray(h_eq(x), float)
            pen += rho * np.sum(h ** 2)
        return fx + pen
    return F

def suite_engineering():
    funs = []

    # 1) Tension/Compression Spring
    def f_spring(x):
        x1, x2, x3 = x
        return (x3 + 2.0) * x2 * (x1 ** 2)

    def g_spring(x):
        x1, x2, x3 = x
        g1 = 1 - (x2 ** 3 * x3) / (71785 * (x1 ** 4))
        g2 = (4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4)) + 1 / (5108 * x1 ** 2) - 1
        g3 = 1 - (140.45 * x1) / (x2 ** 2 * x3)
        g4 = (x2 + x1) / 1.5 - 1
        return [g1, g2, g3, g4]

    lb = np.array([0.05, 0.25, 2.0])
    ub = np.array([2.0, 1.3, 15.0])
    funs.append(dict(
        fid="ENG1", name="Spring Design (penalty)", fun=penalty(f_spring, g_spring),
        lb=lb, ub=ub, fopt=None, suite="ENGINEERING", D=3
    ))

    # 2) Pressure Vessel
    def f_pv(x):
        x1, x2, x3, x4 = x
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * (x3 ** 2) + 3.1661 * (x1 ** 2) * x4 + 19.84 * (x1 ** 2) * x3

    def g_pv(x):
        x1, x2, x3, x4 = x
        return [
            0.0193 * x3 - x1,
            0.00954 * x3 - x2,
            (math.pi * (x3 ** 2) * x4 + (4 / 3) * math.pi * (x3 ** 3)) - 1296000,
            x4 - 240
        ]

    lb = np.array([0.0625, 0.0625, 10.0, 10.0])
    ub = np.array([5.0, 5.0, 200.0, 240.0])
    funs.append(dict(
        fid="ENG2", name="Pressure Vessel (penalty)", fun=penalty(f_pv, g_pv),
        lb=lb, ub=ub, fopt=None, suite="ENGINEERING", D=4
    ))

    # 3) Welded Beam
    def f_wb(x):
        h, l, t, b = x
        return 1.10471 * h * h * l + 0.04811 * t * b * (14.0 + l)

    def g_wb(x):
        h, l, t, b = x
        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        tau_max = 13600.0
        sigma_max = 30000.0
        delta_max = 0.25
        M = P * (L + l / 2.0)
        R = math.sqrt((l ** 2) / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2 * math.sqrt(2) * h * l * ((l ** 2) / 12.0 + ((h + t) / 2.0) ** 2)
        tau_p = P / (math.sqrt(2) * h * l)
        tau_pp = M * R / J
        tau = math.sqrt(tau_p ** 2 + 2 * tau_p * tau_pp * l / (2 * R) + tau_pp ** 2)
        sigma = 6 * P * L / (b * (t ** 2))
        delta = 4 * P * (L ** 3) / (E * b * (t ** 3))
        Pc = (4.013 * E * math.sqrt((t ** 2) * (b ** 6) / 36.0) / (L ** 2)) * (1 - t / (2 * L) * math.sqrt(E / (4 * G)))
        return [
            tau - tau_max,
            sigma - sigma_max,
            h - b,
            0.10471 * h * h + 0.04811 * t * b * (14 + l) - 5.0,
            0.125 - h,
            delta - delta_max,
            P - Pc
        ]

    lb = np.array([0.1, 0.1, 0.1, 0.1])
    ub = np.array([2.0, 10.0, 10.0, 2.0])
    funs.append(dict(
        fid="ENG3", name="Welded Beam (penalty)", fun=penalty(f_wb, g_wb),
        lb=lb, ub=ub, fopt=None, suite="ENGINEERING", D=4
    ))

    # 4) Speed Reducer
    def f_sr(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return (
            0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
            + 7.4777 * (x6 ** 3 + x7 ** 3)
            + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
        )

    def g_sr(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return [
            27 / (x1 * x2 ** 2 * x3) - 1,
            397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1,
            1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1,
            1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1,
            (1 / (110 * x6 ** 3)) * math.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) - 1,
            (1 / (85 * x7 ** 3)) * math.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) - 1,
            x2 * x3 / 40 - 1,
            5 * x2 / x1 - 1,
            x1 / (12 * x2) - 1,
            (1.5 * x6 + 1.9) / x4 - 1,
            (1.1 * x7 + 1.9) / x5 - 1
        ]

    lb = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0])
    ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
    funs.append(dict(
        fid="ENG4", name="Speed Reducer (penalty)", fun=penalty(f_sr, g_sr),
        lb=lb, ub=ub, fopt=None, suite="ENGINEERING", D=7
    ))

    return funs
