import numpy as np
from dataclasses import dataclass

# =========================
# HELPERS
# =========================

def rng(seed):
    return np.random.default_rng(int(seed))

def clamp(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)

def mean_centroid_diversity(X):
    c = X.mean(axis=0, keepdims=True)
    d = np.linalg.norm(X - c, axis=1)
    return float(d.mean())

def eval_pop(fun, X, stop):
    vals = np.empty(X.shape[0], float)
    for i in range(X.shape[0]):
        vals[i] = fun(X[i])
    stop.fe += X.shape[0]
    return vals


# =========================
# STOP CLASS
# =========================

@dataclass
class StopState:
    fe_budget: int
    fe: int = 0
    best: float = np.inf
    hit_fe: float = np.nan
    success: bool = False


# =========================
# PSO BASELINE
# =========================

def PSO(fun, lb, ub, fe_budget, N=50, seed=0, w=0.72, c1=1.49, c2=1.49, track_div=False):
    r = rng(seed); D = lb.size
    X = r.uniform(lb, ub, (N,D))
    V = r.uniform(-np.abs(ub-lb), np.abs(ub-lb), (N,D))*0.1
    stop = StopState(fe_budget=fe_budget)

    FX = eval_pop(fun, X, stop)
    P = X.copy(); FP = FX.copy()
    g = P[np.argmin(FP)].copy(); fg = FP.min()

    hist_best=[]; hist_div=[]
    while stop.fe < stop.fe_budget:
        if fg < stop.best:
            stop.best = fg
        hist_best.append(float(fg))
        if track_div: hist_div.append(mean_centroid_diversity(X))

        r1 = r.random((N,D)); r2=r.random((N,D))
        V = w*V + c1*r1*(P-X) + c2*r2*(g-X)
        X = clamp(X + V, lb, ub)
        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        P[upd] = X[upd]; FP[upd] = FX[upd]
        j = np.argmin(FP)
        if FP[j] < fg:
            fg = float(FP[j]); g = P[j].copy()

    return g, fg, stop, np.array(hist_best), np.array(hist_div)


# =========================
# FL-PSO NUMERIC CORE
# =========================

def FL_PSO_NUMERIC(fun, lb, ub, N=50, T=2000, seed=0,
                   w=0.7, c1=1.5, c2=1.5, eta=0.08, mu=1.0,
                   early=None, log_run=False,
                   center_mode="none",
                   use_fractional_memory=True,
                   M_mem=10, alpha_mem=0.6,
                   use_ou_drift=True, kappa=0.05,
                   use_langevin_noise=True, sigma0=0.3, beta=1.0, decay="exp",
                   inject_terms=True,
                   vmax_ratio=0.2, stagnation_reset=None,
                   track_div=False):

    rng_ = rng(seed)
    D = lb.size
    x = rng_.uniform(lb, ub, (N, D))
    v = np.zeros_like(x)

    # Fractional memory
    if use_fractional_memory:
        w_mem = 1.0 / (np.arange(1, M_mem + 1, dtype=float) ** float(alpha_mem))
        w_mem = w_mem / w_mem.sum()
        V_hist = [np.zeros_like(v) for _ in range(M_mem)]
    else:
        V_hist = None
        w_mem = None

    vmax = float(vmax_ratio) * (ub - lb)

    fx = np.array([fun(xx) for xx in x])
    pbest, fp = x.copy(), fx.copy()
    gidx = int(np.argmin(fp))
    gbest, fg = pbest[gidx].copy(), float(fp[gidx])

    curve = np.zeros(T, dtype=float)
    div_curve = np.zeros(T, dtype=float) if track_div else None

    no_improve = 0

    for t in range(T):
        r1, r2 = rng_.random((N, D)), rng_.random((N, D))

        # residual
        if center_mode == "centroid":
            center = x.mean(axis=0, keepdims=True)
            residual = -(x - center)
        elif center_mode == "pbest_centroid":
            center = pbest.mean(axis=0, keepdims=True)
            residual = -(x - center)
        else:
            residual = -x

        # fractional memory
        if use_fractional_memory:
            frac_term = np.zeros_like(v)
            for m, w_m in enumerate(w_mem):
                frac_term += w_m * V_hist[m]
        else:
            frac_term = 0.0

        # OU drift
        drift = (-float(kappa)) * (x - gbest) if use_ou_drift else 0.0

        # noise
        if use_langevin_noise:
            if decay == "exp":
                frac = max(0.0, 1.0 - t / max(1, T))
            else:
                frac = 0.5 * (1.0 + np.cos(np.pi * t / max(1, T)))

            sigma_t = float(sigma0) * (frac ** float(beta))
            noise = sigma_t * rng_.standard_normal(v.shape)
        else:
            noise = 0.0

        base = (w * v
                + c1 * r1 * (pbest - x)
                + c2 * r2 * (gbest - x)
                + (eta * float(mu)) * residual)

        if inject_terms:
            v = base + frac_term + drift + noise
        else:
            v = base

        v = np.clip(v, -vmax, vmax)
        x = clamp(x + v, lb, ub)

        fx = np.array([fun(xx) for xx in x])
        upd = fx < fp
        if np.any(upd):
            pbest[upd] = x[upd]
            fp[upd] = fx[upd]

        gidx = int(np.argmin(fp))
        if fp[gidx] < fg:
            gbest, fg = pbest[gidx].copy(), float(fp[gidx])
            no_improve = 0
        else:
            no_improve += 1

        if use_fractional_memory:
            V_hist = [v.copy()] + V_hist[:-1]

        curve[t] = fg

        if track_div:
            div_curve[t] = mean_centroid_diversity(x)

    if not track_div:
        div_curve = np.array([], float)

    return gbest, float(fg), np.asarray(curve, float), np.asarray(div_curve, float)


# =========================
# FL-PSO WRAPPER
# =========================

def FL_PSO(fun, lb, ub, fe_budget, N=50, seed=0,
           w=0.7, c1=1.5, c2=1.5, eta=0.08,
           use_frac=True, M_mem=10, alpha=0.6, mu=1.0,
           use_ou=True, kappa=0.05,
           use_noise=True, sigma0=0.3, beta=1.0, decay="exp",
           inject_terms=True,
           vmax_ratio=0.2, center_mode="none",
           stagnation_reset=None,
           early=None, log_run=False,
           track_div=False):

    T = max(1, int(fe_budget // max(1, N)) - 1)

    gbest, fg, curve, div_curve = FL_PSO_NUMERIC(
        fun, lb, ub,
        N=N, T=T, seed=seed,
        w=w, c1=c1, c2=c2, eta=eta, mu=mu,
        center_mode=center_mode,
        use_fractional_memory=use_frac,
        M_mem=M_mem, alpha_mem=alpha,
        use_ou_drift=use_ou, kappa=kappa,
        use_langevin_noise=use_noise,
        sigma0=sigma0, beta=beta, decay=decay,
        inject_terms=inject_terms,
        vmax_ratio=vmax_ratio,
        stagnation_reset=stagnation_reset,
        track_div=track_div
    )

    stop = StopState(fe_budget=int(fe_budget))
    fe_used = (1 + int(len(curve))) * int(N)
    stop.fe = int(min(int(fe_budget), fe_used))
    stop.best = float(fg)

    return gbest, float(fg), stop, np.asarray(curve, float), np.asarray(div_curve, float)
