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
    vals = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        try:
            vals[i] = float(fun(X[i]))
        except Exception:
            vals[i] = np.inf
        stop.fe += 1
        if stop.fe >= stop.fe_budget and i < X.shape[0] - 1:
            vals[i + 1:] = np.inf
            break
    return vals


def update_success_state(stop, fg, target=None):
    stop.best = float(min(stop.best, fg))
    if target is not None and fg <= target and not stop.success:
        stop.success = True
        stop.hit_fe = float(stop.fe)


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

def PSO(
    fun,
    lb,
    ub,
    fe_budget,
    N=50,
    seed=0,
    w=0.72,
    c1=1.49,
    c2=1.49,
    track_div=False,
    early=None,
    log_run=False,
):
    r = rng(seed)
    D = lb.size

    X = r.uniform(lb, ub, (N, D))
    V = r.uniform(-np.abs(ub - lb), np.abs(ub - lb), (N, D)) * 0.1
    stop = StopState(fe_budget=int(fe_budget))

    FX = eval_pop(fun, X, stop)
    P = X.copy()
    FP = FX.copy()

    j = int(np.argmin(FP))
    g = P[j].copy()
    fg = float(FP[j])

    hist_best = []
    hist_div = []

    target = None
    if isinstance(early, (int, float)) and np.isfinite(early):
        target = float(early)

    while stop.fe < stop.fe_budget:
        update_success_state(stop, fg, target=target)
        hist_best.append(float(fg))
        if track_div:
            hist_div.append(mean_centroid_diversity(X))

        if target is not None and fg <= target:
            if log_run:
                print(f"[PSO] Early target reached: fg={fg:.6e}, fe={stop.fe}")
            break

        r1 = r.random((N, D))
        r2 = r.random((N, D))

        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        X = clamp(X + V, lb, ub)

        FX = eval_pop(fun, X, stop)
        upd = FX < FP
        if np.any(upd):
            P[upd] = X[upd]
            FP[upd] = FX[upd]

        j = int(np.argmin(FP))
        if FP[j] < fg:
            fg = float(FP[j])
            g = P[j].copy()

        if log_run and len(hist_best) % 50 == 0:
            print(f"[PSO] iter={len(hist_best):4d}, fe={stop.fe:6d}, best={fg:.6e}")

    update_success_state(stop, fg, target=target)

    if not track_div:
        hist_div = []

    return g, fg, stop, np.asarray(hist_best, dtype=float), np.asarray(hist_div, dtype=float)


# =========================
# FL-PSO NUMERIC CORE
# =========================

def FL_PSO_NUMERIC(
    fun,
    lb,
    ub,
    stop,
    N=50,
    seed=0,
    w=0.7,
    c1=1.5,
    c2=1.5,
    eta=0.08,
    mu=1.0,
    early=None,
    log_run=False,
    center_mode="origin",
    use_fractional_memory=True,
    M_mem=10,
    alpha_mem=0.6,
    use_ou_drift=True,
    kappa=0.05,
    use_langevin_noise=True,
    sigma0=0.3,
    beta=1.0,
    decay="exp",
    inject_terms=True,
    vmax_ratio=0.2,
    stagnation_reset=None,
    track_div=False,
):
    rng_ = rng(seed)
    D = lb.size

    x = rng_.uniform(lb, ub, (N, D))
    v = np.zeros_like(x)

    if use_fractional_memory:
        w_mem = 1.0 / (np.arange(1, M_mem + 1, dtype=float) ** float(alpha_mem))
        w_mem = w_mem / w_mem.sum()
        V_hist = [np.zeros_like(v) for _ in range(M_mem)]
    else:
        w_mem = None
        V_hist = None

    vmax = np.asarray(float(vmax_ratio) * (ub - lb), dtype=float)

    fx = eval_pop(fun, x, stop)
    pbest = x.copy()
    fp = fx.copy()

    gidx = int(np.argmin(fp))
    gbest = pbest[gidx].copy()
    fg = float(fp[gidx])

    curve = []
    div_curve = []

    no_improve = 0
    target = None
    if isinstance(early, (int, float)) and np.isfinite(early):
        target = float(early)

    iter_id = 0
    while stop.fe < stop.fe_budget:
        update_success_state(stop, fg, target=target)
        curve.append(float(fg))
        if track_div:
            div_curve.append(mean_centroid_diversity(x))

        if target is not None and fg <= target:
            if log_run:
                print(f"[FL-PSO] Early target reached: fg={fg:.6e}, fe={stop.fe}")
            break

        r1 = rng_.random((N, D))
        r2 = rng_.random((N, D))

        if center_mode in ("origin", "none"):
            residual = -x
        elif center_mode == "centroid":
            center = x.mean(axis=0, keepdims=True)
            residual = -(x - center)
        elif center_mode == "pbest_centroid":
            center = pbest.mean(axis=0, keepdims=True)
            residual = -(x - center)
        else:
            raise ValueError(f"Unknown center_mode: {center_mode}")

        if use_fractional_memory:
            frac_term = np.zeros_like(v)
            for m, w_m in enumerate(w_mem):
                frac_term += w_m * V_hist[m]
        else:
            frac_term = np.zeros_like(v)

        drift = (-float(kappa)) * (x - gbest) if use_ou_drift else np.zeros_like(v)

        if use_langevin_noise:
            tau = iter_id / max(1, (stop.fe_budget // max(1, N)))
            if decay == "exp":
                frac = np.exp(-5.0 * tau)
            elif decay == "linear":
                frac = max(0.0, 1.0 - tau)
            elif decay == "cosine":
                frac = 0.5 * (1.0 + np.cos(np.pi * min(1.0, tau)))
            else:
                raise ValueError(f"Unknown decay mode: {decay}")

            sigma_t = float(sigma0) * (float(frac) ** float(beta))
            noise = sigma_t * rng_.standard_normal(v.shape)
        else:
            noise = np.zeros_like(v)

        base = (
            w * v
            + c1 * r1 * (pbest - x)
            + c2 * r2 * (gbest - x)
            + (eta * float(mu)) * residual
        )

        if inject_terms:
            v = base + frac_term + drift + noise
        else:
            v = base

        v = np.clip(v, -vmax, vmax)
        x = clamp(x + v, lb, ub)

        fx = eval_pop(fun, x, stop)

        upd = fx < fp
        if np.any(upd):
            pbest[upd] = x[upd]
            fp[upd] = fx[upd]

        gidx = int(np.argmin(fp))
        if fp[gidx] < fg:
            gbest = pbest[gidx].copy()
            fg = float(fp[gidx])
            no_improve = 0
        else:
            no_improve += 1

        if use_fractional_memory:
            V_hist = [v.copy()] + V_hist[:-1]

        if stagnation_reset is not None and no_improve >= int(stagnation_reset):
            n_reset = max(1, N // 5)
            reset_idx = rng_.choice(N, size=n_reset, replace=False)
            x[reset_idx] = rng_.uniform(lb, ub, (n_reset, D))
            v[reset_idx] = 0.0

            fx_reset = eval_pop(fun, x[reset_idx], stop)
            pbest[reset_idx] = x[reset_idx]
            fp[reset_idx] = fx_reset

            gidx = int(np.argmin(fp))
            if fp[gidx] < fg:
                gbest = pbest[gidx].copy()
                fg = float(fp[gidx])

            no_improve = 0

            if log_run:
                print(f"[FL-PSO] stagnation reset at iter={iter_id}, fe={stop.fe}")

        if log_run and iter_id % 50 == 0:
            print(f"[FL-PSO] iter={iter_id:4d}, fe={stop.fe:6d}, best={fg:.6e}")

        iter_id += 1

        if stop.fe >= stop.fe_budget:
            break

    update_success_state(stop, fg, target=target)

    if not track_div:
        div_curve = []

    return (
        gbest,
        float(fg),
        stop,
        np.asarray(curve, dtype=float),
        np.asarray(div_curve, dtype=float),
    )


# =========================
# FL-PSO WRAPPER
# =========================

def FL_PSO(
    fun,
    lb,
    ub,
    fe_budget,
    N=50,
    seed=0,
    w=0.7,
    c1=1.5,
    c2=1.5,
    eta=0.08,
    use_frac=True,
    M_mem=10,
    alpha=0.6,
    mu=1.0,
    use_ou=True,
    kappa=0.05,
    use_noise=True,
    sigma0=0.3,
    beta=1.0,
    decay="exp",
    inject_terms=True,
    vmax_ratio=0.2,
    center_mode="origin",
    stagnation_reset=None,
    early=None,
    log_run=False,
    track_div=False,
):
    stop = StopState(fe_budget=int(fe_budget))

    gbest, fg, stop, curve, div_curve = FL_PSO_NUMERIC(
        fun=fun,
        lb=lb,
        ub=ub,
        stop=stop,
        N=N,
        seed=seed,
        w=w,
        c1=c1,
        c2=c2,
        eta=eta,
        mu=mu,
        early=early,
        log_run=log_run,
        center_mode=center_mode,
        use_fractional_memory=use_frac,
        M_mem=M_mem,
        alpha_mem=alpha,
        use_ou_drift=use_ou,
        kappa=kappa,
        use_langevin_noise=use_noise,
        sigma0=sigma0,
        beta=beta,
        decay=decay,
        inject_terms=inject_terms,
        vmax_ratio=vmax_ratio,
        stagnation_reset=stagnation_reset,
        track_div=track_div,
    )

    return (
        gbest,
        float(fg),
        stop,
        np.asarray(curve, dtype=float),
        np.asarray(div_curve, dtype=float),
    )
