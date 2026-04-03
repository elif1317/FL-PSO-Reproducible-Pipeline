from src.flpso.optimizers import FL_PSO, PSO

# =========================
# DEFAULT HYPERPARAMETERS
# =========================

FL_PSO_DEFAULTS = dict(
    w=0.7,
    c1=1.5,
    c2=1.5,
    eta=0.08,
    alpha=0.6,
    mu=1.0,
    kappa=0.05,
    sigma0=0.3,
    beta=1.0,
    decay="exp",
    inject_terms=True,
)

PSO_DEFAULTS = dict(
    w=0.72,
    c1=1.49,
    c2=1.49,
)

# =========================
# ALGORITHM CONFIGURATION
# =========================

def get_algorithms():
    return {
        "FL_PSO_FULL": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=True,
            use_ou=True,
            use_noise=True,
            **FL_PSO_DEFAULTS,
            **kw
        ),

        "FL_PSO_NO_FRAC": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=False,
            use_ou=True,
            use_noise=True,
            **FL_PSO_DEFAULTS,
            **kw
        ),

        "FL_PSO_NO_OU": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=True,
            use_ou=False,
            use_noise=True,
            **FL_PSO_DEFAULTS,
            **kw
        ),

        "FL_PSO_NO_NOISE": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=True,
            use_ou=True,
            use_noise=False,
            **FL_PSO_DEFAULTS,
            **kw
        ),

        "FL_PSO_BASE": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=False,
            use_ou=False,
            use_noise=False,
            inject_terms=False,
            **FL_PSO_DEFAULTS,
            **kw
        ),

        "PSO": lambda f, lb, ub, fe, **kw: PSO(
            f, lb, ub, fe,
            **PSO_DEFAULTS,
            **kw
        ),
    }
