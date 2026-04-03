from src.flpso.optimizers import FL_PSO, PSO

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
            inject_terms=True,
            **kw
        ),

        "FL_PSO_noFrac": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=False,
            use_ou=True,
            use_noise=True,
            inject_terms=True,
            **kw
        ),

        "FL_PSO_noOU": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=True,
            use_ou=False,
            use_noise=True,
            inject_terms=True,
            **kw
        ),

        "FL_PSO_noNoise": lambda f, lb, ub, fe, **kw: FL_PSO(
            f, lb, ub, fe,
            use_frac=True,
            use_ou=True,
            use_noise=False,
            inject_terms=True,
            **kw
        ),

        "PSO": lambda f, lb, ub, fe, **kw: PSO(
            f, lb, ub, fe,
            **kw
        ),
    }
