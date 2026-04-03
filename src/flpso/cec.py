from pathlib import Path
import numpy as np

# =========================
# CEC WRAPPER LOADER
# =========================

def try_load_cec_wrappers(cec2017_dir, cec2022_dir):
    """
    Try to load CEC benchmark implementations.
    Returns dict with available suites.
    """

    wrappers = {
        "cec2017": None,
        "cec2022": None,
    }

    # Example placeholder logic
    if Path(cec2017_dir).exists():
        wrappers["cec2017"] = "AVAILABLE"

    if Path(cec2022_dir).exists():
        wrappers["cec2022"] = "AVAILABLE"

    return wrappers


# =========================
# CEC 2017 SUITE
# =========================

def suite_cec2017_api(D, base_dir):
    """
    Build CEC2017 benchmark suite.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"CEC2017 directory not found: {base_dir}")

    # Placeholder structure (senin gerçek kodunu buraya taşıyacağız)
    funs = []

    # burada senin mevcut loader kodun olacak

    return funs


# =========================
# CEC 2022 SUITE
# =========================

def suite_cec2022_api(D, base_dir):
    """
    Build CEC2022 benchmark suite.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"CEC2022 directory not found: {base_dir}")

    funs = []

    # burada senin mevcut loader kodun olacak

    return funs
