# FL-PSO Reproducible Pipeline

This repository provides a **fully reproducible experimental pipeline** for the proposed:

> **Residual-Guided Fractional-Langevin Particle Swarm Optimization (FL-PSO)**

The framework integrates:
- fractional-order memory effects,
- Ornstein–Uhlenbeck drift,
- time-decaying Langevin noise,
- and residual-guided correction dynamics.

---

## 🚀 Features

- Modular FL-PSO implementations (full + ablation variants)
- Classical benchmark suite (24 functions)
- Engineering optimization problems
- Config-driven experiment setup
- Reproducible pipeline structure
- Statistical evaluation support (Wilcoxon, Friedman, etc.)

---

## 📂 Repository Structure

```text
FL-PSO-Reproducible-Pipeline/
├── configs/              # Experiment configurations
├── scripts/              # Main pipeline entry
├── src/flpso/            # Core implementation
│   ├── optimizers.py
│   ├── benchmarks.py
│   ├── config.py
│   └── cec.py
├── results/              # Output (auto-generated)
├── requirements.txt
├── CITATION.cff
├── README.md
└── LICENSE
