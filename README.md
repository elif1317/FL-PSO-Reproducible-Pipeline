# FL-PSO: Residual-Guided Fractional-Langevin Particle Swarm Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible experimental framework for **Residual-Guided Fractional-Langevin Particle Swarm Optimization (FL-PSO)**.

---

## 🚀 Features

- Modular FL-PSO implementations (full + ablation variants)
- Classical benchmark suite (24 functions)
- Engineering optimization problems
- Config-driven experiment setup
- Reproducible pipeline structure
- Statistical evaluation (Wilcoxon, Friedman, etc.)

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Pipeline

```bash
python scripts/run_full_pipeline.py
```

---

## 📂 Repository Structure

```text
FL-PSO-Reproducible-Pipeline/
├── configs/
├── scripts/
│   └── run_full_pipeline.py
├── src/
│   └── flpso/
│       ├── optimizers.py
│       ├── benchmarks.py
│       ├── config.py
│       └── cec.py
├── results/
├── requirements.txt
├── CITATION.cff
├── README.md
└── LICENSE
```

---

## 📊 Supported Benchmarks

### Classical Functions
- 24 standard optimization functions  
- unimodal, multimodal, hybrid, composite  

### Engineering Problems
- Spring design  
- Pressure vessel  
- Welded beam  
- Speed reducer  

---

## 📁 Data Layout

```text
data/
├── cec2017/
├── cec2022/
└── wrappers/
```

CEC data sources:
- https://github.com/P-N-Suganthan/CEC2017  
- Official CEC2022 sources  

---

## 📈 Output

Results are saved under:

```text
results/
```

Includes:
- per-run logs  
- aggregated statistics  
- convergence curves  
- diversity metrics  
- statistical test results  

---

## 🔁 Reproducibility

- Fixed seed (2025)
- Deterministic hashing-based runs
- No hard-coded paths
- Config-driven experiments

---

## 📖 Citation

```bibtex
@article{DEMIR2026102367,
title = {Residual-guided Fractional-Langevin Particle Swarm Optimization: A hybrid dynamics framework for global optimization},
journal = {Swarm and Evolutionary Computation},
volume = {104},
pages = {102367},
year = {2026},
doi = {https://doi.org/10.1016/j.swevo.2026.102367},
author = {Elif Demir and Yusuf Zeren and Suayip Toprakseven and Alpaslan Demirci}
}
```

See also `CITATION.cff`.

---

## 📬 Contact

For academic collaboration, open an issue or contact the authors.
