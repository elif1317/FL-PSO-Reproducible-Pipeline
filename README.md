# 🚀 FL-PSO Reproducible Pipeline  
**Residual-guided Fractional–Langevin Particle Swarm Optimization**

---

## 📌 Overview

This repository provides a **fully reproducible experimental pipeline** for the paper:

> **Residual-guided Fractional–Langevin Particle Swarm Optimization: A hybrid dynamics framework for global optimization**  
> Swarm and Evolutionary Computation, 2026  
> DOI: https://doi.org/10.1016/j.swevo.2026.102367

The proposed **FL-PSO framework** integrates:
- Fractional-order memory effects  
- Langevin-type stochastic stabilization  
- Residual-guided correction dynamics  

into a unified swarm optimization scheme.

---

## 🎯 Purpose of This Repository

This repository is designed to:

- ✅ Reproduce all experimental results reported in the paper  
- ✅ Provide a modular and extensible benchmarking pipeline  
- ✅ Enable fair comparison with classical and modern optimization algorithms  
- ✅ Ensure **full transparency and reproducibility**

---

## ⚡ Quick Start (30 seconds)

```bash
git clone https://github.com/elif1317/FL-PSO-Reproducible-Pipeline.git
cd FL-PSO-Reproducible-Pipeline
pip install -r requirements.txt

python scripts/run_full_pipeline.py
```

### ✔ Expected Output

```
results/
  classical/
  ablation/
  engineering/
  statistics/
  plots/
```

Includes:
- Per-run logs  
- Aggregated results  
- Statistical test outputs  
- Convergence plots  

---

## 🧠 What This Repository Reproduces

| Paper Component | Reproduced |
|------|--------|
| Classical benchmark results (CEC-style) | ✅ |
| Ablation study (FL-PSO components) | ✅ |
| Convergence behavior analysis | ✅ |
| Statistical comparison (Wilcoxon, Friedman) | ✅ |
| Engineering optimization cases | ✅ |

---

## 🏗️ Repository Structure

```
FL-PSO-Reproducible-Pipeline/
│
├── configs/              # Experiment configurations
├── scripts/              # Execution scripts (entry points)
├── src/flpso/            # Core implementation
│
├── data/                 # (User-provided or auto-created)
├── results/              # (Generated outputs)
│
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
```

---

## ⚙️ Pipeline Architecture

```
Configuration
     ↓
Benchmark Functions / Engineering Problems
     ↓
FL-PSO Variants + Baselines
     ↓
Multiple Independent Runs
     ↓
Statistical Analysis
     ↓
Tables + Plots + Logs
```

---

## 🧪 Algorithm Variants

The following configurations are supported:

- **FL-PSO (Full Model)**  
- FL-PSO w/o fractional memory  
- FL-PSO w/o OU drift  
- FL-PSO w/o Langevin perturbation  
- Residual-only PSO  

These enable **systematic ablation analysis**.

---

## 📊 Statistical Evaluation

The pipeline includes:

- Wilcoxon signed-rank test  
- Friedman test  
- Holm / Bonferroni corrections  
- Win/Loss analysis  

### Output

```
results/statistics/
  wilcoxon.csv
  friedman.csv
  rankings.csv
```

---

## 📦 Dependencies

Main dependencies:

- numpy  
- scipy  
- pandas  
- matplotlib  
- opfunu  
- tqdm  
- pyyaml  

Tested with:

```
Python >= 3.9
```

---

## 📁 Data Setup

### Benchmark Functions

- Classical benchmarks: included or generated automatically  
- CEC benchmarks: optional external setup  

If required:

```
data/
  cec2017/
  cec2022/
```

> ⚠️ If CEC datasets are not installed, the pipeline will fallback to available functions.

---

## ▶️ Running Experiments

```bash
python scripts/run_full_pipeline.py
```

---

## 📈 Example Console Output

```
[INFO] Root directory: ...
[INFO] Output directory: ...
[INFO] Loaded algorithms: FL-PSO, PSO, DE
[INFO] Benchmark: CEC2022
[INFO] Seed: 42

[INFO] Running experiments...
[INFO] Completed successfully.
```

---

## 🔬 Reproducibility Guarantees

- Fixed random seeds  
- Config-driven experiment setup  
- Deterministic evaluation pipeline  
- Explicit logging of all runs  

---

## 📊 Generated Outputs

```
results/
  logs/
  tables/
  statistics/
  plots/
```

Includes:

- Convergence curves  
- Performance tables  
- Statistical comparisons  

---

## 🧩 Extensibility

You can easily:

- Add new algorithms (`src/flpso/`)  
- Add new benchmark functions  
- Modify experiment settings (`configs/`)  
- Extend statistical analysis  

---

## 📚 Citation

If you use this repository, please cite:

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

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgments

Developed within the framework of:

- Yıldız Technical University  
- Research in fractional dynamics and optimization  

---

## 🚀 Final Note

This repository is intended to serve as a **transparent, reproducible, and extensible research platform** for hybrid optimization algorithms.
