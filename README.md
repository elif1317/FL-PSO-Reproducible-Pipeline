# FL-PSO Reproducible Pipeline

**Residual-guided Fractional–Langevin Particle Swarm Optimization**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.swevo.2026.102367-orange)

---

## Overview

This repository provides a **fully reproducible experimental pipeline** for the paper:

> Residual-guided Fractional–Langevin Particle Swarm Optimization: A hybrid dynamics framework for global optimization  
> *Swarm and Evolutionary Computation*, 2026  
> DOI: https://doi.org/10.1016/j.swevo.2026.102367  

The proposed **FL-PSO framework** integrates:

* Fractional-order memory effects  
* Langevin-type stochastic stabilization  
* Residual-guided correction dynamics  

into a unified swarm optimization paradigm.

---

## Purpose of This Repository

This repository is designed to:

* Reproduce all experimental results reported in the paper  
* Provide a modular and extensible benchmarking pipeline  
* Enable fair comparison with classical and modern optimization algorithms  
* Ensure full transparency and reproducibility  

---

## Quick Start

```bash
git clone https://github.com/elif1317/FL-PSO-Reproducible-Pipeline.git
cd FL-PSO-Reproducible-Pipeline
pip install -r requirements.txt

# Run CLI pipeline
python scripts/run_full_pipeline.py

# Optional: launch GUI
python scripts/run_gui.py
```

Estimated runtime:

* Small test: ~2–5 minutes  
* Full benchmark: ~1–3 hours (CPU dependent)

---

## Graphical User Interface (GUI)

This interface provides an interactive layer on top of the reproducible pipeline, enabling exploratory analysis, visualization, and user-friendly experiment configuration beyond the command-line workflow.

### Launch the GUI

```bash
python scripts/run_gui.py
```

### Features

* Interactive selection of benchmark suites and functions  
* Selection of optimization algorithms and FL-PSO ablation variants  
* Adjustable experimental parameters (population size, FE budget, runs, seeds)  
* Real-time progress monitoring and logging  
* Convergence and diversity visualization  
* Statistical summaries and ranking tables  
* Export of results to CSV format  

### Implementation Notes

* The GUI is currently implemented as a standalone module (`gui/` directory)  
* All benchmark functions and algorithms are defined internally  
* No external datasets or configuration files are required for GUI execution  

---

## Mapping to Paper Results

| Paper Element        | Output Location        |
| -------------------- | ---------------------- |
| Benchmark Tables     | `results/classical/`   |
| Ablation Study       | `results/ablation/`    |
| Convergence Figures  | `results/plots/`       |
| Statistical Tests    | `results/statistics/`  |
| Engineering Problems | `results/engineering/` |

---

## Repository Structure

```
FL-PSO-Reproducible-Pipeline/
│
├── configs/              # Experiment configurations
├── scripts/              # Execution scripts (CLI + GUI launcher)
├── src/flpso/            # Core implementation
├── gui/                  # Graphical User Interface
│
├── data/                 # Input data (optional / user-provided)
├── results/              # Generated outputs
│
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
```

---

## Pipeline Architecture

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

## Algorithm Variants

The pipeline supports:

* FL-PSO (Full Model)  
* FL-PSO without fractional memory  
* FL-PSO without OU drift  
* FL-PSO without Langevin perturbation  
* Residual-only PSO  

These enable systematic ablation analysis.

---

## Statistical Evaluation

Implemented statistical methods:

* Wilcoxon signed-rank test  
* Friedman test  
* Holm / Bonferroni corrections  
* Win/Loss analysis  

Output example:

```
results/statistics/
  wilcoxon.csv
  friedman.csv
  rankings.csv
```

---

## Dependencies

Main dependencies:

* numpy  
* scipy  
* pandas  
* matplotlib  
* opfunu  
* tqdm  
* pyyaml  

Tested with:

```
Python >= 3.9
```

Note: Tkinter is required for the GUI and is usually included in standard Python installations.

---

## Data Setup

### Benchmark Functions

* Classical benchmarks: included / auto-generated  
* CEC benchmarks: optional external setup  

Example:

```
data/
  cec2017/
  cec2022/
```

If CEC datasets are not installed, fallback benchmarks are used.

---

## Running Experiments

```bash
python scripts/run_full_pipeline.py
```

Note: Current version initializes and validates the pipeline structure.  
Full experiment modules are modular and extendable via configs.

---

## Example Console Output

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

## Reproducibility Guarantees

* Fixed random seeds  
* Config-driven execution  
* Deterministic pipeline  
* Full logging of all runs  

---

## Extensibility

You can easily:

* Add new algorithms (`src/flpso/`)  
* Add new benchmarks  
* Modify configs (`configs/`)  
* Extend statistical modules  

---

## Repository Status

This repository accompanies a peer-reviewed publication.  
The pipeline is structured for reproducibility and is under active development for extended benchmarks and further GUI enhancements.

---

## Citation

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

## License

MIT License

---

## Acknowledgments

Developed within:

* Yıldız Technical University  
* Research in fractional dynamics and optimization  

---

## Final Note

This repository is intended as a transparent, reproducible, and extensible research platform for hybrid optimization algorithms.
