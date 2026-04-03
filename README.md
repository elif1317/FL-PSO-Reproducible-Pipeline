# FL-PSO Reproducible Pipeline

This repository provides a reproducible experimental pipeline for the proposed **Residual-Guided Fractional-Langevin Particle Swarm Optimization (FL-PSO)** framework.

The project includes:

- modular optimizer implementations,
- benchmark evaluation,
- engineering optimization problems,
- ablation settings,
- reproducible experiment configuration.

---

## Repository Structure

```text
FL-PSO-Reproducible-Pipeline/
├── configs/
├── scripts/
├── src/
│   └── flpso/
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Run Pipeline

Run the full experimental pipeline with:

```bash
python scripts/run_full_pipeline.py
```

## Data Layout

Benchmark-related files should be placed under:

```text
data/
├── cec2017/
├── cec2022/
└── wrappers/
```

## Current Scope

The repository currently contains:

- FL-PSO and baseline optimizer implementations,
- engineering and benchmark problem definitions,
- modular configuration support,
- an experimental pipeline entry point.

## Citation

If you use this repository, please cite the associated article listed in `CITATION.cff`.
