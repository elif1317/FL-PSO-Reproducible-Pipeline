# FL-PSO: Residual-Corrected Fractional–Langevin Particle Swarm Optimization

**Reproducible Experimental Pipeline**

This repository contains a **single-file**, fully reproducible experimental pipeline for the **Residual-Corrected Fractional–Langevin Particle Swarm Optimization (FL-PSO)** algorithm presented in the accompanying manuscript.

The pipeline includes:
- Classical benchmark functions
- Optional support for CEC 2017 / CEC 2022 suites (via Python wrappers or input data)
- Fixed function-evaluation (FE) budgets
- Multiple independent runs with stable seeding
- Convergence curves + diversity tracking
- Basic statistical summaries

All numerical results reported in the paper can be reproduced under transparent and fixed evaluation protocols.

## Features

- Single Python script → easy to run and review
- No hard-coded paths → portable & repository-relative
- Graceful handling of missing CEC resources
- Command-line interface with sensible defaults
- Outputs: results tables, convergence curves, runtime info

## Requirements

```bash
pip install -r requirements.txt
