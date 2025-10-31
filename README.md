# PyMC-Marketing: Practical Explorations

Explorations of practical and advanced use cases for [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing), a Bayesian media mix modeling framework.

## Contents

- **Blogs** — In-depth articles on real-world applications
  - Hyperparameter optimization of causal MMM using Bayesian optimization (Optuna + PyMC-Marketing)
  - Time series forecasting with Chronos and PyMC-Marketing

- **Notebooks** — Interactive explorations and experiments
- **Models** — Reusable model implementations
- **Scripts** — Utility scripts and data processing

## Setup

Install dependencies using `uv`:

```bash
uv run pip install -e .
```

## Running Code

Use `uv run` for all commands:

```bash
uv run python scripts/some_script.py
uv run jupyter notebook
uv run pytest
```

## Key Dependencies

- **PyMC-Marketing** — Bayesian media mix models
- **Optuna** — Bayesian hyperparameter optimization
- **Polars** — Data manipulation
- **JAX/PyTensor** — Probabilistic computation
- **ArviZ** — Posterior visualization and diagnostics
