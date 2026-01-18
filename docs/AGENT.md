# Quantitative Trading Project

## Overview
Self-learning environment for quantitative trading strategies using Marimo notebooks.

## Tech Stack
- marimo - notebook interface
- polars - dataframe library
- narwhals - pandas/polars compatibility
- akshare - Chinese market data

## Quick Start
```bash
uv sync        # install deps
marimo run path-to.py     # start notebook server
```

## Code Patterns

Emphasize on:
- Type consistency
- Logic consistency and robustness
- Functional style as could

### Fetching Data
```python
import akshare as ak
import narwhals as nw

df = nw.from_native(ak.stock_zh_a_spot(), eager_only=True)
```

### Polars Operations
```python
df.filter(pl.col("涨跌幅") > 5)
```

## Configuration

See `pyproject.toml` for dependencies.

## Commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Install dependencies |
| `marimo run` | Notebook server |
| `pytest tests/` | Run tests |
| `ruff check .` | Lint code |
| `ruff format .` | Format code |
| `mypy .` | Type check |

## Notes
- Keep notebooks in `notebooks/`
- Use narwhals when interfacing with akshare
- Tests live in `tests/`
- Utility modules in `notebooks/utils/`
