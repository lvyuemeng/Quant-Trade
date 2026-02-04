# Quantitative Trade

Quantitative trading self-learning environment with Marimo notebooks.

## Aim

Learn quantitative trading concepts and techniques through hands-on practice, focusing on:

- Understanding market data and price patterns
- Building and visualizing technical indicators
- Developing trading strategies with real Chinese A-share data

## Objectives

1. **Data Acquisition**: Fetch and process Chinese stock market data using akshare
2. **Data Analysis**: Use polars + narwhals for efficient data manipulation
3. **Visualization**: Create interactive K-line charts with plotly
4. **Strategy Development**: Experiment with quantitative trading ideas

## Tech Stack

| Component | Technology |
|-----------|------------|
| Notebook | marimo >= 0.19.4 |
| Data Processing | polars, narwhals |
| Data Source | akshare (Chinese A-shares) |
| Visualization | plotly, seaborn |
| Dev Tools | ruff, mypy |

## Quick Start

```bash
# Install dependencies
uv sync

# Start Marimo notebook server
marimo run notesbooks/*
marimo edit --watch notesbooks/*
```

## Requirements

- `uv`
- `git`
- `just`

# Caveat

To use `BaoStock` source in batch reading, please use in **windows**:

```python
if __name__ == "__main__":
    multiprocessing.freeze_support()
	...
```

Due to the child processes re-importing and re-executing the entire module infinitely.
