# Getting started

## Install

Polars TA uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv add polars-ta-lib
```

Or, working inside this repository:

```bash
uv sync
```

## Your first indicator

Every indicator function accepts either a column name (`str`) or an existing `pl.Expr`, and returns a `pl.Expr`. Nothing is computed until you plug it into `.with_columns(...)` and call `.collect()` (or run it on an eager `DataFrame` directly).

```python
import polars as pl
from polars_ta import momentum

df = pl.read_csv("ohlcv.csv")  # columns: open, high, low, close, volume

out = df.with_columns(
    momentum.rsi("close", window=14).alias("rsi_14"),
)
```

## Composing multiple indicators

Because indicators are expressions, you can compute as many as you like in a single `.with_columns(...)` call — Polars will parallelize them across columns.

```python
from polars_ta import momentum, trend, volatility, volume

out = df.lazy().with_columns(
    momentum.rsi("close").alias("rsi_14"),
    trend.macd("close").alias("macd"),
    volatility.average_true_range("high", "low", "close").alias("atr_14"),
    volume.on_balance_volume("close", "volume").alias("obv"),
).collect()
```

See [examples/quickstart.py](https://github.com/Dante-Berth/Polars_TA/blob/main/examples/quickstart.py) in the repository for a runnable version:

```bash
uv run python examples/quickstart.py
```

## Handling missing data

Two independent tools exist for missing/invalid data:

1. **Per-indicator `fillna`.** Every indicator takes `fillna: bool = False`. When `True`, the leading `nulls` produced by the rolling window are forward/backward-filled or replaced with an indicator-appropriate default (e.g. RSI defaults to 50, Williams %R to -50).
2. **`polars_ta.utils.DataCleaner`.** A DataFrame-level utility to detect and repair NaN/inf/null values *before* indicators run — see [How-to guides](how_to_guides.md#clean-invalid-values-before-computing-indicators).

## Next steps

- [Concepts](concepts.md) explains the design choices behind the expression-first API.
- [How-to guides](how_to_guides.md) has task-oriented recipes (streaming, backtesting frames, cleaning data).
- The [API reference](api.md) lists every function with its full signature.
