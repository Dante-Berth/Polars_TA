# How-to guides

Task-oriented recipes. For explanations of *why* things work this way, see [Concepts](concepts.md).

## Compute an indicator on a LazyFrame

```python
import polars as pl
from polars_ta import momentum

lf = pl.scan_csv("ohlcv.csv")
out = lf.with_columns(momentum.rsi("close")).collect()
```

## Run on data larger than memory (streaming)

Pass `engine="streaming"` to `.collect()` — no changes to the indicator calls themselves:

```python
out = lf.with_columns(momentum.rsi("close")).collect(engine="streaming")
```

## Clean invalid values before computing indicators

`polars_ta.utils.DataCleaner` detects and repairs `NaN`/`null`/`inf`/excessively large values in numeric columns before they reach an indicator:

```python
from polars_ta.utils import DataCleaner

# Drop any row containing an invalid numeric value
clean_df = DataCleaner.dropna(df)

# Or find which rows are bad, for logging
bad_rows = DataCleaner.get_invalid_indices(df)

# Or repair in place via linear interpolation + forward-fill
healed_df = DataCleaner.approximate_invalid_values(df)
```

## Fill the warm-up period of an indicator

Pass `fillna=True` to any indicator:

```python
from polars_ta import momentum

out = df.with_columns(momentum.rsi("close", fillna=True))
```

See [Concepts → the fillna convention](concepts.md#the-fillna-convention) for what default value each indicator falls back to.

## Use the namespaced class API instead of top-level functions

Every top-level function (`momentum.rsi`, `trend.macd`, ...) has an equivalent `staticmethod` on a `*Indicators` class, if you prefer explicit namespacing:

```python
from polars_ta.momentum import MomentumIndicators

out = df.with_columns(MomentumIndicators.rsi("close"))
```

## Build a custom indicator on top of existing ones

Because everything is a `pl.Expr`, you can freely combine library indicators with your own logic:

```python
from polars_ta import momentum, volatility

out = df.with_columns(
    (momentum.rsi("close") - 50).alias("rsi_centered"),
    (volatility.average_true_range("high", "low", "close") / pl.col("close") * 100)
    .alias("atr_pct"),
)
```

## Benchmark indicator throughput

```bash
uv run python benchmarks/bench_indicators.py
```

Runs a bundle of ~12 indicators across eager, lazy, and streaming engines at 10K/100K/1M rows.
