# How-to guides

Task-oriented recipes. For explanations of *why* things work this way, see [Concepts](concepts.md).

## Compute an indicator on a LazyFrame

```python
import polars as pl
from polars_ta import momentum

lf = pl.scan_csv("ohlcv.csv")
out = lf.with_columns(momentum.rsi("close")).collect()
```

## Compute indicators per symbol on a multi-asset frame

Every indicator is a plain expression, so per-symbol computation is just
`.over("symbol")` — one expression, grouped execution, no state leaking across
symbol boundaries (the first bars of one symbol never see another symbol's
tail). This holds for all indicators, including the sequential
`map_batches`-based ones (KAMA, PSAR, VPIN, Hurst), and is enforced by
`tests/test_multi_asset.py`.

```python
import polars as pl
from polars_ta import momentum, trend

df = pl.read_parquet("all_symbols.parquet")  # columns: symbol, open, high, low, close, volume
out = df.with_columns(
    momentum.rsi("close").over("symbol").alias("rsi"),
    trend.macd("close").over("symbol").alias("macd"),
)
```

Warm-up nulls restart at each symbol boundary, exactly as if each symbol had
been computed on its own frame.

## Rank symbols cross-sectionally at each timestamp

Every indicator above computes a rolling statistic *through time* for one
symbol via `.over("symbol")`. [`quant.cross_sectional_zscore`](api.md#polars_ta.quant.cross_sectional_zscore)
and [`quant.cross_sectional_rank`](api.md#polars_ta.quant.cross_sectional_rank)
do the opposite: they compare symbols *against each other at the same
instant*, the building block of a factor/ranking strategy. Group by the
timestamp column instead of the symbol column:

```python
import polars as pl
from polars_ta import quant

# Long format: one row per (timestamp, symbol).
df = pl.DataFrame({
    "timestamp": [1, 1, 1, 2, 2, 2],
    "symbol": ["A", "B", "C", "A", "B", "C"],
    "momentum": [1.0, 2.0, 3.0, -1.0, 0.0, 5.0],
})

out = df.with_columns(
    quant.cross_sectional_zscore("momentum").over("timestamp").alias("z"),
    quant.cross_sectional_rank("momentum").over("timestamp").alias("rank_pct"),
)
```

A cross-section with zero spread (every symbol tied) yields null rather than
a divide-by-zero, and `cross_sectional_rank(..., pct=False)` returns a dense
integer rank instead of a `[0, 1]` percentile.

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
