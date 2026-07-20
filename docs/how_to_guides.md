# How-to guides

Task-oriented recipes. For explanations of *why* things work this way, see [Concepts](concepts.md).

## Call indicators as `pl.col(...).ta.<name>()`

Importing `polars_ta` registers a `.ta` namespace on every Polars expression,
so indicators read like native Polars methods instead of free functions:

```python
import polars as pl
import polars_ta  # importing registers the .ta namespace

df = pl.read_csv("ohlcv.csv")

out = df.with_columns(
    pl.col("close").ta.rsi(14).alias("rsi"),
    pl.col("close").ta.macd().alias("macd"),
    pl.col("high").ta.average_true_range("low", "close").alias("atr"),
    pl.col("close").ta.on_balance_volume("volume").alias("obv"),
)
```

The expression you call `.ta` on is bound to the indicator's **first input** —
`close` for most indicators, `high` for the high-anchored ones
(`average_true_range`, `stoch`, `aroon_up`, ...). Every remaining input column
is passed as an ordinary argument, in the same order the free function takes
it. Because each `.ta` method returns a plain `pl.Expr`, everything else in
this guide — `.over("symbol")`, streaming, `fillna` — works through the
namespace unchanged:

```python
out = df.with_columns(
    pl.col("close").ta.rsi(14).over("symbol").alias("rsi")
)
```

The `.ta` methods and the module-level functions
(`momentum.rsi("close", 14)`) are the *same code* — a byte-for-byte-identical
dispatch layer — so pick whichever reads better; there is no behavioural
difference. A few functions that don't take a single price series
(`cross_sectional_zscore`, `cross_sectional_rank`, `regime_conditional_signal`)
stay free-function-only. `polars_ta.TA_INDICATORS` lists every name reachable
via `.ta`.

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
symbol via `.over("symbol")`. [`quant.cross_sectional_zscore`](api/quant.md#polars_ta.quant.cross_sectional_zscore)
and [`quant.cross_sectional_rank`](api/quant.md#polars_ta.quant.cross_sectional_rank)
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

## Add calendar/seasonality features to a feature pipeline

[`polars_ta.calendar`](api/calendar.md#polars_ta.calendar) is the one module that takes a
timestamp column instead of price/volume. Convert an epoch column to a real
`pl.Datetime` first with `pl.from_epoch`, then derive day-of-week, intraday
position, and month-end features:

```python
import polars as pl
from polars_ta import calendar

df = pl.read_parquet("ohlcv.parquet")  # has an integer epoch-ms column
df = df.with_columns(
    pl.from_epoch("timestamp_open", time_unit="ms").alias("ts")
)

out = df.with_columns(
    calendar.day_of_week("ts").alias("dow"),
    calendar.hour_of_day("ts").alias("hour"),
    calendar.is_month_end("ts").alias("is_month_end"),
)
```

`calendar.bars_since_session_open` needs a session-boundary column (a date
column for daily sessions, or your own session-id for intraday sessions with
gaps) and is applied with `.over(...)` so the bar count restarts at zero for
every session:

```python
out = df.with_columns(pl.col("ts").dt.date().alias("date")).with_columns(
    calendar.bars_since_session_open("date").over("date").alias("bar_in_session")
)
```

This module deliberately does not hardcode market-specific trading-session
windows (e.g. FX Asian/London/NY hours) — those are UTC-hour conventions that
vary by instrument. `hour_of_day` / `minute_of_day` give you the raw
building blocks to define your own session boundaries per instrument.

## Regime-conditional trend/mean-reversion switch

[`quant.regime_conditional_signal`](api/quant.md#polars_ta.quant.regime_conditional_signal)
switches between two already-computed signal expressions based on a regime
score, row by row — a hard threshold switch, not a smooth blend. Wire it to
[`quant.hurst_ribbon`](api/quant.md#polars_ta.quant.hurst_ribbon) to trend-follow
when the market is persistent and mean-revert when it isn't:

```python
from polars_ta import quant, trend, volatility

trend_signal = trend.ema_indicator("close", window=10) - trend.ema_indicator(
    "close", window=30
)
reversion_signal = volatility.bollinger_pband("close") - 0.5

out = df.with_columns(
    **quant.hurst_ribbon("close", scales=(16, 32, 64))
).with_columns(
    quant.regime_conditional_signal(
        "h_ribbon_avg", 0.5, trend_signal, reversion_signal
    ).alias("composite_signal")
)
```

`regime` doesn't have to be Hurst — any expression works (ADX, Shannon
entropy, a volatility z-score), and `signal_above`/`signal_below` can be any
two pre-computed indicator expressions. A null `regime` value produces a
null output rather than silently falling back to either branch. See the
["Regime-conditional composite signal"](examples.md#regime-conditional-composite-signal-on-real-btcusdt-data)
example for a full runnable version plotted on real BTCUSDT data.

## Size positions by tail risk, not just volatility

Volatility is symmetric; the left tail and the equity-curve path are what
actually hurt. The `quant` risk block gives you those directly, as rolling
causal expressions:

```python
from polars_ta import quant

out = df.with_columns(
    # Expected shortfall: mean of the worst 5% of returns (a *coherent* risk
    # measure — unlike VaR, it's sub-additive), reported as a positive loss.
    quant.rolling_cvar("close", window=100, alpha=0.05).alias("cvar_5"),
    # Modified (Cornish-Fisher) VaR: the Gaussian VaR quantile corrected for
    # the window's skewness and excess kurtosis, so it doesn't understate
    # crash risk the way symmetric vol does.
    quant.cornish_fisher_var("close", window=100, alpha=0.05).alias("cf_var_5"),
    # Worst peak-to-trough decline over the trailing window (a positive
    # fraction), and return per unit of that pain.
    quant.rolling_max_drawdown("close", window=252).alias("mdd"),
    quant.calmar_ratio("close", window=252).alias("calmar"),
)
```

A common pattern is inverse-risk sizing: scale each bar's target exposure by
`1 / cvar_5` (clipped), so the book leans out exactly as the tail fattens.
`calmar_ratio` and `gain_to_pain` are null in windows where the metric is
undefined (no drawdown, no losing bars) rather than reporting a fabricated
infinity — filter those out before ranking.

Distribution-shape features are leading indicators of regime fragility:
persistent negative [`rolling_skew`](api/quant.md#polars_ta.quant.rolling_skew) and
rising [`rolling_kurtosis`](api/quant.md#polars_ta.quant.rolling_kurtosis) flag a
market becoming crash-prone *before* realized volatility moves.

## Whiten a feature and monitor its alpha decay

A raw price series is non-stationary (it carries all the memory); its return
series is stationary but has thrown that memory away.
[`quant.frac_diff`](api/quant.md#polars_ta.quant.frac_diff) sits between the two —
fractional differentiation (López de Prado, *Advances in Financial ML*, ch. 5)
makes a series (approximately) stationary while retaining most of its long
memory:

```python
from polars_ta import quant

out = df.with_columns(
    # d in (0, 1): d->1 is an ordinary log return (stationary, memoryless),
    # d->0 keeps almost all memory. d~0.3-0.5 often passes an ADF test while
    # still predicting.
    quant.frac_diff("close", d=0.4, window=100).alias("fd_close"),
)
```

To check whether a signal still works, track its **information coefficient** —
the rolling correlation between the signal and the realized *forward* return.
Build the forward return explicitly, then correlate:

```python
out = df.with_columns(
    my_signal.alias("signal"),
    pl.col("close").pct_change().shift(-5).alias("fwd_ret_5"),
).with_columns(
    quant.rolling_ic("signal", "fwd_ret_5", window=100).alias("ic"),
)
```

!!! warning "The IC series is forward-looking by construction"
    `rolling_ic` correlates against a *future* return, so it is a research /
    monitoring diagnostic only — a decaying IC is the earliest sign the alpha
    is dying — and must **never** be fed back in as a live trading input. That
    would be look-ahead leakage.

## Build a factor book: beta, idiosyncratic vol, downside beta, momentum

For a cross-sectional/factor strategy you need each asset's relationship to a
benchmark *through time*, carried on the same frame as a `benchmark` price
column:

```python
from polars_ta import quant

out = df.with_columns(
    # Market beta of the asset's returns on the benchmark's.
    quant.rolling_beta_to("close", "benchmark", window=60).alias("beta"),
    # The vol a beta hedge leaves behind — the tradeable, asset-specific risk.
    quant.idiosyncratic_vol("close", "benchmark", window=60).alias("idio_vol"),
    # Beta estimated only on bars where the benchmark fell (Ang-Chen): the
    # regime that matters for tail hedging, which symmetric beta averages away.
    quant.downside_beta("close", "benchmark", window=60).alias("down_beta"),
    # Jegadeesh-Titman "12-1" momentum: return over the lookback but skipping
    # the most recent month, to drop short-term reversal.
    quant.momentum_12_1("close", lookback=252, skip=21).alias("mom_12_1"),
)
```

Every one of these is a per-symbol rolling expression, so on a long-format
multi-asset frame apply it with `.over("symbol")`, then rank the momentum
factor across symbols at each timestamp by chaining the cross-sectional
helpers above:

```python
out = df.with_columns(
    quant.momentum_12_1("close").over("symbol").alias("mom")
).with_columns(
    quant.cross_sectional_rank("mom").over("timestamp").alias("mom_rank")
)
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
