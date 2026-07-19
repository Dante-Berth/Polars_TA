# Polars TA

[![CI](https://github.com/Dante-Berth/Polars_TA/actions/workflows/ci.yml/badge.svg)](https://github.com/Dante-Berth/Polars_TA/actions/workflows/ci.yml)

Technical analysis indicators built on [Polars](https://pola.rs) expressions instead of pandas.

Every indicator is a plain `pl.Expr`, so it composes naturally with `.with_columns(...)`, works on both `DataFrame` and `LazyFrame`, and runs on Polars' multithreaded, vectorized engine — no row-by-row Python loops (aside from a couple of genuinely recursive indicators like KAMA and PSAR, which use `map_batches`).

## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Or install as a dependency in another project:

```bash
uv add polars-ta-lib
```

## Quickstart

```python
import polars as pl
from polars_ta import momentum, trend, volatility, volume

df = pl.read_csv("ohlcv.csv")  # columns: open, high, low, close, volume

out = df.with_columns(
    momentum.rsi("close").alias("rsi_14"),
    trend.macd("close").alias("macd"),
    volatility.average_true_range("high", "low", "close").alias("atr_14"),
    volume.on_balance_volume("close", "volume").alias("obv"),
)
```

See [examples/quickstart.py](examples/quickstart.py) for a fuller example, or run it directly:

```bash
uv run python examples/quickstart.py
```

## Modules

| Module | Contents |
|---|---|
| `polars_ta.momentum` | RSI, TSI, Stochastic (+ signal), Stochastic RSI, Ultimate Oscillator, Williams %R, KAMA, ROC, Awesome Oscillator, PPO, PVO |
| `polars_ta.trend` | SMA/EMA/WMA, MACD, ADX (+DI/-DI), Vortex, TRIX, Mass Index, CCI, DPO, KST, STC, Ichimoku, Aroon, Parabolic SAR |
| `polars_ta.volatility` | ATR, Bollinger Bands, Keltner Channel, Donchian Channel, Ulcer Index |
| `polars_ta.volume` | ADI, OBV, Chaikin Money Flow, Force Index, Ease of Movement, VPT, NVI, Money Flow Index, VWAP |
| `polars_ta.others` | Daily return, daily log return, cumulative return |
| `polars_ta.quant` | Garman-Klass volatility, rolling z-score, volatility-adjusted momentum, micro-price proxy, rolling Sharpe/Sortino, historical volatility, Amihud illiquidity |

Every function also has an equivalent `staticmethod` on a `*Indicators` class (`MomentumIndicators`, `TrendIndicators`, `VolatilityIndicators`, `VolumeIndicators`) if you prefer namespaced access.

Utilities:

- `polars_ta.utils.BaseIndicator` — shared building blocks (`sma`, `ema`, `true_range`, `check_fillna`, `get_min_max`).
- `polars_ta.utils.DataCleaner` — detect and repair NaN/inf/null values in a `DataFrame` (`dropna`, `get_invalid_indices`, `approximate_invalid_values`).

## Conventions

- Column arguments accept either a column name (`str`) or an existing `pl.Expr`.
- Every indicator takes a `fillna: bool = False` flag. When `True`, gaps are forward-filled (and back-filled/defaulted at the start) instead of left as nulls.
- Indicators are pure expressions with no side effects — nothing is evaluated until you call `.collect()` or use them inside `.with_columns(...)`.
- Every indicator also works with Polars' [streaming engine](https://docs.pola.rs/user-guide/lazy/streaming/) (`.collect(engine="streaming")`) for datasets larger than memory.

## Development

```bash
uv sync --group dev
uv run pytest        # unit + numerical reference tests
uv run ruff check .  # lint
uv run ruff format . # format
```

`tests/test_reference.py` cross-checks a sample of indicators (RSI, SMA, ATR, Bollinger Bands) against independent NumPy reference implementations to catch numerical regressions.

## Benchmarks

```bash
uv run python benchmarks/bench_indicators.py
```

Times a bundle of ~12 indicators across eager, lazy, and streaming Polars engines at 10K/100K/1M rows.
