# Polars TA

[![CI](https://github.com/Dante-Berth/Polars_TA/actions/workflows/ci.yml/badge.svg)](https://github.com/Dante-Berth/Polars_TA/actions/workflows/ci.yml)
[![Docs](https://github.com/Dante-Berth/Polars_TA/actions/workflows/docs.yml/badge.svg)](https://github.com/Dante-Berth/Polars_TA/actions/workflows/docs.yml)

Technical analysis indicators built on [Polars](https://pola.rs) expressions instead of pandas — including retail-standard indicators (RSI, MACD, Bollinger Bands, ...) *and* the market-microstructure/order-flow toolkit used on professional trading desks (VPIN, Kyle's lambda, Roll's spread, Yang-Zhang volatility, multi-scale Hurst regime detection).

📖 **Full documentation:** <https://dante-berth.github.io/Polars_TA/> — see the [changelog](CHANGELOG.md) for notable changes.

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

The few genuinely sequential indicators (notably VPIN's volume-bucketing loop)
run a Numba-JIT-compiled kernel when the optional `speed` extra is installed,
and fall back to an identical pure-Python loop otherwise — same output either
way:

```bash
uv add "polars-ta-lib[speed]"
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

The classic retail toolkit — Bollinger Bands, RSI, MACD and ATR — plotted on real Binance BTCUSDT 5-minute data by [examples/plot_classic_indicators.py](examples/plot_classic_indicators.py):

![BTCUSDT classic indicators: price with Bollinger Bands and SMA, RSI, MACD, and ATR](docs/assets/classic_indicators.png)

The trend & volume toolkit — Ichimoku cloud, ADX with +DI/-DI, Aroon oscillator and OBV — via [examples/plot_trend_volume.py](examples/plot_trend_volume.py):

![BTCUSDT trend and volume: price with Ichimoku cloud, ADX, Aroon oscillator, and OBV](docs/assets/trend_volume.png)

## Modules

| Module | Contents |
|---|---|
| `polars_ta.momentum` | RSI, TSI, Stochastic (+ signal), Stochastic RSI, Ultimate Oscillator, Williams %R, KAMA, ROC, Awesome Oscillator, PPO, PVO, Chande Momentum Oscillator, Fisher Transform |
| `polars_ta.trend` | SMA/EMA/WMA, MACD, ADX (+DI/-DI), Vortex, TRIX, Mass Index, CCI, DPO, KST, STC, Ichimoku, Aroon, Parabolic SAR, Hull Moving Average, SuperTrend, Elder Ray (Bull/Bear Power) |
| `polars_ta.volatility` | ATR, Bollinger Bands, Keltner Channel, Donchian Channel, Ulcer Index |
| `polars_ta.volume` | ADI, OBV, Chaikin Money Flow, Force Index, Ease of Movement, VPT, NVI, Money Flow Index, VWAP, Klinger Volume Oscillator |
| `polars_ta.others` | Daily return, daily log return, cumulative return |
| `polars_ta.calendar` | Day of week, weekend flag, hour/minute of day, time since midnight, month of year, month-end window, bars since session open |
| `polars_ta.quant` | Garman-Klass, Parkinson, Rogers-Satchell & Yang-Zhang volatility, EWMA (RiskMetrics) volatility, rolling z-score, volatility-adjusted momentum, micro-price proxy, rolling Sharpe/Sortino, historical volatility, Amihud illiquidity, multi-scale Hurst ribbon, relative volume, volatility z-score, cross-sectional rank/z-score, regime-conditional composite signal, rolling CVaR & Cornish-Fisher (modified) VaR, rolling max drawdown & Calmar, rolling skew/kurtosis, gain-to-pain & Jarque-Bera, fractional differentiation, rolling autocorrelation & information coefficient, rolling beta / idiosyncratic vol / downside beta, 12-1 momentum factor |
| `polars_ta.microstructure` | VPIN (order-flow toxicity), Roll's implied spread, Corwin-Schultz high-low spread, Kyle's lambda, Hasbrouck's lambda, effective spread, Lee-Ready trade-side classification, Hurst exponent (R/S), half-life of mean reversion, Lo-MacKinlay variance ratio, Shannon entropy, approximate entropy |

Every function also has an equivalent `staticmethod` on a `*Indicators` class (`MomentumIndicators`, `TrendIndicators`, `VolatilityIndicators`, `VolumeIndicators`) if you prefer namespaced access.

Utilities:

- `polars_ta.utils.BaseIndicator` — shared building blocks (`sma`, `ema`, `true_range`, `check_fillna`, `get_min_max`).
- `polars_ta.utils.DataCleaner` — detect and repair NaN/inf/null values in a `DataFrame` (`dropna`, `get_invalid_indices`, `approximate_invalid_values`).

## Conventions

- Column arguments accept either a column name (`str`) or an existing `pl.Expr`.
- Every indicator takes a `fillna: bool = False` flag. When `True`, gaps are forward-filled (and back-filled/defaulted at the start) instead of left as nulls.
- Indicators are pure expressions with no side effects — nothing is evaluated until you call `.collect()` or use them inside `.with_columns(...)`.
- Every indicator also works with Polars' [streaming engine](https://docs.pola.rs/user-guide/lazy/streaming/) (`.collect(engine="streaming")`) for datasets larger than memory.
- An indicator that needs `k` bars of history returns **null** for its first `k-1` rows (the warm-up) — never a fabricated number — and every indicator supports per-symbol computation on multi-asset frames via `.over("symbol")` with no state leaking across symbols (both properties are enforced by the test suite).

## Development

```bash
uv sync --group dev
uv run pytest        # unit + numerical reference tests
uv run ruff check .  # lint
uv run ruff format . # format
```

The test suite enforces four kinds of guarantee:

- `tests/test_reference.py` — cross-checks indicators (RSI, EMA, MACD, SMA, ATR, ADX, Bollinger Bands, Stochastic, Williams %R, ROC, CCI, OBV, MFI) against independent NumPy reference implementations.
- `tests/test_properties.py` — Hypothesis property tests: length preservation, no NaN/inf leakage, and causality (no lookahead).
- `tests/test_multi_asset.py` — `.over("symbol")` on a multi-asset frame matches computing each symbol separately.
- `tests/test_warmup.py` — warm-up rows are null (never fabricated values), and no nulls appear after warm-up on clean data.

## Benchmarks

```bash
uv run python benchmarks/bench_indicators.py
```

Times a bundle of ~12 indicators across eager, lazy, and streaming Polars engines at 10K/100K/1M rows.

## Professional-desk features and real-data example

`polars_ta.microstructure` and the newer parts of `polars_ta.quant` implement order-flow and regime-detection tools that retail TA libraries typically don't cover: VPIN, Kyle's/Hasbrouck's lambda, Roll's implied spread, Yang-Zhang volatility, and a multi-scale Hurst ribbon. These are tested against `tests/fixtures/btcusdt_5m_sample.arrow` — a 5,000-row slice of real Binance BTCUSDT 5-minute OHLCV data — rather than synthetic noise, since the whole point of these indicators is behavior on real market microstructure.

```bash
uv run python examples/plot_regime_dashboard.py
```

Renders a 3-panel dashboard (price, Hurst-ribbon regime shading, Yang-Zhang volatility + VPIN) to `examples/regime_dashboard.png` — a visual sanity check a human can actually read, not just a table of numbers.

![BTCUSDT regime dashboard: price, Hurst-ribbon regime shading, Yang-Zhang volatility and VPIN](docs/assets/regime_dashboard.png)

The liquidity/microstructure toolkit — Roll vs Corwin-Schultz spread, Kyle's lambda and mean-reversion half-life — via [examples/plot_liquidity.py](examples/plot_liquidity.py):

![BTCUSDT liquidity: price, Roll vs Corwin-Schultz spread, Kyle's lambda, and mean-reversion half-life](docs/assets/liquidity.png)

All of the figures above are committed to the repo and regenerable from a single command — run it after changing an indicator to refresh both the `examples/` copies and the `docs/assets/` copies embedded here:

```bash
uv run python examples/generate_all_figures.py
```

## Documentation site

The docs at <https://dante-berth.github.io/Polars_TA/> are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) + [mkdocstrings](https://mkdocstrings.github.io/), following the [Diátaxis](https://diataxis.fr/) framework (getting started / concepts / how-to guides / examples / API reference), and deploy automatically to GitHub Pages on every push to `main` via `.github/workflows/docs.yml`.

To preview locally:

```bash
uv sync --extra docs
uv run mkdocs serve
```
