# Concepts

## Indicators are expressions, not values

Every function in `polars_ta` returns a `pl.Expr`. It does not accept or return a `Series` directly, and it never triggers computation on its own. This mirrors how Polars itself is designed: build up a plan of expressions, then let the query engine decide how to execute it (single-threaded, multithreaded, or streaming).

The practical benefit is that indicator expressions compose for free:

```python
rsi = momentum.rsi("close")
smoothed_rsi = rsi.rolling_mean(window_size=3)  # nothing "indicator-specific" needed
```

## `str | pl.Expr` inputs

Every column argument accepts either a column name or an already-built expression. Internally, each function does:

```python
close = pl.col(close) if isinstance(close, str) else close
```

This lets you pass a raw column (`"close"`) in the common case, or chain an indicator directly off another expression's output when composing indicators-of-indicators (e.g. `stochrsi`, which is RSI's output fed through a stochastic-style transform).

## The `fillna` convention

Rolling/EWM-based indicators inherently have a warm-up period: the first `window - 1` (or so) rows have no valid value and are `null`. Every indicator exposes a `fillna: bool = False` argument controlling what happens to those rows and to any `inf`/`NaN` produced mid-series (e.g. division by zero in oscillators):

- `fillna=False` (default): leave gaps as `null`. Recommended when you plan to `drop_nulls()` or align indicators against a longer lookback anyway.
- `fillna=True`: forward-fill, then apply an indicator-appropriate default for any values still missing at the start of the series (bounded oscillators default to their neutral value — e.g. RSI → 50, Williams %R → -50 — while unbounded indicators default to 0).

This logic lives once in `BaseIndicator.check_fillna` (`polars_ta/utils.py`) and every indicator calls into it, rather than each indicator improvising its own null-handling.

## Recursive indicators: the `map_batches` escape hatch

Almost every indicator in this library is a pure vectorized Polars expression (rolling windows, EWMs, horizontal aggregations). Two indicators are inherently sequential and can't be expressed that way:

- **KAMA** (Kaufman's Adaptive Moving Average) — each value depends on the *previous KAMA value*, not just raw input columns.
- **Parabolic SAR** — same: each value depends on the previous SAR, trend direction, and extreme point.

For these, the library computes the vectorizable parts (smoothing constants, true range, etc.) as expressions, then drops into a small NumPy loop via `.map_batches(...)` only for the genuinely recursive step. This keeps the "escape hatch" as narrow as possible while staying honest about what Polars' expression engine cannot do in closed form.

## Streaming support

Because indicators are ordinary expressions built from `rolling_*`, `ewm_*`, `shift`, and horizontal aggregations, they work unmodified under Polars' [streaming engine](https://docs.pola.rs/user-guide/lazy/streaming/):

```python
df.lazy().with_columns(momentum.rsi("close")).collect(engine="streaming")
```

`tests/test_indicators.py::test_streaming_engine_matches_default` asserts this produces bit-identical output to the default engine, and `benchmarks/bench_indicators.py` measures the throughput difference at scale.

## Numerical validation strategy

Bounded oscillators (RSI, Stochastic, Williams %R) are checked for their mathematical invariants (e.g. RSI ∈ [0, 100]) directly. A smaller set of indicators (RSI, SMA, ATR, Bollinger Bands) is additionally cross-checked in `tests/test_reference.py` against independent NumPy implementations of the same formulas, to catch numerical regressions that bounds-checking alone would miss (e.g. an off-by-one in a rolling window, or a wrong smoothing constant).

## Retail indicators vs. professional microstructure features

Most of `polars_ta` (`momentum`, `trend`, `volatility`, `volume`) implements the same indicator set found in retail TA libraries like `ta` or `ta-lib` — RSI, MACD, Bollinger Bands, and so on. These operate purely on OHLCV bars and answer "what has price/volume done."

`polars_ta.microstructure`, and the newer half of `polars_ta.quant`, are a different category: **market microstructure and order-flow features**, standard on professional trading/execution desks but essentially absent from retail toolkits, because they require reasoning about *within-bar* trade dynamics rather than just bar-level OHLCV:

- **VPIN** ([`microstructure.vpin`](api.md#polars_ta.microstructure.vpin)) infers buy/sell imbalance from price-change-based bulk volume classification, aggregated into synchronized volume buckets rather than time bars — this is why it takes a `bucket_size` argument instead of a `window` in bars.
- **Kyle's lambda / Hasbrouck's lambda** ([`microstructure.kyle_lambda`](api.md#polars_ta.microstructure.kyle_lambda), [`microstructure.hasbrouck_lambda`](api.md#polars_ta.microstructure.hasbrouck_lambda)) estimate price impact per unit of signed order flow by regressing price changes against a tick-rule proxy for trade direction (the sign of the price change itself, since bar data has no true trade-by-trade direction).
- **Roll's spread** ([`microstructure.roll_spread`](api.md#polars_ta.microstructure.roll_spread)) backs out the effective bid-ask spread purely from the serial covariance of price changes — no quote data required.
- **Yang-Zhang volatility** ([`quant.yang_zhang_volatility`](api.md#polars_ta.quant.yang_zhang_volatility)) is the minimum-variance OHLC volatility estimator that correctly accounts for both overnight jumps and intraday drift, versus simpler close-to-close historical volatility.
- **The Hurst ribbon** ([`quant.hurst_ribbon`](api.md#polars_ta.quant.hurst_ribbon)) and the standalone [`microstructure.hurst_exponent`](api.md#polars_ta.microstructure.hurst_exponent) classify the current regime as trending (H > 0.5) or mean-reverting (H < 0.5) — used to switch strategy family rather than as a standalone trade signal.

None of these are meant as standalone buy/sell signals — they're **regime and risk gauges**: is the market trending or mean-reverting right now, is liquidity thin, is order flow toxic. See [Examples → professional-desk regime dashboard](examples.md#professional-desk-regime-dashboard-on-real-btcusdt-data) for what these look like plotted against real BTCUSDT data, and why that example deliberately doesn't use synthetic data.
