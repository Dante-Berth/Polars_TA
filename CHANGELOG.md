# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`.ta` expression namespace.** Importing `polars_ta` now registers a `.ta`
  accessor on every Polars expression, so all 126 single-price-series
  indicators are callable as `pl.col("close").ta.rsi(14)` in addition to the
  existing `momentum.rsi("close", 14)` free-function form. The calling
  expression is bound to the indicator's first input; remaining columns are
  passed as arguments (`pl.col("high").ta.average_true_range("low", "close")`).
  It is a thin, byte-for-byte-identical dispatch layer over the same functions
  — `.over(...)`, streaming and `fillna` work through it unchanged — generated
  from the module functions at import time so it can't drift. Cross-sectional
  and regime-composite helpers (which don't take a single price series) stay
  free-function-only. `polars_ta.TA_INDICATORS` lists every exposed name. New
  "Call indicators as `pl.col(...).ta.<name>()`" how-to guide.

- New `quant` portfolio/risk features across four families that retail TA
  libraries typically omit — all pure Polars expressions with the library's
  standard null warm-up and `.over("symbol")` guarantees:
  - **Tail risk & drawdown:** `rolling_cvar` (Conditional VaR / expected
    shortfall — a coherent risk measure, unlike VaR),
    `cornish_fisher_var` (skew/kurtosis-adjusted "modified VaR" that captures
    crash risk symmetric vol misses), `rolling_max_drawdown` (causal, trailing
    peak), and `calmar_ratio` (annualized return per unit of worst-case path
    pain).
  - **Distribution shape:** `rolling_skew`, `rolling_kurtosis` (excess),
    `gain_to_pain` (net move per unit of downside suffered), and `jarque_bera`
    (a single normality-test statistic that rises when either tail asymmetry
    or fat-tailedness appears) — leading indicators of regime fragility that
    complement the entropy tools.
  - **Signal conditioning:** `frac_diff` (fixed-width fractional
    differentiation, López de Prado *AFML* ch. 5 — stationarity while
    retaining long memory), `rolling_autocorr`, and `rolling_ic` (rolling
    information coefficient for alpha-decay monitoring; documented as a
    forward-looking research diagnostic, never a live input).
  - **Cross-sectional / factor plumbing:** `rolling_beta_to`,
    `idiosyncratic_vol` (residual vol a beta hedge leaves behind),
    `downside_beta` (Ang-Chen / Bawa-Lindenberg, estimated only on
    down-benchmark bars), and `momentum_12_1` (Jegadeesh-Titman skip-a-month
    momentum factor). Combine the cross-sectional ones with
    `.over(timestamp)` and the existing `cross_sectional_rank` / `zscore`.

- New `quant.regime_conditional_signal`: a compositional building block that
  switches between two pre-computed signal expressions based on a regime
  score (a hard threshold switch, not a smooth blend) — e.g. trend-follow
  when `hurst_ribbon`'s `h_ribbon_avg >= 0.5`, mean-revert otherwise. Not
  wired specifically to Hurst: `regime` and the two signal branches can be
  any expressions. A null regime value produces null output rather than
  silently falling back to either branch. New
  `examples/plot_regime_conditional_signal.py` capstone figure wiring it to
  the existing Hurst ribbon on real BTCUSDT data, plus a
  "Regime-conditional trend/mean-reversion switch" how-to guide.

- New `polars_ta.calendar` module: `day_of_week`, `is_weekend`,
  `hour_of_day`, `minute_of_day`, `time_since_midnight`, `month_of_year`,
  `is_month_end`, and `bars_since_session_open`. The first module whose
  input is a timestamp column rather than price/volume — deliberately does
  not hardcode market-specific session windows (e.g. FX Asian/London/NY
  hours), since those are UTC-hour conventions that vary by instrument.
  New "Add calendar/seasonality features" how-to guide.

- New microstructure/regime indicators: `microstructure.shannon_entropy`
  (normalized entropy of the binned return distribution — distributional
  concentration, not directional persistence) and
  `microstructure.approximate_entropy` (pattern predictability; documented
  as O(window²) per row, so it's meant for small windows as a slow-moving
  regime gate, not a per-bar signal over a large lookback). New
  `examples/plot_entropy.py` figure and docs section showing both against
  the existing Hurst ribbon on real BTCUSDT data.

- New trend indicators: `trend.supertrend` (ATR-banded stop-and-reverse
  line), `trend.hull_moving_average` (lower-lag alternative to SMA/EMA), and
  `trend.elder_bull_power` / `trend.elder_bear_power` (Elder Ray).
- New momentum indicators: `momentum.cmo` (Chande Momentum Oscillator) and
  `momentum.fisher_transform` (Ehlers' Fisher Transform, using the original
  double-EMA-damped recursion rather than a one-shot `atanh`, which would
  saturate at the clip boundary on noisy real data).
- New volume indicator: `volume.klinger_volume_oscillator`.
- New quant indicators: `quant.ewma_volatility` (RiskMetrics-style
  exponentially-weighted volatility) and `quant.cross_sectional_zscore` /
  `quant.cross_sectional_rank` — the library's first cross-sectional
  indicators, comparing symbols against each other at each timestamp
  (`.over(timestamp_column)`) rather than through time for one symbol.
- New microstructure indicator: `microstructure.lee_ready_trade_sign`
  (Lee-Ready buy/sell trade classification, falling back to the tick test on
  bar data without quotes).
- A new documentation example (`examples/plot_new_indicators.py`) and
  "New indicators" section in the docs covering all of the above on real
  BTCUSDT data, plus a "Rank symbols cross-sectionally" how-to guide.
- Reference, warm-up, smoke, and multi-asset (`.over("symbol")`) test
  coverage for every new indicator, keeping the suite at 100% line coverage.

### Changed

- **The str-or-Expr argument convention is now universal.** Every indicator's
  column arguments accept either a column name (`str`) or an existing `pl.Expr`
  — previously ~1/3 of functions coerced their first argument with an
  unconditional `pl.col(...)` and rejected an `Expr`. This is what makes the
  new `.ta` namespace work uniformly across the whole library, and is enforced
  by the test suite.

### Fixed

- **`momentum.kama` and the `trend.adx` family reported fabricated values
  during their warm-up window.** KAMA's efficiency-ratio guard turned the null
  ("not enough data yet") denominator into `0.0`, starting the recursion at
  bar 0 with no history; the ADX zero-denominator guard added in 0.2.0 used
  `fill_null(0.0)`, which also filled warm-up nulls, so `adx`/`adx_pos`/
  `adx_neg` reported from bar 0. All four now surface the first `window - 1`
  rows as nulls, like every other indicator. Post-warm-up values are unchanged
  for KAMA/`adx_pos`/`adx_neg`; `adx` values shift slightly because the Wilder
  recursion is now seeded from the first post-warm-up DX value instead of
  bar 1.
- **`microstructure.vpin` crashed with `IndexError` when total volume exceeded
  `bucket_size × n_bars`.** The per-bucket accumulators were sized by bar
  count, but a single bar can fill several buckets. Buffers are now sized by
  `total volume / bucket_size`.

- **`momentum.kama` returned float `NaN` instead of null for its warm-up
  rows**, violating the library-wide "not enough data yet is null" convention.
- **`quant.log_return`, `microstructure.rolling_beta`, and
  `microstructure.rolling_cov` rejected column-name strings** (they only
  accepted `pl.Expr`, unlike every other indicator). They now coerce strings
  like the rest of the API.

### Added

- **100% line coverage**, enforced in CI (`--cov-fail-under=100`): a smoke
  suite (`tests/test_smoke.py`) exercises every public indicator in both
  fillna modes, plus edge-case tests for class-only combination indicators,
  the Keltner EMA variant, Ichimoku visual mode, smoothed VPT, and degenerate
  inputs (flat/short series, never-filling VPIN buckets).
- **Multi-asset guarantee:** every indicator (including the sequential
  `map_batches`-based KAMA, PSAR, VPIN, and Hurst) works per-symbol via
  `.over("symbol")` with no state leaking across symbols, enforced by a new
  `tests/test_multi_asset.py` and documented in the how-to guides.
- **Warm-up policy tests** (`tests/test_warmup.py`) pinning each indicator's
  exact first-valid row: warm-up rows are null, never fabricated numbers, and
  no nulls appear after warm-up on clean data.
- Reference cross-checks for EMA, MACD, OBV, MFI, Stochastic %K, Williams %R,
  ROC, and CCI against independent NumPy implementations.
- The package now ships a `py.typed` marker, so type checkers see the
  library's annotations; PyPI classifiers declare Python 3.10–3.13 support.
- CI reports line coverage (`pytest-cov`).

## [0.2.0] - 2026-07-19

### Fixed

- **`trend.adx` returned all-NaN on real data.** `adx`, `adx_pos`, and `adx_neg`
  divided by the true range and by `(+DI + -DI)` without guarding a zero
  denominator, so a flat or low-range bar produced `inf`/`NaN`. Because those
  values feed a Wilder `ewm_mean` (infinite memory), a single `NaN` poisoned the
  entire column — on the real BTCUSDT fixture `adx()` was NaN for all 5000 rows.
  The zero denominators are now nulled and the result filled with `0` (no
  directional movement). `adx_pos`/`adx_neg` output is unchanged on data without
  flat bars; `adx` now returns valid values in `[0, 100]`. Regression tests added
  (`test_adx_matches_reference`, `test_adx_flat_market_is_finite`).
- **Log-return indicators no longer leak `-inf`/`NaN` on non-positive prices.**
  A price `<= 0` now yields a null return instead of poisoning downstream rolling
  windows (`hasbrouck_lambda`, `hurst_exponent`, `variance_ratio`,
  `quant.historical_volatility`).

### Added

- New volatility estimators: `quant.parkinson_volatility` and
  `quant.rogers_satchell_volatility`, completing the OHLC realized-volatility
  family alongside Garman-Klass and Yang-Zhang.
- New microstructure indicators: `microstructure.corwin_schultz_spread`
  (high-low bid-ask spread estimator) and `microstructure.half_life`
  (Ornstein-Uhlenbeck half-life of mean reversion).
- Optional `speed` extra (`pip install polars-ta-lib[speed]`) that JIT-compiles
  the sequential VPIN volume-bucketing loop with Numba; falls back to an
  identical pure-Python loop when absent (output is byte-for-byte the same).
- Documentation figures rendered on the real BTCUSDT fixture — classic
  indicators, trend & volume, and liquidity & microstructure — plus a
  one-command regeneration script (`examples/generate_all_figures.py`).
- Hypothesis property tests (`tests/test_properties.py`) covering length
  preservation, no NaN/inf leakage, and causality (no lookahead).

### Changed

- `microstructure.hurst_exponent` rewritten from a per-row `rolling_map` to a
  single vectorized `map_batches` pass over a sliding-window view (roughly
  minutes → seconds on large inputs); output unchanged.
- Shared rolling-covariance / OLS-slope / log-return logic factored into an
  internal `polars_ta._internal` module and reused across the microstructure and
  quant indicators; output verified byte-identical.

## [0.1.0]

- Initial release: Polars-native technical-analysis indicators (momentum, trend,
  volatility, volume), a `quant` module, a `microstructure` module (VPIN, Kyle's
  and Hasbrouck's lambda, Roll's spread, multi-scale Hurst ribbon), packaging
  with uv, tests, benchmarks, and an MkDocs documentation site.
