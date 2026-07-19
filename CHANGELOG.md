# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
