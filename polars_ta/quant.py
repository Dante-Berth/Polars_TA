import math

import polars as pl

from polars_ta._internal import log_return


def garman_klass_volatility(
    open_price: str, high: str, low: str, close: str, window: int = 20
) -> pl.Expr:
    o, h, lo, c = pl.col(open_price), pl.col(high), pl.col(low), pl.col(close)
    log_hl = (h / lo).log().pow(2)
    log_co = (c / o).log().pow(2)
    gk_variance = (0.5 * log_hl) - ((2 * math.log(2) - 1) * log_co)
    return gk_variance.rolling_mean(window_size=window).sqrt()


def rolling_z_score(close: str, window: int = 20) -> pl.Expr:
    c = pl.col(close)
    rolling_mean = c.rolling_mean(window_size=window)
    rolling_std = c.rolling_std(window_size=window)
    safe_std = pl.when(rolling_std == 0).then(1.0).otherwise(rolling_std)
    return (c - rolling_mean) / safe_std


def vol_adjusted_momentum(close: str, window: int = 20) -> pl.Expr:
    c = pl.col(close)
    ret = (c / c.shift(window)) - 1.0
    daily_ret = c.pct_change()
    vol = daily_ret.rolling_std(window_size=window)
    safe_vol = pl.when(vol == 0).then(1.0).otherwise(vol)
    return ret / safe_vol


def micro_price_proxy(high: str, low: str, close: str, volume: str) -> pl.Expr:
    h, lo, c, v = pl.col(high), pl.col(low), pl.col(close), pl.col(volume)
    typical_price = (h + lo + c) / 3.0
    log_v = pl.when(v <= 1).then(1.0).otherwise(v.log())
    return typical_price * log_v


def rolling_sharpe_ratio(
    close: str, window: int = 63, risk_free_rate: float = 0.0
) -> pl.Expr:
    """Annualized Rolling Sharpe Ratio (Assuming 252 trading days)"""
    ret = pl.col(close).pct_change()

    # Assuming daily data, we annualize by multiplying by sqrt(252)
    rolling_mean = ret.rolling_mean(window_size=window) - (risk_free_rate / 252)
    rolling_std = ret.rolling_std(window_size=window)

    safe_std = pl.when(rolling_std == 0).then(1.0).otherwise(rolling_std)
    sharpe = (rolling_mean / safe_std) * math.sqrt(252)

    return sharpe.alias(f"sharpe_{window}")


def rolling_sortino_ratio(
    close: str, window: int = 63, risk_free_rate: float = 0.0
) -> pl.Expr:
    """Annualized Rolling Sortino Ratio (Penalizes only downside volatility)"""
    ret = pl.col(close).pct_change()

    downside_ret = pl.when(ret < 0).then(ret).otherwise(0.0)

    rolling_mean = ret.rolling_mean(window_size=window) - (risk_free_rate / 252)
    downside_std = downside_ret.rolling_std(window_size=window)

    safe_down_std = pl.when(downside_std == 0).then(1.0).otherwise(downside_std)
    sortino = (rolling_mean / safe_down_std) * math.sqrt(252)

    return sortino.alias(f"sortino_{window}")


def historical_volatility(close: str, window: int = 21) -> pl.Expr:
    """Annualized Historical Volatility (Close-to-Close)"""
    log_ret = log_return(pl.col(close))

    # Standard deviation of log returns * sqrt(252 trading days)
    hv = log_ret.rolling_std(window_size=window) * math.sqrt(252)
    return hv.alias(f"hist_vol_{window}")


def ewma_volatility(
    close: str, window: int = 21, lambda_: float = 0.94
) -> pl.Expr:
    """Annualized EWMA (RiskMetrics-style) volatility of log returns.

    Unlike `historical_volatility`'s flat rolling window, older squared
    returns decay geometrically (weight `lambda_ ** k`), so a volatility
    spike shows up immediately and fades out smoothly instead of dropping off
    a cliff `window` bars later. `lambda_=0.94` is the RiskMetrics daily
    default; `min_samples=window` keeps the same warm-up convention as every
    other volatility estimator here even though the EWM itself has infinite
    memory.
    """
    log_ret = log_return(pl.col(close))
    sq_ret = log_ret.pow(2)
    variance = sq_ret.ewm_mean(alpha=1 - lambda_, adjust=False, min_samples=window)
    ewma_vol = variance.sqrt() * math.sqrt(252)
    return ewma_vol.alias(f"ewma_vol_{window}")


def parkinson_volatility(
    high: str, low: str, window: int = 20, trading_periods: int = 252
) -> pl.Expr:
    """Parkinson (1980) high-low range volatility estimator, annualized.

    Uses only the intraday high-low range, which makes it far more efficient
    than close-to-close historical volatility (it exploits the whole bar, not
    just the endpoints). It assumes no drift and no overnight jumps, so it
    complements Garman-Klass and Yang-Zhang rather than replacing them.
    """
    h, lo = pl.col(high), pl.col(low)
    factor = 1.0 / (4.0 * math.log(2.0))
    park_var = (factor * (h / lo).log().pow(2)).rolling_mean(window_size=window)
    return (park_var.sqrt() * math.sqrt(trading_periods)).alias(
        f"parkinson_vol_{window}"
    )


def rogers_satchell_volatility(
    open_price: str,
    high: str,
    low: str,
    close: str,
    window: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Rogers-Satchell (1991) volatility estimator, annualized.

    Unlike Parkinson and Garman-Klass, this estimator is unbiased in the
    presence of a non-zero drift, using all four OHLC prices. It does not
    account for overnight gaps (that is what Yang-Zhang adds on top), so it is
    the natural mid-point of the OHLC-volatility family.
    """
    o, h, lo, c = (
        pl.col(open_price),
        pl.col(high),
        pl.col(low),
        pl.col(close),
    )
    log_ho = (h / o).log()
    log_hc = (h / c).log()
    log_lo = (lo / o).log()
    log_lc = (lo / c).log()
    rs = log_ho * log_hc + log_lo * log_lc
    rs_var = rs.rolling_mean(window_size=window)
    return (rs_var.sqrt() * math.sqrt(trading_periods)).alias(f"rs_vol_{window}")


def yang_zhang_volatility(
    open_price: str,
    high: str,
    low: str,
    close: str,
    window: int = 20,
    trading_periods: int = 252,
) -> pl.Expr:
    """Yang-Zhang (2000) volatility estimator, annualized.

    Combines overnight (open-to-prev-close), open-to-close drift, and a
    Rogers-Satchell high/low term into a single minimum-variance estimator
    that (unlike close-to-close historical volatility or Garman-Klass) is
    unbiased in the presence of both overnight jumps and intraday drift.
    This is the volatility estimator of choice on professional vol desks
    when only OHLC bars (no tick data) are available.
    """
    o = pl.col(open_price)
    h = pl.col(high)
    lo = pl.col(low)
    c = pl.col(close)

    log_ho = (h / o.shift(1)).log()
    log_lo_ = (lo / o).log()
    log_co = (c / o).log()
    log_oc = (o / c.shift(1)).log()
    log_cc = (c / c.shift(1)).log()

    rs_term = log_ho * (log_ho - log_co) + log_lo_ * (log_lo_ - log_co)

    open_vol = (log_oc**2).rolling_sum(window_size=window) / (window - 1)
    close_vol = (log_cc**2).rolling_sum(window_size=window) / (window - 1)
    window_rs = rs_term.rolling_sum(window_size=window) / (window - 1)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_variance = open_vol + k * close_vol + (1 - k) * window_rs

    return (yz_variance.sqrt() * math.sqrt(trading_periods)).alias(
        f"yz_volatility_{window}"
    )


def hurst_ribbon(
    close: str, scales: tuple[int, ...] = (16, 32, 64)
) -> dict[str, pl.Expr]:
    """Multi-scale Hurst ribbon: rescaled-range Hurst exponent computed at
    several window scales simultaneously, plus two derived regime features.

    Returns a dict of expressions rather than a single one — pass its values
    straight into ``with_columns(**hurst_ribbon("close").values())`` or
    unpack individual keys. Keys are ``h_{scale}`` for each scale, plus:

    - ``h_ribbon_avg``: mean Hurst across scales — overall trending
      (>0.5) vs mean-reverting (<0.5) regime.
    - ``h_ribbon_tilt``: shortest-scale H minus longest-scale H — positive
      means short-term trend is stronger than long-term (breakout
      potential), negative means short-term is exhausting relative to the
      longer trend.

    Unlike :func:`hurst_exponent`'s full R/S analysis (accurate but
    O(window) per row), this uses the cheap log(range/std)/log(window)
    approximation, which is fast enough to run at several scales at once
    and is the form used in production multi-scale regime detectors.
    """
    ln_p = pl.col(close).log()
    log_ret = ln_p.diff(1)

    h_exprs: dict[str, pl.Expr] = {}
    for w in scales:
        rng = ln_p.rolling_max(window_size=w) - ln_p.rolling_min(window_size=w)
        std = log_ret.rolling_std(window_size=w)
        safe_std = pl.when(std == 0).then(None).otherwise(std)
        h_exprs[f"h_{w}"] = ((rng / safe_std).log() / math.log(w)).alias(f"h_{w}")

    avg_expr = sum(h_exprs[f"h_{w}"] for w in scales) / len(scales)
    tilt_expr = h_exprs[f"h_{scales[0]}"] - h_exprs[f"h_{scales[-1]}"]

    h_exprs["h_ribbon_avg"] = avg_expr.alias("h_ribbon_avg")
    h_exprs["h_ribbon_tilt"] = tilt_expr.alias("h_ribbon_tilt")
    return h_exprs


def relative_volume(volume: str, window: int = 100) -> pl.Expr:
    """Relative volume (RVol): current volume vs its rolling mean.

    Spikes in RVol often mark the start or end of a regime — a standard
    "is something happening right now" gate on execution/monitoring desks.
    """
    v = pl.col(volume)
    return (v / v.rolling_mean(window_size=window)).alias(f"rvol_{window}")


def volatility_z_score(high: str, low: str, window: int = 100) -> pl.Expr:
    """Z-score of the rolling high-low range against its own recent history.

    Flags volatility expansion/contraction relative to the recent norm,
    independent of the absolute price level — used to gate position sizing
    or strategy switching on a volatility-regime shift.
    """
    hl_range = (pl.col(high) - pl.col(low)).rolling_mean(window_size=window)
    mean = hl_range.rolling_mean(window_size=window)
    std = hl_range.rolling_std(window_size=window)
    safe_std = pl.when(std == 0).then(None).otherwise(std)
    return ((hl_range - mean) / safe_std).alias(f"vol_z_score_{window}")


def cross_sectional_zscore(value: str) -> pl.Expr:
    """Cross-sectional z-score of `value` at each timestamp.

    Unlike every other indicator in this library, which computes a rolling
    statistic *through time* for one symbol, this compares symbols *against
    each other at the same instant* — the core building block of a
    factor/ranking strategy. It is a per-symbol expression like any other,
    but is only meaningful applied with `.over(timestamp_column)` on a
    long-format multi-asset frame (columns: timestamp, symbol, value, ...),
    grouping across symbols rather than across time:

        df.with_columns(
            quant.cross_sectional_zscore("momentum").over("timestamp")
        )

    A cross-section with zero spread (all symbols tied) yields null rather
    than a divide-by-zero.
    """
    v = pl.col(value)
    mean = v.mean()
    std = v.std()
    safe_std = pl.when(std == 0).then(None).otherwise(std)
    return (v - mean) / safe_std


def cross_sectional_rank(value: str, pct: bool = True) -> pl.Expr:
    """Cross-sectional rank of `value` at each timestamp, in `[0, 1]` by
    default (`pct=True`) or as a dense integer rank (`pct=False`).

    Same usage as :func:`cross_sectional_zscore`: apply with
    `.over(timestamp_column)` on a long-format multi-asset frame to rank
    symbols against each other at each instant, not through time.
    """
    v = pl.col(value)
    if pct:
        return v.rank(method="average") / v.count()
    return v.rank(method="dense")


def amihud_illiquidity(close: str, volume: str, window: int = 21) -> pl.Expr:
    """Rolling Amihud Illiquidity (Price impact per dollar traded)"""
    c = pl.col(close)
    v = pl.col(volume)

    daily_ret_abs = c.pct_change().abs()
    dollar_volume = c * v

    safe_dollar_vol = pl.when(dollar_volume == 0).then(None).otherwise(dollar_volume)

    # Ratio of absolute return to dollar volume, smoothed over a window
    amihud = (daily_ret_abs / safe_dollar_vol).rolling_mean(window_size=window)

    # Multiply by 10^6 to make the numbers human-readable (standard practice)
    return (amihud * 1_000_000).alias(f"amihud_illiq_{window}")


def regime_conditional_signal(
    regime: str | pl.Expr,
    threshold: float,
    signal_above: str | pl.Expr,
    signal_below: str | pl.Expr,
    above_or_equal: bool = True,
) -> pl.Expr:
    """Switch between two pre-computed signal expressions based on a
    regime score, row by row: `signal_above` where `regime >= threshold`
    (or `>` if `above_or_equal=False`), `signal_below` otherwise.

    This is a hard switch, not a smooth blend — the output jumps discretely
    at the threshold rather than fading between the two signals, which
    keeps the composite as interpretable as its inputs (no intermediate
    "40% trend-following" values to explain) at the cost of a
    discontinuity exactly at the boundary. This is a compositional building
    block, not a Hurst-specific helper: `regime` can be any expression —
    `quant.hurst_ribbon(...)["h_ribbon_avg"]`, an ADX reading, a Shannon
    entropy score — and `signal_above`/`signal_below` can be any two
    already-computed indicator expressions you want the regime to arbitrate
    between (they are typically named differently, e.g. a fast EMA-cross
    trend signal vs. a Bollinger %B mean-reversion signal — see the
    "Regime-conditional trend/mean-reversion switch" how-to guide for a
    complete example built on `hurst_ribbon`).

    A null `regime` value produces a null output (arbitration is undefined
    without a regime reading), rather than silently falling back to either
    branch.
    """
    regime_expr = pl.col(regime) if isinstance(regime, str) else regime
    above = pl.col(signal_above) if isinstance(signal_above, str) else signal_above
    below = pl.col(signal_below) if isinstance(signal_below, str) else signal_below

    condition = regime_expr >= threshold if above_or_equal else regime_expr > threshold
    return (
        pl.when(regime_expr.is_null())
        .then(None)
        .when(condition)
        .then(above)
        .otherwise(below)
        .alias("regime_conditional_signal")
    )
