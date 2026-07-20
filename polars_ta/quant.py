import math

import numpy as np
import polars as pl

from polars_ta._internal import (
    as_expr,
    log_return,
    rolling_beta,
    rolling_cov,
    rolling_var,
)


def _norm_ppf(alpha: float) -> float:
    """Standard-normal inverse CDF (quantile) via the inverse error function.

    ``z_alpha = sqrt(2) * erfinv(2*alpha - 1)``. Used for the Gaussian VaR
    quantile so no SciPy dependency is needed for a single scalar lookup.
    """
    from math import erf  # noqa: F401  (kept for symmetry / clarity)

    # Newton refinement on erf is overkill for one scalar; use a rational
    # approximation (Acklam) accurate to ~1e-9 over the usable tail range.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    cc = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if alpha < p_low:
        q = math.sqrt(-2.0 * math.log(alpha))
        return (
            ((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if alpha <= p_high:
        q = alpha - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - alpha))
    return -(
        ((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


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


def ewma_volatility(close: str, window: int = 21, lambda_: float = 0.94) -> pl.Expr:
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


# ---------------------------------------------------------------------------
# Tail-risk & drawdown
#
# Position sizing on a systematic book is driven by the *left tail* and the
# *path*, not by symmetric standard deviation. The Sharpe/Sortino ratios above
# summarize central tendency; these summarize how bad the bad case is and how
# deep the equity curve digs. All are rolling, causal (null warm-up), and
# expressed as pure Polars expressions.
# ---------------------------------------------------------------------------


def rolling_cvar(
    close: str | pl.Expr, window: int = 100, alpha: float = 0.05
) -> pl.Expr:
    """Rolling Conditional Value-at-Risk (Expected Shortfall) of simple returns.

    CVaR at level ``alpha`` is the mean of the worst ``alpha`` fraction of
    returns in the window — the expected loss *given* that the VaR threshold is
    breached. Unlike VaR (a quantile), CVaR is a coherent risk measure
    (sub-additive), which is why it is the tail metric of choice for position
    sizing and the objective in Rockafellar-Uryasev portfolio optimization.

    Reported as a **positive** loss magnitude (the sign is flipped), so larger
    means more tail risk. A window with fewer than ``ceil(1/alpha)`` returns
    can't resolve the ``alpha`` tail and reports null rather than a single-point
    "expected shortfall".
    """
    close = as_expr(close)
    ret = close.pct_change()
    # Number of tail observations averaged; at least 1, and the window must hold
    # enough returns to actually populate that tail.
    k = max(1, math.ceil(alpha * window))

    def _cvar(s: pl.Series) -> pl.Series:
        x = s.to_numpy()
        n = len(x)
        out = [None] * n
        min_needed = max(window, k + 1)
        for i in range(min_needed - 1, n):
            w = x[i - window + 1 : i + 1]
            w = w[~np.isnan(w)]
            if len(w) < k:
                continue
            worst = np.sort(w)[:k]
            out[i] = float(-worst.mean())
        return pl.Series(out, dtype=pl.Float64)

    return ret.map_batches(_cvar, returns_scalar=False).alias(
        f"cvar_{int(alpha * 100)}_{window}"
    )


def cornish_fisher_var(
    close: str | pl.Expr, window: int = 100, alpha: float = 0.05
) -> pl.Expr:
    """Rolling Cornish-Fisher (modified) Value-at-Risk of simple returns.

    Standard Gaussian VaR uses ``mu + z_alpha * sigma``, which understates tail
    risk for the negatively-skewed, fat-tailed return distributions typical of
    risk assets. The Cornish-Fisher expansion corrects the Gaussian quantile
    ``z`` for the sample skewness ``S`` and excess kurtosis ``K``::

        z_cf = z + (z^2-1)/6 * S + (z^3-3z)/24 * K - (2z^3-5z)/36 * S^2

    and VaR = ``-(mu + z_cf * sigma)``, reported as a positive loss magnitude.
    This is the RiskMetrics "modified VaR" and captures crash risk that
    symmetric vol misses, without assuming a parametric fat-tailed family.
    """
    close = as_expr(close)
    ret = close.pct_change()

    z = _norm_ppf(alpha)
    mu = ret.rolling_mean(window_size=window)
    sigma = ret.rolling_std(window_size=window)
    s = ret.rolling_skew(window_size=window)
    # Polars' rolling_kurtosis returns excess kurtosis (normal -> 0).
    k = ret.rolling_kurtosis(window_size=window)

    z_cf = (
        z
        + (z**2 - 1) / 6.0 * s
        + (z**3 - 3 * z) / 24.0 * k
        - (2 * z**3 - 5 * z) / 36.0 * s.pow(2)
    )
    # A zero-dispersion window is a degenerate point mass: skewness/kurtosis are
    # undefined (NaN), and VaR is just -mu (no tail beyond the point). Guard so
    # that constant-return windows yield -mu rather than leaking NaN.
    var = pl.when(sigma == 0).then(-mu).otherwise(-(mu + z_cf * sigma))
    return var.alias(f"cf_var_{int(alpha * 100)}_{window}")


def rolling_max_drawdown(close: str | pl.Expr, window: int = 100) -> pl.Expr:
    """Rolling maximum drawdown over a trailing ``window`` of prices.

    Drawdown at each bar is ``price / running_peak - 1``; the maximum drawdown
    over the window is the most negative such value, reported as a **positive**
    fraction (e.g. ``0.18`` for an 18% peak-to-trough decline). Uses the
    trailing rolling peak (``rolling_max``), so it is fully causal — no
    look-ahead to a future high — and answers "how bad has the worst dip been
    over the last ``window`` bars".
    """
    close = as_expr(close)
    peak = close.rolling_max(window_size=window)
    safe_peak = pl.when(peak == 0).then(None).otherwise(peak)
    drawdown = close / safe_peak - 1.0
    return (-drawdown.rolling_min(window_size=window)).alias(f"max_drawdown_{window}")


def calmar_ratio(close: str | pl.Expr, window: int = 252) -> pl.Expr:
    """Rolling Calmar ratio: annualized return divided by rolling max drawdown.

    Calmar rewards return per unit of *worst-case path pain* rather than per
    unit of volatility (Sharpe) — the metric managed-futures/CTA desks are
    judged on, since it punishes a strategy that makes steady money then gives
    it all back in one drawdown. Numerator is the annualized simple return over
    the window (``(P_t / P_{t-window})^{252/window} - 1``); denominator is
    :func:`rolling_max_drawdown`. A window with zero drawdown yields null (the
    ratio is undefined, not infinite).
    """
    close = as_expr(close)
    total_ret = close / close.shift(window)
    ann_ret = total_ret.pow(252.0 / window) - 1.0

    peak = close.rolling_max(window_size=window)
    safe_peak = pl.when(peak == 0).then(None).otherwise(peak)
    mdd = -(close / safe_peak - 1.0).rolling_min(window_size=window)
    safe_mdd = pl.when(mdd <= 0).then(None).otherwise(mdd)
    return (ann_ret / safe_mdd).alias(f"calmar_{window}")


# ---------------------------------------------------------------------------
# Distribution shape (skewness / kurtosis / path quality)
#
# Higher moments of the return distribution are leading indicators of regime
# fragility: skewness flips negative and kurtosis spikes *before* volatility
# does when a market becomes crash-prone. These complement the entropy /
# variance-ratio regime tools by describing distributional *shape* rather than
# serial dependence.
# ---------------------------------------------------------------------------


def rolling_skew(close: str | pl.Expr, window: int = 60) -> pl.Expr:
    """Rolling skewness of simple returns over ``window`` bars.

    Persistent negative skew ("picks up pennies in front of a steamroller") is
    the classic signature of a strategy or regime carrying hidden crash risk;
    rising positive skew often accompanies momentum/blow-off phases. A
    zero-dispersion window (constant returns) has undefined skew and yields
    null rather than leaking NaN.
    """
    ret = as_expr(close).pct_change()
    return ret.rolling_skew(window_size=window).fill_nan(None).alias(f"skew_{window}")


def rolling_kurtosis(close: str | pl.Expr, window: int = 60) -> pl.Expr:
    """Rolling excess kurtosis of simple returns over ``window`` bars.

    Excess kurtosis (normal distribution -> 0) rising well above 0 flags
    fat tails / clustered extreme moves — a warning that Gaussian VaR and any
    downstream mean-variance sizing are understating tail risk. Pairs naturally
    with :func:`cornish_fisher_var`, which uses exactly this moment. A
    zero-dispersion window (constant returns) has undefined kurtosis and yields
    null rather than leaking NaN.
    """
    ret = as_expr(close).pct_change()
    return (
        ret.rolling_kurtosis(window_size=window)
        .fill_nan(None)
        .alias(f"kurtosis_{window}")
    )


def gain_to_pain(close: str | pl.Expr, window: int = 60) -> pl.Expr:
    """Rolling gain-to-pain ratio: sum of returns divided by sum of losses.

    Defined (Schwager) as ``sum(returns) / sum(|negative returns|)`` over the
    window — total net move per unit of downside suffered. It is more robust
    than Sharpe to a handful of outlier winners (it doesn't reward upside
    volatility at all) and is a standard discretionary-desk screen for
    "smoothness" of a return stream. A window with no losing bars yields null
    (infinite gain-to-pain is not a meaningful finite feature).
    """
    ret = as_expr(close).pct_change()
    total = ret.rolling_sum(window_size=window)
    pain = pl.when(ret < 0).then(-ret).otherwise(0.0).rolling_sum(window_size=window)
    safe_pain = pl.when(pain == 0).then(None).otherwise(pain)
    return (total / safe_pain).alias(f"gain_to_pain_{window}")


def jarque_bera(close: str | pl.Expr, window: int = 60) -> pl.Expr:
    """Rolling Jarque-Bera normality test statistic of simple returns.

    ``JB = (n / 6) * (S**2 + K**2 / 4)`` where ``S`` is the skewness and ``K``
    the excess kurtosis of the returns in the window. Under the null of
    normally distributed returns JB is asymptotically chi-squared with 2
    degrees of freedom, so a value above ~6 rejects normality at the 5% level
    (~9.2 at 1%). It is a single scalar that rises when *either* tail asymmetry
    or fat-tailedness appears — a compact "how non-Gaussian has this regime
    been" gate that subsumes :func:`rolling_skew` and :func:`rolling_kurtosis`
    into one number, which matters because any downstream Gaussian VaR /
    mean-variance sizing is only valid while JB stays small.

    Reuses the same biased (population) moments Polars' ``rolling_skew`` /
    ``rolling_kurtosis`` report, so it is consistent with those two features
    rather than a separately-normalized estimate.
    """
    ret = as_expr(close).pct_change()
    s = ret.rolling_skew(window_size=window)
    k = ret.rolling_kurtosis(window_size=window)  # excess kurtosis
    jb = (window / 6.0) * (s.pow(2) + k.pow(2) / 4.0)
    # A zero-dispersion window has undefined moments; null rather than NaN.
    return jb.fill_nan(None).alias(f"jarque_bera_{window}")


# ---------------------------------------------------------------------------
# Signal conditioning: stationarity & decay
#
# A raw price series is I(1) (non-stationary) but carries all the memory; its
# return series is stationary but memoryless. Fractional differentiation
# (Lopez de Prado, *Advances in Financial ML*, ch. 5) sits between the two,
# and the autocorrelation / information-coefficient tools measure how much
# tradeable structure a feature actually has.
# ---------------------------------------------------------------------------


def _frac_diff_weights(d: float, width: int) -> "list[float]":
    """Binomial fractional-difference weights w_0..w_{width-1} (w_0 = 1)."""
    w = [1.0]
    for k in range(1, width):
        w.append(-w[-1] * (d - k + 1) / k)
    return w


def frac_diff(close: str | pl.Expr, d: float = 0.4, window: int = 100) -> pl.Expr:
    """Fixed-width-window fractional differentiation of the log price.

    Applies the binomial fractional-difference operator ``(1-L)^d`` truncated
    to a fixed window (Lopez de Prado, *Advances in Financial ML*, ch. 5).
    For ``d in (0, 1)`` the output is (approximately) stationary while
    retaining far more of the original series' long memory than a full first
    difference (``d = 1``, ordinary log returns) would — often the difference
    between a feature that passes an ADF stationarity test *and* still predicts,
    versus a return series that is stationary but has thrown its memory away.

    Weights ``w_k = -w_{k-1} (d-k+1)/k`` are applied to the trailing ``window``
    log-prices. Larger ``d`` -> more differencing / less memory; ``d`` near 0
    keeps almost all memory but may not fully stationarize.
    """
    ln_p = as_expr(close).log()
    weights = _frac_diff_weights(d, window)
    # Dot the fixed weight vector against the trailing window: w_0*p_t +
    # w_1*p_{t-1} + ... Each shift is a lag, so this stays a pure vectorized
    # expression (no Python per-row loop) — the loop is only over `window`
    # constant coefficients, built once.
    terms = [weights[k] * ln_p.shift(k) for k in range(window)]
    return sum(terms[1:], terms[0]).alias(f"frac_diff_{d}_{window}")


def rolling_autocorr(close: str | pl.Expr, lag: int = 1, window: int = 60) -> pl.Expr:
    """Rolling lag-``lag`` autocorrelation of simple returns.

    ``corr(r_t, r_{t-lag})`` over a trailing window. Positive lag-1
    autocorrelation is momentum/trending structure; negative is mean reversion
    (bid-ask bounce at short horizons, or genuine reversal at longer ones). A
    direct, sign-explicit complement to the variance ratio and Hurst tools. A
    flat window (zero variance) yields null.
    """
    ret = as_expr(close).pct_change()
    lagged = ret.shift(lag)
    # corr = cov(r, r_lag) / (std(r) * std(r_lag)); use the shared primitives.
    num = rolling_cov(ret, lagged, window)
    var_a = rolling_var(ret, window)
    var_b = rolling_var(lagged, window)
    denom = (var_a * var_b).sqrt()
    safe_denom = pl.when(denom == 0).then(None).otherwise(denom)
    return (num / safe_denom).alias(f"autocorr_{lag}_{window}")


def rolling_ic(
    signal: str | pl.Expr, forward_return: str | pl.Expr, window: int = 60
) -> pl.Expr:
    """Rolling information coefficient: correlation between a signal and the
    realized forward return.

    The IC is the single most important diagnostic for a predictive feature:
    the rolling Pearson correlation between ``signal_t`` and the
    contemporaneous ``forward_return_t`` column (which you construct as a
    *forward* return, e.g. ``close.pct_change().shift(-h)``, so that at
    evaluation time the pair ``(signal_t, fwd_ret_t)`` is aligned). A decaying
    rolling IC is the earliest sign of alpha decay — the feature has stopped
    working — long before the equity curve rolls over.

    **Look-ahead warning:** ``forward_return`` is forward-looking *by
    construction*; that is correct for measuring predictive power but means the
    IC series itself must never be used as a live trading input — it is a
    research/monitoring diagnostic. A flat window yields null.
    """

    sig = as_expr(signal)
    fwd = as_expr(forward_return)
    num = rolling_cov(sig, fwd, window)
    denom = (rolling_var(sig, window) * rolling_var(fwd, window)).sqrt()
    safe_denom = pl.when(denom == 0).then(None).otherwise(denom)
    return (num / safe_denom).alias(f"ic_{window}")


# ---------------------------------------------------------------------------
# Cross-sectional / factor plumbing
#
# Beyond the point-in-time cross_sectional_rank/zscore above, a factor book
# needs each asset's relationship *to a benchmark through time*: its beta, the
# volatility that beta doesn't explain (idiosyncratic vol), and asymmetric
# (downside) beta. Plus the canonical price-momentum factor. All are per-symbol
# rolling expressions; combine with `.over("symbol")` on a long frame.
# ---------------------------------------------------------------------------


def rolling_beta_to(
    close: str | pl.Expr, benchmark: str | pl.Expr, window: int = 60
) -> pl.Expr:
    """Rolling market beta of an asset's returns against a benchmark's returns.

    OLS slope of the asset's simple returns on the benchmark's simple returns
    over a trailing window — the sensitivity a factor-neutral book needs to
    hedge out. ``benchmark`` is a returns-bearing price column (e.g. BTC or an
    index level) carried alongside the asset on the same frame. A flat
    benchmark window yields null.
    """
    r_asset = as_expr(close).pct_change()
    r_bench = as_expr(benchmark).pct_change()
    return rolling_beta(r_asset, r_bench, window).alias(f"beta_{window}")


def idiosyncratic_vol(
    close: str | pl.Expr, benchmark: str | pl.Expr, window: int = 60
) -> pl.Expr:
    """Rolling idiosyncratic (residual) volatility relative to a benchmark.

    The part of an asset's return variance the benchmark does *not* explain:
    ``sqrt(var(r_asset) * (1 - rho^2))`` over the window, where ``rho`` is the
    rolling correlation to the benchmark. This is the risk that survives a
    beta hedge — the tradeable, asset-specific component a stat-arb/relative-
    value book actually harvests, and a cleaner "specialness" measure than raw
    volatility. Annualized by ``sqrt(252)``.
    """

    r_asset = as_expr(close).pct_change()
    r_bench = as_expr(benchmark).pct_change()
    var_a = rolling_var(r_asset, window)
    var_b = rolling_var(r_bench, window)
    cov = rolling_cov(r_asset, r_bench, window)
    denom = var_a * var_b
    rho2 = pl.when(denom == 0).then(0.0).otherwise(cov.pow(2) / denom)
    resid_var = var_a * (1.0 - rho2)
    # Clamp tiny negatives from floating error before the sqrt.
    resid_var = pl.when(resid_var < 0).then(0.0).otherwise(resid_var)
    return (resid_var.sqrt() * math.sqrt(252)).alias(f"idio_vol_{window}")


def downside_beta(
    close: str | pl.Expr, benchmark: str | pl.Expr, window: int = 60
) -> pl.Expr:
    """Rolling downside beta: beta estimated only on bars where the benchmark
    fell.

    Bawa-Lindenberg / Ang-Chen downside beta captures how an asset behaves when
    the market is *down* — the regime that actually matters for tail hedging and
    that symmetric beta averages away. Computed as the OLS slope of asset
    returns on benchmark returns restricted to benchmark-negative bars within
    each trailing window. A window with fewer than two down-benchmark bars (or a
    flat down-benchmark subset) yields null.

    Unlike the symmetric :func:`rolling_beta_to`, the down-bar subset can't use
    the shared rolling-covariance primitive (its windows would be riddled with
    the masked-out up-bars, and Polars' default rolling requires a full window
    of non-nulls), so the per-window OLS on the down subset runs in a single
    ``map_batches`` pass over a stacked ``[asset, bench]`` struct.
    """
    r_asset = as_expr(close).pct_change()
    r_bench = as_expr(benchmark).pct_change()

    def _downside_beta(struct_s: pl.Series) -> pl.Series:
        sub = struct_s.struct.unnest()
        a = sub["a"].to_numpy().astype(float)
        b = sub["b"].to_numpy().astype(float)
        n = len(a)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            aw = a[i - window + 1 : i + 1]
            bw = b[i - window + 1 : i + 1]
            mask = (~np.isnan(aw)) & (~np.isnan(bw)) & (bw < 0)
            if mask.sum() < 2:
                continue
            bd = bw[mask]
            ad = aw[mask]
            # A down subset with a single distinct benchmark return makes the
            # slope undefined; skip it rather than divide by (near-)zero
            # variance, mirroring how the shared rolling_beta helper guards its
            # windows. Test min == max rather than var() == 0.0, since the
            # variance of identical floats is a tiny non-zero rounding residual.
            if bd.min() == bd.max():
                continue
            out[i] = float(np.cov(ad, bd, bias=True)[0, 1] / bd.var())
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return (
        pl.struct([r_asset.alias("a"), r_bench.alias("b")])
        .map_batches(_downside_beta, returns_scalar=False)
        .alias(f"downside_beta_{window}")
    )


def momentum_12_1(close: str | pl.Expr, lookback: int = 252, skip: int = 21) -> pl.Expr:
    """Cross-sectional price momentum, skipping the most recent ``skip`` bars.

    The canonical Jegadeesh-Titman / Fama-French "momentum 12-1" factor: the
    return from ``lookback`` bars ago up to ``skip`` bars ago
    (``P_{t-skip} / P_{t-lookback} - 1``), deliberately *excluding* the most
    recent month. The skip removes the well-documented short-term reversal
    (bid-ask bounce and 1-month mean reversion) that otherwise contaminates the
    momentum signal. Feed the result through :func:`cross_sectional_rank` or
    :func:`cross_sectional_zscore` ``.over(timestamp)`` to build the ranked
    factor.
    """
    c = as_expr(close)
    return (c.shift(skip) / c.shift(lookback) - 1.0).alias(
        f"momentum_{lookback}_{skip}"
    )
