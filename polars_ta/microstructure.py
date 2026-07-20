"""Market microstructure and order-flow features used on professional/quant desks.

These are largely absent from retail TA libraries (they need bar-level volume
classification, autocovariance of returns, or scaling-law fits rather than a
simple rolling window), but are standard tools for liquidity analysis,
informed-trading detection, and regime classification on institutional desks.
"""

import math

import numpy as np
import polars as pl

from polars_ta._internal import (
    as_expr,
    log_return,
    rolling_beta,
    rolling_cov,
)

# Numba is an optional accelerator (`pip install polars-ta-lib[speed]`). When
# present, the sequential VPIN volume-bucketing loop is JIT-compiled; when
# absent we fall back to the identical pure-Python loop. Output is byte-for-byte
# the same either way — numba only changes how fast the loop runs.
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in so the kernel stays importable without numba."""

        def _decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return _decorator


@njit(cache=True)
def _vpin_bucketing(vol, buy_frac, bucket_size):
    """Sequential volume-bucketing accumulator (the one genuinely serial step).

    Walks bars in order, splitting each bar's classified buy/sell volume into
    equal-sized volume buckets (a bar's volume may straddle a bucket boundary).
    Returns per-bucket buy/sell totals plus, for each bar, the index of the
    bucket it fell into (-1 while the trailing bucket is still filling).
    """
    n = len(vol)
    bucket_idx = np.full(n, -1, dtype=np.int64)
    # A single bar can straddle (and fully fill) several buckets, so the number
    # of buckets is bounded by total volume / bucket_size, not by n.
    max_buckets = int(np.sum(vol) / bucket_size) + 2
    buckets_buy = np.empty(max_buckets, dtype=np.float64)
    buckets_sell = np.empty(max_buckets, dtype=np.float64)

    remaining = float(bucket_size)
    buy_acc = 0.0
    sell_acc = 0.0
    cur_bucket = 0

    for i in range(n):
        v = vol[i]
        total_v = v
        buy_v = v * buy_frac[i]
        sell_v = v - buy_v
        while v > 0.0:
            take = v if v < remaining else remaining
            frac = take / total_v if total_v > 0.0 else 0.0
            buy_acc += buy_v * frac
            sell_acc += sell_v * frac
            remaining -= take
            v -= take
            bucket_idx[i] = cur_bucket
            if remaining <= 0.0:
                buckets_buy[cur_bucket] = buy_acc
                buckets_sell[cur_bucket] = sell_acc
                buy_acc = 0.0
                sell_acc = 0.0
                remaining = float(bucket_size)
                cur_bucket += 1

    return buckets_buy[:cur_bucket], buckets_sell[:cur_bucket], bucket_idx


def roll_spread(close: str | pl.Expr, window: int = 20) -> pl.Expr:
    """Roll (1984) implied bid-ask spread from serial covariance of price changes.

    Estimates the effective spread purely from trade prices, using the fact
    that bid-ask bounce induces negative first-order autocovariance in price
    changes: spread = 2 * sqrt(-cov(delta_p_t, delta_p_t-1)) when that
    covariance is negative (as microstructure theory predicts); the window
    is reported as null when the covariance is non-negative, since the
    model's premise doesn't hold there.
    """
    close = as_expr(close)
    delta_p = close.diff(1)

    cov = rolling_cov(delta_p, delta_p.shift(1), window)

    spread = pl.when(cov < 0).then(2.0 * (-cov).sqrt()).otherwise(None)
    return spread.alias(f"roll_spread_{window}")


def kyle_lambda(
    close: str | pl.Expr, volume: str | pl.Expr, window: int = 20
) -> pl.Expr:
    """Kyle's (1985) lambda: price impact per unit of signed order flow.

    Regresses price changes on signed volume (sign of the price change used
    as a trade-direction proxy, i.e. a tick rule) within a rolling window.
    A steeper slope means the market absorbs less volume per unit of price
    move — thinner, less liquid conditions. This is the workhorse price-impact
    measure for execution/market-making desks sizing orders against liquidity.
    """
    close = as_expr(close)
    volume = as_expr(volume)

    delta_p = close.diff(1)
    signed_volume = volume * delta_p.sign()

    # Slope of price change on signed volume: cov(sv, dp) / var(sv).
    return rolling_beta(delta_p, signed_volume, window).alias(f"kyle_lambda_{window}")


def hasbrouck_lambda(
    close: str | pl.Expr, volume: str | pl.Expr, window: int = 20
) -> pl.Expr:
    """Hasbrouck's (1991) lambda: price impact regressed in log-price /
    sqrt(dollar-volume) space rather than Kyle's raw price/volume space.

    Using sqrt(signed dollar volume) makes the impact measure robust to the
    typical square-root law of market impact, which is closer to what
    execution desks actually observe versus Kyle's linear assumption.
    """
    close = as_expr(close)
    volume = as_expr(volume)

    log_ret = log_return(close)
    dollar_volume = close * volume
    signed_sqrt_dv = dollar_volume.sqrt() * log_ret.sign()

    # Slope of log return on signed sqrt dollar volume.
    return rolling_beta(log_ret, signed_sqrt_dv, window).alias(
        f"hasbrouck_lambda_{window}"
    )


def effective_spread(
    close: str | pl.Expr, mid_price: str | pl.Expr | None = None
) -> pl.Expr:
    """Effective spread proxy: 2 * |close - mid|, in the same units as price.

    When no explicit mid-price/quote data is available (the common case for
    bar data), the previous close is used as a proxy for the prevailing mid,
    which is the standard fallback in the academic microstructure literature
    when only trade prices are observable.
    """
    close = as_expr(close)
    if mid_price is None:
        mid = close.shift(1)
    else:
        mid = as_expr(mid_price)
    return (2.0 * (close - mid).abs()).alias("effective_spread")


def lee_ready_trade_sign(
    close: str | pl.Expr, mid_price: str | pl.Expr | None = None
) -> pl.Expr:
    """Lee-Ready trade-side classification: +1 buy-initiated, -1
    sell-initiated, 0 unclassifiable, per bar/trade.

    The full Lee & Ready (1991) algorithm classifies by the quote test
    (trade price vs. the prevailing bid-ask midpoint) first, falling back to
    the tick test (trade price vs. the previous trade price) only when the
    trade is exactly at the midpoint. Bar data has no quotes, so when
    `mid_price` isn't supplied the quote test is skipped entirely (comparing
    close against its own previous value as "mid" would make the quote test
    degenerate into the tick test) and classification falls straight to the
    tick test — the standard reduction used when only trade prices are
    observable. A genuine tie (flat price) is unclassifiable (`0`), never
    guessed.
    """
    close = as_expr(close)
    prev_close = close.shift(1)
    tick_sign = (
        pl.when(close > prev_close)
        .then(1)
        .when(close < prev_close)
        .then(-1)
        .otherwise(None)
    )

    if mid_price is None:
        return tick_sign.fill_null(0).alias("trade_sign")

    mid = as_expr(mid_price)
    quote_sign = pl.when(close > mid).then(1).when(close < mid).then(-1).otherwise(None)
    return quote_sign.fill_null(tick_sign).fill_null(0).alias("trade_sign")


def vpin(
    close: str | pl.Expr,
    volume: str | pl.Expr,
    bucket_size: int,
    window: int = 50,
) -> pl.Expr:
    """Volume-Synchronized Probability of Informed Trading (Easley, Lopez de
    Prado & O'Hara, 2012).

    Bars are aggregated into equal-sized *volume buckets* (not time bars),
    each bucket's volume is split into buy/sell using bulk volume
    classification (a z-scored price-change CDF, standard for bar data
    without tick-level trade direction), and VPIN is the rolling mean of
    |buy - sell| / total volume across the last `window` buckets. High VPIN
    signals order-flow imbalance consistent with informed trading — the
    canonical warning signal ahead of liquidity-driven crashes (e.g. the
    2010 Flash Crash), used on institutional desks for flow-toxicity
    monitoring rather than directional signal generation.

    Args:
        bucket_size: total volume per synchronized bucket (must match the
            typical scale of `volume` for the instrument).
        window: number of trailing buckets averaged into the VPIN estimate.
    """
    close = pl.col(close) if isinstance(close, str) else close
    volume = pl.col(volume) if isinstance(volume, str) else volume

    def _calc_vpin(struct_s: pl.Series) -> pl.Series:
        df = struct_s.struct.unnest()
        price = df["close"].to_numpy()
        vol = df["volume"].to_numpy()
        n = len(price)

        ret = np.diff(price, prepend=price[0])
        std = np.nanstd(ret) if np.nanstd(ret) > 0 else 1.0
        z = ret / std
        # Standard normal CDF via erf (bulk volume classification, Easley et al.)
        buy_frac = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

        # The volume-bucketing accumulator is the one genuinely sequential
        # step; hand it to the (optionally Numba-compiled) kernel. Everything
        # around it — classification above, rolling mean below — is vectorized.
        buy_arr, sell_arr, bucket_idx = _vpin_bucketing(
            np.ascontiguousarray(vol, dtype=np.float64),
            np.ascontiguousarray(buy_frac, dtype=np.float64),
            float(bucket_size),
        )

        if len(buy_arr) == 0:
            return pl.Series([None] * n, dtype=pl.Float64)

        imbalance = np.abs(buy_arr - sell_arr)
        total = buy_arr + sell_arr
        safe_total = np.where(total == 0, np.nan, total)
        oi = imbalance / safe_total

        # Trailing rolling mean over the last `window` buckets, ignoring
        # zero-total buckets (NaN in `oi`) — vectorized via a sliding-window
        # view instead of a per-bucket Python loop. The first window-1 buckets
        # stay NaN (not enough history yet).
        vpin_buckets = np.full(len(oi), np.nan)
        if len(oi) >= window:
            windows = np.lib.stride_tricks.sliding_window_view(oi, window)
            with np.errstate(invalid="ignore"):
                means = np.where(
                    np.isnan(windows).all(axis=1),
                    np.nan,
                    np.nanmean(windows, axis=1),
                )
            vpin_buckets[window - 1 :] = means

        # Broadcast each bucket's VPIN back onto its member bars via fancy
        # indexing. Bars in the trailing, not-yet-filled bucket carry
        # bucket_idx == -1 and stay null (how every indicator signals "not
        # enough data yet").
        out = np.full(n, np.nan)
        valid = (bucket_idx >= 0) & (bucket_idx < len(vpin_buckets))
        out[valid] = vpin_buckets[bucket_idx[valid]]
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    expr = pl.struct([close.alias("close"), volume.alias("volume")]).map_batches(
        _calc_vpin, returns_scalar=False
    )
    return expr.alias(f"vpin_{bucket_size}_{window}")


def _hurst_from_window(x: np.ndarray) -> float:
    """Rescaled-range (R/S) Hurst exponent for a single 1-D window of returns."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return float("nan")

    # Classic R/S analysis: split the return series into sub-windows of
    # increasing size, and regress log(R/S) on log(window size). The slope
    # of that regression is the Hurst exponent.
    max_chunk = n // 2
    chunk_sizes = sorted(
        {int(s) for s in np.unique(np.geomspace(8, max_chunk, num=10).astype(int))}
    )

    log_sizes = []
    log_rs = []
    for chunk in chunk_sizes:
        n_chunks = n // chunk
        if n_chunks < 1:  # pragma: no cover - chunk <= n//2 guarantees >= 2
            continue
        # Vectorize over all non-overlapping chunks at once instead of a
        # Python loop per chunk: reshape into (n_chunks, chunk) and reduce
        # along the last axis.
        trimmed = x[: n_chunks * chunk].reshape(n_chunks, chunk)
        dev = np.cumsum(trimmed - trimmed.mean(axis=1, keepdims=True), axis=1)
        r = dev.max(axis=1) - dev.min(axis=1)
        s_dev = trimmed.std(axis=1)
        mask = s_dev > 0
        if mask.any():
            log_sizes.append(np.log(chunk))
            log_rs.append(np.log(np.mean(r[mask] / s_dev[mask])))

    if len(log_sizes) < 2:
        return float("nan")

    return float(np.polyfit(log_sizes, log_rs, 1)[0])


def hurst_exponent(close: str | pl.Expr, window: int = 100) -> pl.Expr:
    """Rolling Hurst exponent via rescaled-range (R/S) analysis.

    H < 0.5 indicates a mean-reverting regime, H = 0.5 a random walk, and
    H > 0.5 a trending/persistent regime. Quant desks use this to switch
    strategy family (momentum vs mean-reversion) rather than as a trade
    signal by itself.

    Implemented with a single ``map_batches`` pass and a sliding-window view
    (no per-row Python ``rolling_map``): each window's R/S regression is
    computed over NumPy-reshaped chunks, so cost is dominated by vectorized
    array ops rather than an O(window) Python loop per row.
    """
    close = as_expr(close)
    log_ret = log_return(close)

    def _rolling_hurst(s: pl.Series) -> pl.Series:
        x = s.to_numpy()
        n = len(x)
        out = np.full(n, np.nan)
        if n < window:
            return pl.Series(out, dtype=pl.Float64).fill_nan(None)
        # Sliding windows as a zero-copy view; evaluate R/S per window.
        views = np.lib.stride_tricks.sliding_window_view(x, window)
        for i in range(views.shape[0]):
            out[window - 1 + i] = _hurst_from_window(views[i])
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return log_ret.map_batches(_rolling_hurst, returns_scalar=False).alias(
        f"hurst_{window}"
    )


def variance_ratio(close: str | pl.Expr, window: int = 20, lag: int = 2) -> pl.Expr:
    """Lo-MacKinlay (1988) variance ratio test statistic, rolled over `window`.

    VR(lag) = Var(lag-period return) / (lag * Var(1-period return)). Under
    the random-walk null hypothesis VR = 1; VR > 1 indicates positive serial
    correlation (trending), VR < 1 indicates mean reversion. Used by quant
    desks to test whether a random-walk assumption (and therefore standard
    options-pricing / risk models built on it) actually holds for an
    instrument over a given regime.
    """
    close = as_expr(close)
    ret_1 = log_return(close, periods=1)
    ret_lag = log_return(close, periods=lag)

    var_1 = ret_1.rolling_var(window_size=window)
    var_lag = ret_lag.rolling_var(window_size=window)

    safe_var_1 = pl.when(var_1 == 0).then(None).otherwise(var_1)
    vr = var_lag / (lag * safe_var_1)
    return vr.alias(f"variance_ratio_{lag}_{window}")


def corwin_schultz_spread(
    high: str | pl.Expr, low: str | pl.Expr, window: int = 20
) -> pl.Expr:
    """Corwin-Schultz (2012) high-low bid-ask spread estimator.

    Recovers the effective spread from two consecutive daily high-low ranges
    only (no volume, no quotes). The insight is that the high-low range
    reflects both the true variance (which scales with the time interval) and
    the spread (which does not), so combining a single-bar range with a
    two-bar range separates the two. This is the modern successor to Roll's
    estimator and is far more robust on OHLC bar data; negative per-bar
    estimates (which the model treats as zero spread) are floored at zero and
    the result is averaged over ``window`` bars.
    """
    high = as_expr(high)
    low = as_expr(low)

    # beta: sum of two consecutive single-bar squared log high/low ranges.
    hl = (high / low).log().pow(2)
    beta = hl + hl.shift(1)

    # gamma: squared log range over the two-bar high and two-bar low.
    high2 = pl.max_horizontal(high, high.shift(1))
    low2 = pl.min_horizontal(low, low.shift(1))
    gamma = (high2 / low2).log().pow(2)

    denom = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (beta.sqrt() * (math.sqrt(2.0) - 1.0) / denom) - (gamma / denom).sqrt()

    # Spread as a fraction of price; negative alpha -> zero spread.
    ea = alpha.exp()
    spread = 2.0 * (ea - 1.0) / (1.0 + ea)
    spread = pl.when(spread < 0).then(0.0).otherwise(spread)
    return spread.rolling_mean(window_size=window).alias(f"corwin_schultz_{window}")


def half_life(close: str | pl.Expr, window: int = 60) -> pl.Expr:
    """Half-life of mean reversion from a rolling Ornstein-Uhlenbeck fit.

    Regresses the change in price on the lagged price level within each
    window (an AR(1) / discretized OU fit): ``dP_t = a + b * P_{t-1}``. When
    ``b < 0`` the series is mean-reverting and the half-life — the expected
    number of bars to close half the gap to the mean — is ``-ln(2) / ln(1+b)``.
    Non-mean-reverting windows (``b >= 0``) are reported as null. This is the
    workhorse "how fast does it revert" number on stat-arb desks, pairing
    naturally with :func:`variance_ratio` and :func:`hurst_exponent`.
    """
    close = as_expr(close)
    y = close.diff(1)  # dP_t
    x = close.shift(1)  # P_{t-1}

    # AR(1) slope b in dP_t = a + b * P_{t-1}.
    b = rolling_beta(y, x, window)
    # Only mean-reverting fits (b in (-1, 0)) yield a finite positive half-life.
    valid = (b < 0) & (b > -1)
    hl = pl.when(valid).then(-math.log(2) / (1.0 + b).log()).otherwise(None)
    return hl.alias(f"half_life_{window}")


def _shannon_entropy_from_window(x: np.ndarray, n_bins: int) -> float:
    """Shannon entropy (in bits) of the binned distribution of a single
    1-D window of returns, normalized to [0, 1] by dividing by log2(n_bins)
    (the maximum possible entropy for that many bins)."""
    x = x[~np.isnan(x)]
    if len(x) < n_bins:
        return float("nan")

    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts[counts > 0] / len(x)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(n_bins))


def shannon_entropy(
    close: str | pl.Expr, window: int = 50, n_bins: int = 10
) -> pl.Expr:
    """Rolling Shannon entropy (normalized to [0, 1]) of the binned
    distribution of log returns within each window.

    A value near 1 means returns within the window are spread roughly
    uniformly across bins — high "surprise"/complexity, consistent with a
    noisy or regime-shifting market. A value near 0 means returns cluster
    into a few bins — low complexity, consistent with a persistent trend or
    a tightly range-bound market. Unlike the Hurst exponent (which measures
    *directional persistence*), entropy measures *distributional
    concentration* and doesn't care about sign or serial correlation, so the
    two are complementary regime signals rather than redundant ones.

    A window with fewer non-null returns than `n_bins` can't populate every
    bin meaningfully and reports null rather than a misleadingly low entropy.
    """
    close = as_expr(close)
    log_ret = log_return(close)

    def _rolling_entropy(s: pl.Series) -> pl.Series:
        x = s.to_numpy()
        n = len(x)
        out = np.full(n, np.nan)
        if n < window:
            return pl.Series(out, dtype=pl.Float64).fill_nan(None)
        views = np.lib.stride_tricks.sliding_window_view(x, window)
        for i in range(views.shape[0]):
            out[window - 1 + i] = _shannon_entropy_from_window(views[i], n_bins)
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return log_ret.map_batches(_rolling_entropy, returns_scalar=False).alias(
        f"shannon_entropy_{window}"
    )


def _approximate_entropy_from_window(
    x: np.ndarray, m: int, r_frac: float
) -> float:
    """Approximate entropy (Pincus, 1991) of a single 1-D window.

    O(window^2) per window (all pairwise embedded-vector distances) — the
    textbook algorithm has no known sub-quadratic exact form, so this is
    only affordable at the small-to-moderate window sizes (a few dozen bars)
    typical for a "how predictable has this series been recently" gate, not
    as a rolling feature computed over thousands of bars of history at once.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < m + 2:
        return float("nan")

    r = r_frac * np.std(x)
    if r == 0:
        return float("nan")

    def _phi(m_: int) -> float:
        n_vec = n - m_ + 1
        # Embed into overlapping m_-length vectors via a sliding-window view
        # (zero-copy), then compute all pairwise Chebyshev (max-norm)
        # distances at once instead of a nested Python loop.
        vectors = np.lib.stride_tricks.sliding_window_view(x, m_)[:n_vec]
        dist = np.max(
            np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=-1
        )
        counts = np.sum(dist <= r, axis=1)
        return float(np.mean(np.log(counts / n_vec)))

    return float(_phi(m) - _phi(m + 1))


def approximate_entropy(
    close: str | pl.Expr, window: int = 30, m: int = 2, r_frac: float = 0.2
) -> pl.Expr:
    """Rolling approximate entropy (ApEn) of log returns within each window.

    Measures how predictable consecutive `m`-length patterns are: low ApEn
    means the series repeats similar short patterns (more regular/
    predictable), high ApEn means patterns rarely recur (more random). `r`
    (the similarity tolerance) is set as a fraction (`r_frac`, Pincus'
    convention is ~0.2) of the window's own standard deviation, so it
    self-scales with local volatility rather than needing an absolute
    threshold tuned per instrument.

    **Cost warning:** this is O(window^2) per row via `map_batches` (see
    :func:`_approximate_entropy_from_window`) — the classic algorithm has no
    faster exact form. Keep `window` in the tens, not hundreds, on large
    frames; this is meant as a slow-moving regime gate, not a per-bar signal
    computed over a huge lookback.
    """
    close = as_expr(close)
    log_ret = log_return(close)

    def _rolling_apen(s: pl.Series) -> pl.Series:
        x = s.to_numpy()
        n = len(x)
        out = np.full(n, np.nan)
        if n < window:
            return pl.Series(out, dtype=pl.Float64).fill_nan(None)
        views = np.lib.stride_tricks.sliding_window_view(x, window)
        for i in range(views.shape[0]):
            out[window - 1 + i] = _approximate_entropy_from_window(
                views[i], m, r_frac
            )
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return log_ret.map_batches(_rolling_apen, returns_scalar=False).alias(
        f"approx_entropy_{window}"
    )
