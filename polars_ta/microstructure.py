"""Market microstructure and order-flow features used on professional/quant desks.

These are largely absent from retail TA libraries (they need bar-level volume
classification, autocovariance of returns, or scaling-law fits rather than a
simple rolling window), but are standard tools for liquidity analysis,
informed-trading detection, and regime classification on institutional desks.
"""

import math

import numpy as np
import polars as pl


def roll_spread(close: str | pl.Expr, window: int = 20) -> pl.Expr:
    """Roll (1984) implied bid-ask spread from serial covariance of price changes.

    Estimates the effective spread purely from trade prices, using the fact
    that bid-ask bounce induces negative first-order autocovariance in price
    changes: spread = 2 * sqrt(-cov(delta_p_t, delta_p_t-1)) when that
    covariance is negative (as microstructure theory predicts); the window
    is reported as null when the covariance is non-negative, since the
    model's premise doesn't hold there.
    """
    close = pl.col(close) if isinstance(close, str) else close
    delta_p = close.diff(1)
    delta_p_lag = delta_p.shift(1)

    cov = (
        (delta_p - delta_p.rolling_mean(window_size=window))
        * (delta_p_lag - delta_p_lag.rolling_mean(window_size=window))
    ).rolling_mean(window_size=window)

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
    close = pl.col(close) if isinstance(close, str) else close
    volume = pl.col(volume) if isinstance(volume, str) else volume

    delta_p = close.diff(1)
    signed_volume = volume * delta_p.sign()

    cov = (
        (delta_p - delta_p.rolling_mean(window_size=window))
        * (signed_volume - signed_volume.rolling_mean(window_size=window))
    ).rolling_mean(window_size=window)
    var = (
        (signed_volume - signed_volume.rolling_mean(window_size=window)) ** 2
    ).rolling_mean(window_size=window)

    safe_var = pl.when(var == 0).then(None).otherwise(var)
    return (cov / safe_var).alias(f"kyle_lambda_{window}")


def hasbrouck_lambda(
    close: str | pl.Expr, volume: str | pl.Expr, window: int = 20
) -> pl.Expr:
    """Hasbrouck's (1991) lambda: price impact regressed in log-price /
    sqrt(dollar-volume) space rather than Kyle's raw price/volume space.

    Using sqrt(signed dollar volume) makes the impact measure robust to the
    typical square-root law of market impact, which is closer to what
    execution desks actually observe versus Kyle's linear assumption.
    """
    close = pl.col(close) if isinstance(close, str) else close
    volume = pl.col(volume) if isinstance(volume, str) else volume

    log_ret = (close / close.shift(1)).log()
    dollar_volume = close * volume
    signed_sqrt_dv = dollar_volume.sqrt() * log_ret.sign()

    cov = (
        (log_ret - log_ret.rolling_mean(window_size=window))
        * (signed_sqrt_dv - signed_sqrt_dv.rolling_mean(window_size=window))
    ).rolling_mean(window_size=window)
    var = (
        (signed_sqrt_dv - signed_sqrt_dv.rolling_mean(window_size=window)) ** 2
    ).rolling_mean(window_size=window)

    safe_var = pl.when(var == 0).then(None).otherwise(var)
    return (cov / safe_var).alias(f"hasbrouck_lambda_{window}")


def effective_spread(
    close: str | pl.Expr, mid_price: str | pl.Expr | None = None
) -> pl.Expr:
    """Effective spread proxy: 2 * |close - mid|, in the same units as price.

    When no explicit mid-price/quote data is available (the common case for
    bar data), the previous close is used as a proxy for the prevailing mid,
    which is the standard fallback in the academic microstructure literature
    when only trade prices are observable.
    """
    close = pl.col(close) if isinstance(close, str) else close
    if mid_price is None:
        mid = close.shift(1)
    else:
        mid = pl.col(mid_price) if isinstance(mid_price, str) else mid_price
    return (2.0 * (close - mid).abs()).alias("effective_spread")


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

        buckets_buy: list[float] = []
        buckets_sell: list[float] = []
        remaining = bucket_size
        buy_acc = 0.0
        sell_acc = 0.0
        bucket_idx = np.full(n, -1, dtype=np.int64)

        cur_bucket = 0
        for i in range(n):
            v = vol[i]
            buy_v = v * buy_frac[i]
            sell_v = v - buy_v
            while v > 0:
                take = min(v, remaining)
                frac = take / vol[i] if vol[i] > 0 else 0.0
                buy_acc += buy_v * frac
                sell_acc += sell_v * frac
                remaining -= take
                v -= take
                bucket_idx[i] = cur_bucket
                if remaining <= 0:
                    buckets_buy.append(buy_acc)
                    buckets_sell.append(sell_acc)
                    buy_acc, sell_acc = 0.0, 0.0
                    remaining = bucket_size
                    cur_bucket += 1

        if not buckets_buy:
            return pl.Series([None] * n, dtype=pl.Float64)

        buy_arr = np.array(buckets_buy)
        sell_arr = np.array(buckets_sell)
        imbalance = np.abs(buy_arr - sell_arr)
        total = buy_arr + sell_arr
        safe_total = np.where(total == 0, np.nan, total)
        oi = imbalance / safe_total

        vpin_buckets = np.full(len(oi), np.nan)
        for i in range(len(oi)):
            if i + 1 >= window:
                vpin_buckets[i] = np.nanmean(oi[i + 1 - window : i + 1])

        # Broadcast each bucket's VPIN back onto its member bars. Bars in the
        # trailing, not-yet-filled bucket get a null (consistent with how
        # every other indicator signals "not enough data yet").
        out = np.full(n, np.nan)
        for i in range(n):
            b = bucket_idx[i]
            if 0 <= b < len(vpin_buckets):
                out[i] = vpin_buckets[b]
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    expr = pl.struct([close.alias("close"), volume.alias("volume")]).map_batches(
        _calc_vpin, returns_scalar=False
    )
    return expr.alias(f"vpin_{bucket_size}_{window}")


def hurst_exponent(close: str | pl.Expr, window: int = 100) -> pl.Expr:
    """Rolling Hurst exponent via rescaled-range (R/S) analysis.

    H < 0.5 indicates a mean-reverting regime, H = 0.5 a random walk, and
    H > 0.5 a trending/persistent regime. Quant desks use this to switch
    strategy family (momentum vs mean-reversion) rather than as a trade
    signal by itself.
    """
    close = pl.col(close) if isinstance(close, str) else close
    log_ret = (close / close.shift(1)).log()

    def _calc_hurst(s: pl.Series) -> float:
        x = s.to_numpy()
        x = x[~np.isnan(x)]
        n = len(x)
        if n < 20:
            return float("nan")

        # Classic R/S analysis: split the return series into sub-windows of
        # increasing size, and regress log(R/S) on log(window size). The
        # slope of that regression is the Hurst exponent.
        max_chunk = n // 2
        chunk_sizes = sorted(
            {int(s) for s in np.unique(np.geomspace(8, max_chunk, num=10).astype(int))}
        )

        log_sizes = []
        log_rs = []
        for chunk in chunk_sizes:
            n_chunks = n // chunk
            if n_chunks < 1:
                continue
            rs_vals = []
            for i in range(n_chunks):
                segment = x[i * chunk : (i + 1) * chunk]
                mean = segment.mean()
                deviation = np.cumsum(segment - mean)
                r = deviation.max() - deviation.min()
                s_dev = segment.std()
                if s_dev > 0:
                    rs_vals.append(r / s_dev)
            if rs_vals:
                log_sizes.append(np.log(chunk))
                log_rs.append(np.log(np.mean(rs_vals)))

        if len(log_sizes) < 2:
            return float("nan")

        slope = np.polyfit(log_sizes, log_rs, 1)[0]
        return float(slope)

    return log_ret.rolling_map(_calc_hurst, window_size=window).alias(
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
    close = pl.col(close) if isinstance(close, str) else close
    ret_1 = (close / close.shift(1)).log()
    ret_lag = (close / close.shift(lag)).log()

    var_1 = ret_1.rolling_var(window_size=window)
    var_lag = ret_lag.rolling_var(window_size=window)

    safe_var_1 = pl.when(var_1 == 0).then(None).otherwise(var_1)
    vr = var_lag / (lag * safe_var_1)
    return vr.alias(f"variance_ratio_{lag}_{window}")
