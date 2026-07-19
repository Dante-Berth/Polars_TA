"""Cross-check indicator values against independent NumPy reference implementations."""

import numpy as np
import polars as pl

from polars_ta import momentum, trend, volatility


def _make_ohlcv(n: int = 300, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    volume = rng.uniform(1e4, 1e6, n)
    return pl.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


def _compare(ours: np.ndarray, ref: np.ndarray, atol: float = 1e-6) -> None:
    mask = ~(np.isnan(ours) | np.isnan(ref))
    assert mask.sum() > 0
    assert np.allclose(ours[mask], ref[mask], atol=atol, rtol=1e-4)


def _ref_rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=np.nan)
    up = np.where(diff > 0, diff, 0.0)
    down = np.where(diff < 0, -diff, 0.0)
    alpha = 1.0 / window

    def wilder_ewm(x: np.ndarray) -> np.ndarray:
        out = np.full(len(x), np.nan)
        out[0] = x[0] if not np.isnan(x[0]) else 0.0
        for i in range(1, len(x)):
            xi = x[i] if not np.isnan(x[i]) else 0.0
            out[i] = alpha * xi + (1 - alpha) * out[i - 1]
        return out

    ema_up = wilder_ewm(up)
    ema_down = wilder_ewm(down)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ema_up / ema_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where(ema_down == 0, 100.0, rsi)
    rsi[:window] = np.nan
    return rsi


def _ref_sma(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        out[i] = x[i - window + 1 : i + 1].mean()
    return out


def _ref_atr(high, low, close, window: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR, seeded like Polars' ewm_mean(adjust=False):
    the recursion starts at the first non-null True Range value.
    """
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.nanmax(
        np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]),
        axis=0,
    )
    alpha = 1.0 / window
    out = np.full(len(tr), np.nan)
    out[1] = tr[1]
    for i in range(2, len(tr)):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i - 1]
    out[: window - 1] = np.nan  # library requires `window` samples before reporting
    return out


def _wilder_ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    """Wilder smoothing seeded like Polars' ewm_mean(adjust=False): start the
    recursion at the first non-null value, propagating nulls before it."""
    out = np.full(len(x), np.nan)
    started = False
    for i in range(len(x)):
        xi = x[i]
        if np.isnan(xi):
            if started:
                out[i] = out[i - 1]  # carry forward within the recursion
            continue
        if not started:
            out[i] = xi
            started = True
        else:
            out[i] = alpha * xi + (1 - alpha) * out[i - 1]
    return out


def _ref_adx(high, low, close, window: int = 14):
    """Independent Wilder ADX / +DI / -DI reference matching this library's
    ewm_mean(adjust=False) seeding, with flat-market denominators guarded."""
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.nanmax(
        np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]),
        axis=0,
    )
    tr[0] = np.nan

    up = np.diff(high, prepend=np.nan)
    down = -np.diff(low, prepend=np.nan)
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)

    alpha = 1.0 / window
    s_tr = _wilder_ewm(tr, alpha)
    s_pos = _wilder_ewm(pos_dm, alpha)
    s_neg = _wilder_ewm(neg_dm, alpha)

    with np.errstate(divide="ignore", invalid="ignore"):
        dip = np.where(s_tr == 0, 0.0, 100 * s_pos / s_tr)
        din = np.where(s_tr == 0, 0.0, 100 * s_neg / s_tr)
        di_sum = dip + din
        dx = np.where(di_sum == 0, 0.0, 100 * np.abs(dip - din) / di_sum)
    adx = _wilder_ewm(dx, alpha)
    return adx, dip, din


def test_adx_matches_reference():
    # Regression guard for the flat-market divide-by-zero that previously made
    # adx() return all-NaN. Wilder's EMA has infinite memory, so compare the
    # converged tail (past the warm-up) of the recursive formula.
    df = _make_ohlcv()
    high, low, close = (
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    ref_adx, ref_dip, ref_din = _ref_adx(high, low, close)

    ours_adx = df.select(trend.adx("high", "low", "close").alias("v"))["v"].to_numpy()
    ours_dip = df.select(trend.adx_pos("high", "low", "close").alias("v"))[
        "v"
    ].to_numpy()
    ours_din = df.select(trend.adx_neg("high", "low", "close").alias("v"))[
        "v"
    ].to_numpy()

    _compare(ours_adx[100:], ref_adx[100:], atol=1e-6)
    _compare(ours_dip[100:], ref_dip[100:], atol=1e-6)
    _compare(ours_din[100:], ref_din[100:], atol=1e-6)
    # And the property that regressed: no NaN leakage past warm-up.
    assert np.isfinite(ours_adx[100:]).all()


def test_adx_flat_market_is_finite():
    # A perfectly flat market has zero true range everywhere; ADX must report
    # zero directional movement, not NaN.
    n = 120
    flat = np.full(n, 100.0)
    df = pl.DataFrame({"high": flat, "low": flat, "close": flat})
    for fn in ("adx", "adx_pos", "adx_neg"):
        out = df.select(getattr(trend, fn)("high", "low", "close").alias("v"))[
            "v"
        ].to_numpy()
        finite = out[~np.isnan(out)]
        assert np.isfinite(finite).all()
        assert (finite == 0).all()


def test_rsi_matches_reference():
    df = _make_ohlcv()
    ours = df.select(momentum.rsi("close", window=14).alias("v"))["v"].to_numpy()
    ref = _ref_rsi(df["close"].to_numpy(), window=14)
    _compare(ours, ref, atol=1e-6)


def test_sma_matches_reference():
    df = _make_ohlcv()
    ours = df.select(trend.sma_indicator("close", 20).alias("v"))["v"].to_numpy()
    ref = _ref_sma(df["close"].to_numpy(), 20)
    _compare(ours, ref, atol=1e-9)


def test_atr_matches_reference():
    # Wilder's EMA has infinite memory, so its early values are sensitive to
    # exactly how the recursion is seeded; only the converged tail is a
    # meaningful cross-check of the recursive formula itself.
    df = _make_ohlcv()
    ours = df.select(volatility.average_true_range("high", "low", "close").alias("v"))[
        "v"
    ].to_numpy()
    ref = _ref_atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy())
    _compare(ours[100:], ref[100:], atol=1e-6)


def test_bollinger_matches_reference():
    df = _make_ohlcv()
    close = df["close"].to_numpy()
    window, window_dev = 20, 2
    ours = df.select(
        volatility.bollinger_hband("close", window, window_dev).alias("v")
    )["v"].to_numpy()

    mavg = _ref_sma(close, window)
    mstd = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        mstd[i] = close[i - window + 1 : i + 1].std(ddof=0)
    ref = mavg + window_dev * mstd
    _compare(ours, ref, atol=1e-6)
