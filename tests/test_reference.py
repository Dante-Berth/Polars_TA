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
