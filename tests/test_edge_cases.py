"""Edge-case coverage for branches the smoke suite doesn't reach: class-only
combination indicators, non-default flags, and degenerate inputs."""

import numpy as np
import polars as pl

from polars_ta import microstructure as ms
from polars_ta import trend, volatility, volume
from polars_ta.trend import ComplexTrendIndicators, TrendIndicators
from polars_ta.volume import VolumeIndicators

N = 300
_rng = np.random.default_rng(21)
_close = 100 + np.cumsum(_rng.normal(0, 1, N))
DF = pl.DataFrame(
    {
        "high": _close + _rng.uniform(0.1, 1.0, N),
        "low": _close - _rng.uniform(0.1, 1.0, N),
        "close": _close,
        "volume": _rng.uniform(1e4, 1e5, N),
    }
)


def _finite_and_full_length(expr: pl.Expr) -> np.ndarray:
    out = DF.select(expr.alias("v"))["v"]
    assert len(out) == N
    vals = out.drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.isfinite(vals).all()
    return vals


def test_aroon_indicator_diff():
    up = DF.select(trend.aroon_up("high", "low").alias("v"))["v"].to_numpy()
    down = DF.select(trend.aroon_down("high", "low").alias("v"))["v"].to_numpy()
    diff = DF.select(TrendIndicators.aroon_indicator("high", "low").alias("v"))[
        "v"
    ].to_numpy()
    np.testing.assert_allclose(diff, up - down, atol=1e-12)


def test_kst_diff_is_kst_minus_signal():
    kst = DF.select(trend.kst("close").alias("v"))["v"].to_numpy()
    sig = DF.select(trend.kst_sig("close").alias("v"))["v"].to_numpy()
    diff = DF.select(TrendIndicators.kst_diff("close").alias("v"))["v"].to_numpy()
    np.testing.assert_allclose(diff, kst - sig, atol=1e-9)


def test_vortex_diff_is_pos_minus_neg():
    pos = DF.select(trend.vortex_indicator_pos("high", "low", "close").alias("v"))[
        "v"
    ].to_numpy()
    neg = DF.select(trend.vortex_indicator_neg("high", "low", "close").alias("v"))[
        "v"
    ].to_numpy()
    diff = DF.select(
        ComplexTrendIndicators.vortex_diff("high", "low", "close").alias("v")
    )["v"].to_numpy()
    np.testing.assert_allclose(diff, pos - neg, atol=1e-9)


def test_ichimoku_visual_mode_shifts_forward():
    for fn in (trend.ichimoku_a, trend.ichimoku_b):
        plotted = _finite_and_full_length(fn("high", "low", visual=True))
        assert len(plotted) == N  # shift + mean-fill leaves no gaps


def test_keltner_ema_variant():
    # original_version=False switches the midline from typical-price SMA to
    # close EMA +/- ATR bands.
    hband = _finite_and_full_length(
        volatility.keltner_channel_hband("high", "low", "close", original_version=False)
    )
    lband = _finite_and_full_length(
        volatility.keltner_channel_lband("high", "low", "close", original_version=False)
    )
    assert (hband[-50:] > lband[-50:]).all()


def test_vpt_smoothed():
    raw = DF.select(volume.volume_price_trend("close", "volume").alias("v"))["v"]
    smoothed = DF.select(
        VolumeIndicators.volume_price_trend(
            "close", "volume", smoothing_factor=5
        ).alias("v")
    )["v"]
    assert smoothed.null_count() > raw.null_count()  # rolling warm-up added
    assert np.isfinite(smoothed.drop_nulls().to_numpy()).all()


def test_psar_single_row():
    df1 = DF.head(1)
    out = df1.select(trend.psar("high", "low", "close").alias("v"))["v"]
    assert len(out) == 1


def test_hurst_flat_series_is_null():
    # Constant prices -> zero-variance returns -> R/S undefined -> null.
    flat = pl.DataFrame({"close": np.full(150, 100.0)})
    out = flat.select(ms.hurst_exponent("close", window=100).alias("v"))["v"]
    assert out.null_count() == len(out)


def test_hurst_window_too_short_is_null():
    out = DF.select(ms.hurst_exponent("close", window=15).alias("v"))["v"]
    assert out.null_count() == len(out)  # < 20 samples per window -> undefined


def test_vpin_bucket_never_fills():
    # A bucket_size far above total volume: no bucket ever completes, so
    # every row is warm-up null.
    out = DF.select(ms.vpin("close", "volume", bucket_size=1e12).alias("v"))["v"]
    assert out.null_count() == len(out)
