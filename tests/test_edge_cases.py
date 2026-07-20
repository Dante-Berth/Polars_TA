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


def test_supertrend_shorter_than_atr_window_is_all_null():
    # Fewer bars than the ATR warm-up window: basic_upper/lower never leave
    # their warm-up null, so the whole output stays null (the all-NaN branch
    # of the map_batches kernel).
    short = DF.head(5)
    out = short.select(
        trend.supertrend("high", "low", "close", window=10).alias("v")
    )["v"]
    assert out.null_count() == len(out)


def test_klinger_volume_oscillator_trend_flip():
    # A V-shaped price path forces a trend flip partway through, exercising
    # the branch where the cumulative range resets instead of accumulating.
    close = np.concatenate([np.linspace(100, 90, 40), np.linspace(90, 110, 40)])
    df = pl.DataFrame(
        {
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(80, 1000.0),
        }
    )
    out = df.select(
        volume.klinger_volume_oscillator(
            "high", "low", "close", "volume", window_fast=5, window_slow=10
        ).alias("v")
    )["v"]
    vals = out.drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.isfinite(vals).all()


def test_vpin_bucket_never_fills():
    # A bucket_size far above total volume: no bucket ever completes, so
    # every row is warm-up null.
    out = DF.select(ms.vpin("close", "volume", bucket_size=1e12).alias("v"))["v"]
    assert out.null_count() == len(out)


def test_shannon_entropy_constant_series_is_zero():
    # A flat price -> all-zero returns -> every observation falls in one bin
    # -> minimum possible entropy.
    flat = pl.DataFrame({"close": np.full(100, 100.0)})
    out = flat.select(ms.shannon_entropy("close", window=50, n_bins=10).alias("v"))[
        "v"
    ]
    vals = out.drop_nulls().to_numpy()
    assert len(vals) > 0
    np.testing.assert_allclose(vals, 0.0, atol=1e-12)


def test_shannon_entropy_uniform_noise_is_near_one():
    # Returns spread roughly uniformly across bins -> entropy near the
    # theoretical maximum of 1.0.
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.uniform(-1, 1, 500))
    df = pl.DataFrame({"close": close})
    out = df.select(ms.shannon_entropy("close", window=200, n_bins=10).alias("v"))["v"]
    vals = out.drop_nulls().to_numpy()
    assert vals.mean() > 0.85


def test_approximate_entropy_constant_series_is_null():
    # Zero-variance window -> r (similarity tolerance) is 0 -> undefined,
    # reported as null rather than a fabricated 0.
    flat = pl.DataFrame({"close": np.full(60, 100.0)})
    out = flat.select(ms.approximate_entropy("close", window=30).alias("v"))["v"]
    assert out.null_count() == len(out)


def test_shannon_entropy_fewer_returns_than_bins_is_null():
    # window=5 with n_bins=10: even a full window can't populate 10 bins
    # meaningfully, so every row stays null.
    short = pl.DataFrame({"close": DF["close"].to_numpy()[:20]})
    out = short.select(
        ms.shannon_entropy("close", window=5, n_bins=10).alias("v")
    )["v"]
    assert out.null_count() == len(out)


def test_approximate_entropy_window_shorter_than_m_plus_2_is_null():
    # window=3 with the default m=2 means every window has n < m+2 -> null.
    short = pl.DataFrame({"close": DF["close"].to_numpy()[:20]})
    out = short.select(ms.approximate_entropy("close", window=3, m=2).alias("v"))["v"]
    assert out.null_count() == len(out)


def test_approximate_entropy_repeating_pattern_is_low():
    # A perfectly repeating pattern is maximally predictable -> ApEn near 0.
    pattern = np.tile([1.0, 2.0, 1.0, 0.0], 30)
    close = 100 + np.cumsum(pattern)
    df = pl.DataFrame({"close": close})
    out = df.select(
        ms.approximate_entropy("close", window=40, m=2, r_frac=0.2).alias("v")
    )["v"]
    vals = out.drop_nulls().to_numpy()
    assert len(vals) > 0
    assert vals.mean() < 0.3
