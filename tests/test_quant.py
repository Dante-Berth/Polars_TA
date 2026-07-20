"""Tests for polars_ta.quant, run against a real BTCUSDT 5m sample.

tests/fixtures/btcusdt_5m_sample.arrow is a 5000-row slice of real Binance
BTCUSDT 5-minute OHLCV data (Dec 2025), used instead of synthetic data so
these professional-desk features are validated against real market
microstructure rather than idealized noise.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from polars_ta import quant

FIXTURE = Path(__file__).parent / "fixtures" / "btcusdt_5m_sample.arrow"


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.read_ipc(FIXTURE)


def test_fixture_shape(df):
    assert df.height == 5000
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_yang_zhang_volatility_positive(df):
    out = df.select(
        quant.yang_zhang_volatility("open", "high", "low", "close").alias("v")
    )["v"].drop_nulls()
    assert len(out) > 0
    assert (out >= 0).all()


def test_hurst_ribbon_bounds_and_keys(df):
    exprs = quant.hurst_ribbon("close", scales=(16, 32, 64))
    assert set(exprs) == {"h_16", "h_32", "h_64", "h_ribbon_avg", "h_ribbon_tilt"}

    out = df.with_columns(**exprs)
    for col in ("h_16", "h_32", "h_64", "h_ribbon_avg"):
        vals = out[col].drop_nulls()
        assert len(vals) > 0
        # Hurst exponents from this approximation are not strictly in
        # [0, 1] but should stay within a sane band for liquid crypto data.
        assert vals.min() > -1 and vals.max() < 2

    tilt = out["h_ribbon_tilt"].drop_nulls()
    assert len(tilt) > 0


def test_relative_volume_centered_near_one(df):
    out = df.select(quant.relative_volume("volume").alias("v"))["v"].drop_nulls()
    assert len(out) > 0
    assert (out > 0).all()
    # by construction rvol averages out close to 1 over the full window
    assert 0.5 < out.mean() < 1.5


def test_volatility_z_score_runs(df):
    out = df.select(quant.volatility_z_score("high", "low").alias("v"))[
        "v"
    ].drop_nulls()
    assert len(out) > 0
    assert out.is_finite().all()


def test_parkinson_volatility_positive(df):
    out = df.select(quant.parkinson_volatility("high", "low").alias("v"))[
        "v"
    ].drop_nulls()
    assert len(out) > 0
    assert (out > 0).all()


def test_rogers_satchell_volatility_nonnegative(df):
    out = df.select(
        quant.rogers_satchell_volatility("open", "high", "low", "close").alias("v")
    )["v"].drop_nulls()
    assert len(out) > 0
    assert (out >= 0).all()


def test_quant_features_streaming_equivalence(df):
    lf = df.lazy().with_columns(
        quant.yang_zhang_volatility("open", "high", "low", "close").alias("yz"),
        quant.relative_volume("volume").alias("rvol"),
        **quant.hurst_ribbon("close"),
    )
    default = lf.collect()
    streamed = lf.collect(engine="streaming")
    assert default.equals(streamed)


# --------------------------------------------------------------------------
# Tail-risk & drawdown
# --------------------------------------------------------------------------


def test_norm_ppf_matches_known_quantiles():
    # Standard-normal quantiles, textbook values.
    assert quant._norm_ppf(0.05) == pytest.approx(-1.6448536, abs=1e-5)
    assert quant._norm_ppf(0.01) == pytest.approx(-2.3263479, abs=1e-5)
    assert quant._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert quant._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)


def test_rolling_cvar_matches_numpy_reference(df):
    window, alpha = 100, 0.05
    out = df.select(quant.rolling_cvar("close", window=window, alpha=alpha).alias("v"))[
        "v"
    ].to_numpy()

    ret = df["close"].pct_change().to_numpy()
    k = int(np.ceil(alpha * window))
    i = 3000  # an arbitrary fully-warmed-up row
    w = ret[i - window + 1 : i + 1]
    w = w[~np.isnan(w)]
    expected = -np.sort(w)[:k].mean()
    assert out[i] == pytest.approx(expected, rel=1e-9)


def test_cvar_dominates_when_tail_heavier(df):
    # CVaR (mean of worst alpha) must be >= the VaR quantile magnitude, since
    # the expected shortfall averages losses *at least as bad* as the quantile.
    window, alpha = 100, 0.05
    cvar = df.select(
        quant.rolling_cvar("close", window=window, alpha=alpha).alias("v")
    )["v"].to_numpy()
    ret = df["close"].pct_change().to_numpy()
    i = 3000
    w = ret[i - window + 1 : i + 1]
    w = w[~np.isnan(w)]
    var_q = -np.quantile(w, alpha)
    assert cvar[i] >= var_q - 1e-9


def test_cornish_fisher_var_reduces_to_gaussian_for_normal_data():
    # With (near) symmetric, mesokurtic data the CF correction terms vanish and
    # modified VaR should approach the plain Gaussian VaR mu + z*sigma.
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0, 0.01, 20_000)
    close = 100 * np.cumprod(1 + ret)
    dfn = pl.DataFrame({"close": close})
    w = 2000
    cf = dfn.select(quant.cornish_fisher_var("close", window=w, alpha=0.05).alias("v"))[
        "v"
    ].to_numpy()
    r = pl.Series(close).pct_change().to_numpy()
    i = len(close) - 1
    win = r[i - w + 1 : i + 1]
    gaussian = -(win.mean() + quant._norm_ppf(0.05) * win.std(ddof=1))
    assert cf[i] == pytest.approx(gaussian, rel=0.05)


def test_rolling_max_drawdown_positive_and_bounded(df):
    out = df.select(quant.rolling_max_drawdown("close", window=100).alias("v"))[
        "v"
    ].drop_nulls()
    assert len(out) > 0
    # Drawdown is a positive fraction in [0, 1).
    assert (out >= 0).all()
    assert (out < 1).all()


def test_calmar_null_when_no_drawdown():
    # A strictly monotonically increasing series has zero drawdown -> null Calmar.
    close = np.linspace(100, 200, 400)
    dfm = pl.DataFrame({"close": close})
    out = dfm.select(quant.calmar_ratio("close", window=252).alias("v"))["v"]
    assert out.drop_nulls().len() == 0


# --------------------------------------------------------------------------
# Distribution shape
# --------------------------------------------------------------------------


def test_rolling_skew_matches_numpy(df):
    window = 60
    out = df.select(quant.rolling_skew("close", window=window).alias("v"))[
        "v"
    ].to_numpy()
    ret = df["close"].pct_change().to_numpy()
    i = 3000
    w = ret[i - window + 1 : i + 1]
    # Polars' rolling_skew defaults to bias=True: the population (biased)
    # Fisher-Pearson skewness g1 = m3 / m2**1.5 (no sample correction).
    m = w.mean()
    m2 = np.mean((w - m) ** 2)
    m3 = np.mean((w - m) ** 3)
    expected = m3 / m2**1.5
    assert out[i] == pytest.approx(expected, rel=1e-6)


def test_gain_to_pain_null_when_no_losses():
    close = np.linspace(100, 200, 200)  # every return positive
    dfg = pl.DataFrame({"close": close})
    out = dfg.select(quant.gain_to_pain("close", window=60).alias("v"))["v"]
    assert out.drop_nulls().len() == 0


# --------------------------------------------------------------------------
# Signal conditioning
# --------------------------------------------------------------------------


def test_frac_diff_weights_and_edge_cases():
    # d=1 must reproduce an ordinary first difference of log price.
    w1 = quant._frac_diff_weights(1.0, 5)
    assert w1[0] == pytest.approx(1.0)
    assert w1[1] == pytest.approx(-1.0)
    # weights beyond order 1 are ~0 for d=1 (integer differencing terminates).
    assert all(abs(w) < 1e-12 for w in w1[2:])


def test_frac_diff_d1_equals_log_return(df):
    fd = df.select(quant.frac_diff("close", d=1.0, window=50).alias("v"))[
        "v"
    ].to_numpy()
    lr = df.select((pl.col("close").log().diff(1)).alias("v"))["v"].to_numpy()
    # Ignore warm-up region; compare the overlap.
    mask = ~np.isnan(fd) & ~np.isnan(lr)
    assert np.allclose(fd[mask], lr[mask], atol=1e-9)


def test_frac_diff_retains_more_memory_than_returns(df):
    # Fractional differencing with d<1 should correlate more strongly with the
    # original (level) series than a full first difference does — the whole
    # point of the method (Lopez de Prado, AFML ch. 5).
    fd = df.select(quant.frac_diff("close", d=0.3, window=100).alias("v"))["v"]
    d1 = df.select(quant.frac_diff("close", d=1.0, window=100).alias("v"))["v"]
    level = df["close"]
    frame = pl.DataFrame({"lvl": level, "fd": fd, "d1": d1}).drop_nulls()
    corr_fd = abs(np.corrcoef(frame["lvl"], frame["fd"])[0, 1])
    corr_d1 = abs(np.corrcoef(frame["lvl"], frame["d1"])[0, 1])
    assert corr_fd > corr_d1


def test_rolling_ic_perfect_signal():
    # If the signal *is* the forward return, the IC must be ~1.
    rng = np.random.default_rng(1)
    fwd = rng.normal(0, 1, 500)
    dfi = pl.DataFrame({"sig": fwd, "fwd": fwd})
    out = dfi.select(quant.rolling_ic("sig", "fwd", window=60).alias("v"))[
        "v"
    ].drop_nulls()
    assert (out > 0.999).all()


# --------------------------------------------------------------------------
# Cross-sectional / factor
# --------------------------------------------------------------------------


def test_rolling_beta_to_recovers_known_beta():
    # Construct asset = 1.5 * bench_returns + small noise; beta ~ 1.5.
    rng = np.random.default_rng(2)
    rb = rng.normal(0, 0.01, 4000)
    ra = 1.5 * rb + rng.normal(0, 1e-5, 4000)
    bench = 100 * np.cumprod(1 + rb)
    asset = 100 * np.cumprod(1 + ra)
    dfb = pl.DataFrame({"close": asset, "bench": bench})
    out = dfb.select(quant.rolling_beta_to("close", "bench", window=250).alias("v"))[
        "v"
    ].drop_nulls()
    assert out.median() == pytest.approx(1.5, abs=0.05)


def test_idiosyncratic_vol_near_zero_when_asset_tracks_bench():
    # Asset perfectly tracks benchmark -> residual (idiosyncratic) vol ~ 0.
    rng = np.random.default_rng(3)
    rb = rng.normal(0, 0.01, 2000)
    bench = 100 * np.cumprod(1 + rb)
    asset = bench.copy()
    dfi = pl.DataFrame({"close": asset, "bench": bench})
    out = dfi.select(quant.idiosyncratic_vol("close", "bench", window=120).alias("v"))[
        "v"
    ].drop_nulls()
    assert out.max() < 1e-6


def test_momentum_12_1_skips_recent_window(df):
    lookback, skip = 252, 21
    out = df.select(
        quant.momentum_12_1("close", lookback=lookback, skip=skip).alias("v")
    )["v"].to_numpy()
    c = df["close"].to_numpy()
    i = 3000
    expected = c[i - skip] / c[i - lookback] - 1.0
    assert out[i] == pytest.approx(expected, rel=1e-9)


def test_downside_beta_only_uses_down_bars():
    # Asset up-bars deliberately anti-correlated, down-bars 2x the bench:
    # symmetric beta and downside beta must differ.
    rng = np.random.default_rng(4)
    rb = rng.normal(0, 0.01, 3000)
    ra = np.where(rb < 0, 2.0 * rb, -1.0 * rb) + rng.normal(0, 1e-6, 3000)
    bench = 100 * np.cumprod(1 + rb)
    asset = 100 * np.cumprod(1 + ra)
    dfd = pl.DataFrame({"close": asset, "bench": bench})
    db = dfd.select(quant.downside_beta("close", "bench", window=250).alias("v"))[
        "v"
    ].drop_nulls()
    assert db.median() == pytest.approx(2.0, abs=0.1)


def test_downside_beta_null_without_down_bars():
    # A strictly increasing benchmark has no down-bars, so every window fails
    # the "at least two down-benchmark observations" guard -> all null.
    close = np.linspace(100, 200, 300) + np.sin(np.arange(300))
    bench = np.linspace(100, 300, 300)  # monotone up: no negative bench returns
    dfd = pl.DataFrame({"close": close, "bench": bench})
    out = dfd.select(quant.downside_beta("close", "bench", window=60).alias("v"))["v"]
    assert out.drop_nulls().len() == 0


def test_rolling_cvar_null_when_too_few_finite_returns():
    # A window with some finite returns but fewer than k = ceil(alpha*window)
    # of them can't resolve the tail and stays null (the len(w) < k guard).
    window, alpha = 100, 0.5  # k = 50; keep < 50 finite returns in each window
    close = np.full(200, np.nan)
    close[:30] = np.linspace(100, 110, 30)  # only ~29 finite returns ever
    dfc = pl.DataFrame({"close": close})
    out = dfc.select(
        quant.rolling_cvar("close", window=window, alpha=alpha).alias("v")
    )["v"]
    assert out.drop_nulls().len() == 0


def test_norm_ppf_extreme_tails():
    # Exercise both far-tail branches of the Acklam approximation.
    assert quant._norm_ppf(0.001) == pytest.approx(-3.090232, abs=1e-4)
    assert quant._norm_ppf(0.999) == pytest.approx(3.090232, abs=1e-4)


def test_rolling_cvar_null_when_window_all_nan():
    # A window whose returns are all NaN (leading price gaps) can't populate the
    # tail and must stay null rather than raising.
    close = np.concatenate([[0.0, 0.0, 0.0], np.linspace(100, 120, 197)])
    dfc = pl.DataFrame({"close": close})
    out = dfc.select(quant.rolling_cvar("close", window=100, alpha=0.05).alias("v"))[
        "v"
    ]
    # No crash, and the leading rows (before enough finite returns) are null.
    assert out.slice(0, 5).null_count() == 5
