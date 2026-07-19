import numpy as np
import polars as pl
import pytest

from polars_ta import microstructure as ms


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 2000
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    volume = rng.uniform(1e3, 1e4, n)
    return pl.DataFrame({"close": close, "volume": volume})


def test_roll_spread_nonnegative_or_null(df):
    out = df.select(ms.roll_spread("close").alias("v"))["v"].drop_nulls()
    assert len(out) > 0
    assert (out >= 0).all()


def test_kyle_lambda_runs(df):
    out = df.select(ms.kyle_lambda("close", "volume").alias("v"))["v"].drop_nulls()
    assert len(out) > 0
    assert out.is_finite().all()


def test_hasbrouck_lambda_runs(df):
    out = df.select(ms.hasbrouck_lambda("close", "volume").alias("v"))[
        "v"
    ].drop_nulls()
    assert len(out) > 0
    assert out.is_finite().all()


def test_effective_spread_nonnegative(df):
    out = df.select(ms.effective_spread("close").alias("v"))["v"].drop_nulls()
    assert (out >= 0).all()


def test_effective_spread_with_explicit_mid(df):
    out = df.select(
        ms.effective_spread("close", mid_price=pl.col("close") * 0.999).alias("v")
    )["v"]
    assert out.null_count() == 0


def test_variance_ratio_near_one_for_random_walk(df):
    out = df.select(ms.variance_ratio("close", window=200, lag=2).alias("v"))[
        "v"
    ].drop_nulls()
    assert len(out) > 0
    # a pure random walk should have VR centered near 1
    assert 0.5 < out.mean() < 2.0


def test_hurst_ordering_across_regimes():
    rng = np.random.default_rng(3)
    n = 3000
    noise = rng.normal(0, 1, n)

    mr_ret = np.zeros(n)
    tr_ret = np.zeros(n)
    for i in range(1, n):
        mr_ret[i] = -0.3 * mr_ret[i - 1] + noise[i]
        tr_ret[i] = 0.3 * tr_ret[i - 1] + noise[i]

    close_mr = 100 * np.exp(np.cumsum(mr_ret) * 0.01)
    close_tr = 100 * np.exp(np.cumsum(tr_ret) * 0.01)

    h_mr = (
        pl.DataFrame({"close": close_mr})
        .select(ms.hurst_exponent("close", window=300).alias("h"))["h"]
        .drop_nulls()
        .mean()
    )
    h_tr = (
        pl.DataFrame({"close": close_tr})
        .select(ms.hurst_exponent("close", window=300).alias("h"))["h"]
        .drop_nulls()
        .mean()
    )
    assert h_mr < h_tr


def test_corwin_schultz_nonnegative():
    rng = np.random.default_rng(7)
    n = 1000
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    df = pl.DataFrame({"high": high, "low": low})
    out = df.select(ms.corwin_schultz_spread("high", "low").alias("v"))[
        "v"
    ].drop_nulls()
    assert len(out) > 0
    assert (out >= 0).all()


def test_half_life_positive_for_mean_reverting():
    rng = np.random.default_rng(5)
    n = 3000
    p = np.zeros(n)
    for i in range(1, n):
        # Strongly mean-reverting OU around 100.
        p[i] = p[i - 1] + 0.2 * (100 - p[i - 1]) + rng.normal(0, 0.5)
    out = (
        pl.DataFrame({"close": p})
        .select(ms.half_life("close", window=120).alias("v"))["v"]
        .drop_nulls()
    )
    assert len(out) > 0
    assert (out > 0).all()


def test_vpin_bounded(df):
    out = df.select(
        ms.vpin("close", "volume", bucket_size=50_000, window=10).alias("v")
    )["v"].drop_nulls()
    assert len(out) > 0
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_vpin_lazyframe_and_streaming(df):
    lf = df.lazy().with_columns(
        ms.vpin("close", "volume", bucket_size=50_000, window=10).alias("vpin")
    )
    default = lf.collect()
    streamed = lf.collect(engine="streaming")
    assert default.equals(streamed)
