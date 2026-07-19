"""Tests for polars_ta.quant, run against a real BTCUSDT 5m sample.

tests/fixtures/btcusdt_5m_sample.arrow is a 5000-row slice of real Binance
BTCUSDT 5-minute OHLCV data (Dec 2025), used instead of synthetic data so
these professional-desk features are validated against real market
microstructure rather than idealized noise.
"""

from pathlib import Path

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


def test_quant_features_streaming_equivalence(df):
    lf = df.lazy().with_columns(
        quant.yang_zhang_volatility("open", "high", "low", "close").alias("yz"),
        quant.relative_volume("volume").alias("rvol"),
        **quant.hurst_ribbon("close"),
    )
    default = lf.collect()
    streamed = lf.collect(engine="streaming")
    assert default.equals(streamed)
