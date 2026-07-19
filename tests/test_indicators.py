import numpy as np
import polars as pl
import pytest

from polars_ta import momentum, others, quant, trend, volatility, volume
from polars_ta.utils import DataCleaner


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.uniform(1e4, 1e6, n)
    return pl.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_rsi_bounds(df):
    out = df.select(momentum.rsi("close").alias("rsi")).drop_nulls()
    assert out["rsi"].min() >= 0
    assert out["rsi"].max() <= 100


def test_stoch_bounds(df):
    out = df.select(momentum.stoch("high", "low", "close").alias("k")).drop_nulls()
    assert out["k"].min() >= 0
    assert out["k"].max() <= 100


def test_williams_r_bounds(df):
    out = df.select(
        momentum.williams_r("high", "low", "close").alias("wr")
    ).drop_nulls()
    assert out["wr"].min() >= -100
    assert out["wr"].max() <= 0


def test_macd_runs(df):
    out = df.select(
        trend.macd("close").alias("macd"),
        trend.macd_signal("close").alias("sig"),
        trend.macd_diff("close").alias("hist"),
    ).drop_nulls()
    assert len(out) > 0
    assert np.allclose(out["hist"], out["macd"] - out["sig"], atol=1e-9)


def test_sma_ema(df):
    out = df.select(
        trend.sma_indicator("close", 20).alias("sma"),
        trend.ema_indicator("close", 20).alias("ema"),
    ).drop_nulls()
    assert abs(out["sma"].mean() - df["close"][19:].mean()) < 5


def test_bollinger_ordering(df):
    out = df.select(
        volatility.bollinger_hband("close").alias("h"),
        volatility.bollinger_mavg("close").alias("m"),
        volatility.bollinger_lband("close").alias("l"),
    ).drop_nulls()
    assert (out["h"] >= out["m"]).all()
    assert (out["m"] >= out["l"]).all()


def test_atr_positive(df):
    out = df.select(
        volatility.average_true_range("high", "low", "close").alias("atr")
    ).drop_nulls()
    assert (out["atr"] > 0).all()


def test_donchian_ordering(df):
    out = df.select(
        volatility.donchian_channel_hband("high", "low", "close").alias("h"),
        volatility.donchian_channel_lband("high", "low", "close").alias("l"),
    ).drop_nulls()
    assert (out["h"] >= out["l"]).all()


def test_obv_and_adi_run(df):
    out = df.select(
        volume.on_balance_volume("close", "volume").alias("obv"),
        volume.acc_dist_index("high", "low", "close", "volume").alias("adi"),
        volume.money_flow_index("high", "low", "close", "volume").alias("mfi"),
    )
    assert out["obv"].null_count() == 0
    mfi = out["mfi"].drop_nulls()
    assert mfi.min() >= 0 and mfi.max() <= 100


def test_vwap(df):
    out = df.select(
        volume.volume_weighted_average_price("high", "low", "close", "volume").alias(
            "vwap"
        )
    ).drop_nulls()
    assert out["vwap"].min() > df["low"].min() - 1
    assert out["vwap"].max() < df["high"].max() + 1


def test_quant_features(df):
    out = df.select(
        quant.garman_klass_volatility("open", "high", "low", "close").alias("gk"),
        quant.rolling_z_score("close").alias("z"),
        quant.rolling_sharpe_ratio("close").alias("sharpe"),
        quant.historical_volatility("close").alias("hv"),
    ).drop_nulls()
    assert len(out) > 0
    assert (out["gk"] >= 0).all()
    assert (out["hv"] >= 0).all()


def test_returns(df):
    out = df.select(
        others.daily_return("close").alias("dr"),
        others.daily_log_return("close").alias("dlr"),
        others.cumulative_return("close").alias("cr"),
    )
    dr = out["dr"].drop_nulls()
    dlr = out["dlr"].drop_nulls()
    assert np.allclose(np.log1p(dr / 100) * 100, dlr, atol=1e-9)
    expected = (df["close"][-1] / df["close"][0] - 1) * 100
    assert abs(out["cr"][-1] - expected) < 1e-9


def test_fillna_removes_nulls(df):
    out = df.select(momentum.rsi("close", fillna=True).alias("rsi"))
    assert out["rsi"].null_count() == 0


def test_lazyframe_support(df):
    out = df.lazy().with_columns(momentum.rsi("close").alias("rsi")).collect()
    assert "rsi" in out.columns


def test_streaming_engine_matches_default(df):
    lf = df.lazy().with_columns(
        momentum.rsi("close").alias("rsi"),
        trend.macd("close").alias("macd"),
        volatility.average_true_range("high", "low", "close").alias("atr"),
        volume.on_balance_volume("close", "volume").alias("obv"),
    )
    default = lf.collect()
    streamed = lf.collect(engine="streaming")
    assert default.equals(streamed)


def test_data_cleaner():
    bad = pl.DataFrame({"a": [1.0, float("nan"), 3.0, float("inf"), 5.0]})
    assert len(DataCleaner.dropna(bad)) == 3
    assert DataCleaner.get_invalid_indices(bad) == [1, 3]
    healed = DataCleaner.approximate_invalid_values(bad)
    assert healed["a"].is_nan().sum() == 0
    assert healed["a"][1] == 2.0
