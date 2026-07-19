"""Exhaustive smoke coverage: every public indicator, both fillna modes.

These tests don't validate numerics (test_reference.py does) — they guarantee
that every indicator in the public API builds, evaluates, preserves length,
and never leaks NaN/inf, and that the ``fillna=True`` path leaves no gaps.
"""

import numpy as np
import polars as pl
import pytest

from polars_ta import microstructure as ms
from polars_ta import momentum, others, quant, trend, volatility, volume
from polars_ta.utils import BaseIndicator

N = 300


def _make_df() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 1, N))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, N)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, N)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1e4, 1e5, N),
        }
    )


DF = _make_df()

# fillna-aware indicators (ta-style API): name -> builder taking fillna kwarg
FILLNA_CASES = {
    # momentum
    "rsi": lambda f: momentum.rsi("close", fillna=f),
    "tsi": lambda f: momentum.tsi("close", fillna=f),
    "ultimate_oscillator": lambda f: momentum.ultimate_oscillator(
        "high", "low", "close", fillna=f
    ),
    "stoch": lambda f: momentum.stoch("high", "low", "close", fillna=f),
    "stoch_signal": lambda f: momentum.stoch_signal("high", "low", "close", fillna=f),
    "williams_r": lambda f: momentum.williams_r("high", "low", "close", fillna=f),
    "awesome_oscillator": lambda f: momentum.awesome_oscillator(
        "high", "low", fillna=f
    ),
    "kama": lambda f: momentum.kama("close", fillna=f),
    "roc": lambda f: momentum.roc("close", fillna=f),
    "stochrsi": lambda f: momentum.stochrsi("close", fillna=f),
    "stochrsi_k": lambda f: momentum.stochrsi_k("close", fillna=f),
    "stochrsi_d": lambda f: momentum.stochrsi_d("close", fillna=f),
    "ppo": lambda f: momentum.ppo("close", fillna=f),
    "ppo_signal": lambda f: momentum.ppo_signal("close", fillna=f),
    "ppo_hist": lambda f: momentum.ppo_hist("close", fillna=f),
    "pvo": lambda f: momentum.pvo("volume", fillna=f),
    "pvo_signal": lambda f: momentum.pvo_signal("volume", fillna=f),
    "pvo_hist": lambda f: momentum.pvo_hist("volume", fillna=f),
    # trend
    "ema_indicator": lambda f: trend.ema_indicator("close", fillna=f),
    "sma_indicator": lambda f: trend.sma_indicator("close", fillna=f),
    "wma_indicator": lambda f: trend.wma_indicator("close", fillna=f),
    "macd": lambda f: trend.macd("close", fillna=f),
    "macd_signal": lambda f: trend.macd_signal("close", fillna=f),
    "macd_diff": lambda f: trend.macd_diff("close", fillna=f),
    "adx": lambda f: trend.adx("high", "low", "close", fillna=f),
    "adx_pos": lambda f: trend.adx_pos("high", "low", "close", fillna=f),
    "adx_neg": lambda f: trend.adx_neg("high", "low", "close", fillna=f),
    "vortex_pos": lambda f: trend.vortex_indicator_pos(
        "high", "low", "close", fillna=f
    ),
    "vortex_neg": lambda f: trend.vortex_indicator_neg(
        "high", "low", "close", fillna=f
    ),
    "trix": lambda f: trend.trix("close", fillna=f),
    "mass_index": lambda f: trend.mass_index("high", "low", fillna=f),
    "cci": lambda f: trend.cci("high", "low", "close", fillna=f),
    "dpo": lambda f: trend.dpo("close", fillna=f),
    "kst": lambda f: trend.kst("close", fillna=f),
    "kst_sig": lambda f: trend.kst_sig("close", fillna=f),
    "stc": lambda f: trend.stc("close", fillna=f),
    "ichimoku_conversion_line": lambda f: trend.ichimoku_conversion_line(
        "high", "low", fillna=f
    ),
    "ichimoku_base_line": lambda f: trend.ichimoku_base_line("high", "low", fillna=f),
    "ichimoku_a": lambda f: trend.ichimoku_a("high", "low", fillna=f),
    "ichimoku_b": lambda f: trend.ichimoku_b("high", "low", fillna=f),
    "aroon_up": lambda f: trend.aroon_up("high", "low", fillna=f),
    "aroon_down": lambda f: trend.aroon_down("high", "low", fillna=f),
    "psar": lambda f: trend.psar("high", "low", "close", fillna=f),
    # volatility
    "atr": lambda f: volatility.average_true_range("high", "low", "close", fillna=f),
    "bollinger_mavg": lambda f: volatility.bollinger_mavg("close", fillna=f),
    "bollinger_hband": lambda f: volatility.bollinger_hband("close", fillna=f),
    "bollinger_lband": lambda f: volatility.bollinger_lband("close", fillna=f),
    "bollinger_wband": lambda f: volatility.bollinger_wband("close", fillna=f),
    "bollinger_pband": lambda f: volatility.bollinger_pband("close", fillna=f),
    "bollinger_hband_ind": lambda f: volatility.bollinger_hband_indicator(
        "close", fillna=f
    ),
    "bollinger_lband_ind": lambda f: volatility.bollinger_lband_indicator(
        "close", fillna=f
    ),
    "kc_mband": lambda f: volatility.keltner_channel_mband(
        "high", "low", "close", fillna=f
    ),
    "kc_hband": lambda f: volatility.keltner_channel_hband(
        "high", "low", "close", fillna=f
    ),
    "kc_lband": lambda f: volatility.keltner_channel_lband(
        "high", "low", "close", fillna=f
    ),
    "kc_wband": lambda f: volatility.keltner_channel_wband(
        "high", "low", "close", fillna=f
    ),
    "kc_pband": lambda f: volatility.keltner_channel_pband(
        "high", "low", "close", fillna=f
    ),
    "kc_hband_ind": lambda f: volatility.keltner_channel_hband_indicator(
        "high", "low", "close", fillna=f
    ),
    "kc_lband_ind": lambda f: volatility.keltner_channel_lband_indicator(
        "high", "low", "close", fillna=f
    ),
    "dc_hband": lambda f: volatility.donchian_channel_hband(
        "high", "low", "close", fillna=f
    ),
    "dc_lband": lambda f: volatility.donchian_channel_lband(
        "high", "low", "close", fillna=f
    ),
    "dc_mband": lambda f: volatility.donchian_channel_mband(
        "high", "low", "close", fillna=f
    ),
    "dc_wband": lambda f: volatility.donchian_channel_wband(
        "high", "low", "close", fillna=f
    ),
    "dc_pband": lambda f: volatility.donchian_channel_pband(
        "high", "low", "close", fillna=f
    ),
    "ulcer_index": lambda f: volatility.ulcer_index("close", fillna=f),
    # volume
    "acc_dist_index": lambda f: volume.acc_dist_index(
        "high", "low", "close", "volume", fillna=f
    ),
    "obv": lambda f: volume.on_balance_volume("close", "volume", fillna=f),
    "cmf": lambda f: volume.chaikin_money_flow(
        "high", "low", "close", "volume", fillna=f
    ),
    "force_index": lambda f: volume.force_index("close", "volume", fillna=f),
    "eom": lambda f: volume.ease_of_movement("high", "low", "volume", fillna=f),
    "sma_eom": lambda f: volume.sma_ease_of_movement("high", "low", "volume", fillna=f),
    "vpt": lambda f: volume.volume_price_trend("close", "volume", fillna=f),
    "nvi": lambda f: volume.negative_volume_index("close", "volume", fillna=f),
    "mfi": lambda f: volume.money_flow_index(
        "high", "low", "close", "volume", fillna=f
    ),
    "vwap": lambda f: volume.volume_weighted_average_price(
        "high", "low", "close", "volume", fillna=f
    ),
    # others
    "daily_return": lambda f: others.daily_return("close", fillna=f),
    "daily_log_return": lambda f: others.daily_log_return("close", fillna=f),
    "cumulative_return": lambda f: others.cumulative_return("close", fillna=f),
}

# quant / microstructure indicators (no fillna flag)
PLAIN_CASES = {
    "log_return": quant.log_return("close"),
    "garman_klass": quant.garman_klass_volatility("open", "high", "low", "close"),
    "rolling_z_score": quant.rolling_z_score("close"),
    "vol_adj_momentum": quant.vol_adjusted_momentum("close"),
    "micro_price_proxy": quant.micro_price_proxy("high", "low", "close", "volume"),
    "rolling_sharpe": quant.rolling_sharpe_ratio("close"),
    "rolling_sortino": quant.rolling_sortino_ratio("close"),
    "historical_vol": quant.historical_volatility("close"),
    "parkinson_vol": quant.parkinson_volatility("high", "low"),
    "rogers_satchell_vol": quant.rogers_satchell_volatility(
        "open", "high", "low", "close"
    ),
    "yang_zhang_vol": quant.yang_zhang_volatility("open", "high", "low", "close"),
    "relative_volume": quant.relative_volume("volume"),
    "volatility_z_score": quant.volatility_z_score("high", "low"),
    "amihud": quant.amihud_illiquidity("close", "volume"),
    "roll_spread": ms.roll_spread("close", window=20),
    "kyle_lambda": ms.kyle_lambda("close", "volume", window=20),
    "hasbrouck_lambda": ms.hasbrouck_lambda("close", "volume", window=20),
    "effective_spread": ms.effective_spread("close"),
    "vpin": ms.vpin("close", "volume", bucket_size=50_000),
    "hurst_exponent": ms.hurst_exponent("close"),
    "variance_ratio": ms.variance_ratio("close", window=40),
    "corwin_schultz": ms.corwin_schultz_spread("high", "low"),
    "half_life": ms.half_life("close", window=40),
    "rolling_beta": ms.rolling_beta("close", "open", window=20),
    "rolling_cov": ms.rolling_cov("close", "open", window=20),
}


def _check_basic(name: str, out: pl.Series) -> None:
    assert len(out) == N, f"{name}: length changed"
    vals = out.drop_nulls().to_numpy()
    assert len(vals) > 0, f"{name}: all-null output"
    assert np.isfinite(vals).all(), f"{name}: leaked NaN/inf"


@pytest.mark.parametrize("name", sorted(FILLNA_CASES))
def test_indicator_default(name):
    out = DF.select(FILLNA_CASES[name](False).alias("v"))["v"]
    _check_basic(name, out)


@pytest.mark.parametrize("name", sorted(FILLNA_CASES))
def test_indicator_fillna_leaves_no_gaps(name):
    out = DF.select(FILLNA_CASES[name](True).alias("v"))["v"]
    _check_basic(name, out)
    assert out.null_count() == 0, f"{name}: fillna=True left {out.null_count()} nulls"


@pytest.mark.parametrize("name", sorted(PLAIN_CASES))
def test_quant_and_microstructure(name):
    out = DF.select(PLAIN_CASES[name].alias("v"))["v"]
    _check_basic(name, out)


def test_quant_hurst_ribbon():
    exprs = quant.hurst_ribbon("close", scales=(64, 128))
    out = DF.select(*exprs.values())
    assert out.width == len(exprs)
    for col in out.columns:
        _check_basic(col, out[col])


def test_get_min_max():
    lo = DF.select(BaseIndicator.get_min_max("high", "low", "min").alias("v"))["v"]
    hi = DF.select(BaseIndicator.get_min_max("high", "low", "max").alias("v"))["v"]
    assert (lo.to_numpy() == DF["low"].to_numpy()).all()
    assert (hi.to_numpy() == DF["high"].to_numpy()).all()
    with pytest.raises(ValueError):
        BaseIndicator.get_min_max("high", "low", "median")


def test_check_fillna_bfill_mode():
    # value=-1 selects forward-fill + back-fill instead of a constant.
    s = pl.DataFrame({"x": [None, 1.0, None, 2.0]})
    out = s.select(BaseIndicator.check_fillna(pl.col("x"), True, value=-1).alias("v"))[
        "v"
    ]
    assert out.null_count() == 0
