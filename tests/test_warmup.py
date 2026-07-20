"""Warm-up / null policy audit.

Library-wide convention: an indicator that needs `k` bars of history surfaces
"not enough data yet" as **null** for exactly the first `k-1` rows — never a
fabricated number (0, the close price, ...) and never NaN/inf — and reports a
value on every row after its warm-up (on clean, well-behaved data).

Each entry pins the exact first-valid row index so a regression in either
direction (leaking fabricated warm-up values, or masking too much) fails
loudly.
"""

import numpy as np
import polars as pl
import pytest

from polars_ta import microstructure as ms
from polars_ta import momentum, quant, trend, volatility, volume

N = 400


def _make_df() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, N))
    bench = 100 + np.cumsum(rng.normal(0, 1, N))
    return pl.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + rng.uniform(0.1, 1.0, N),
            "low": close - rng.uniform(0.1, 1.0, N),
            "close": close,
            "volume": rng.uniform(1e4, 1e5, N),
            "bench": bench,
        }
    )


DF = _make_df()

# (expression, expected first-valid row index)
CASES = {
    "sma_12": (trend.sma_indicator("close", window=12), 11),
    "ema_12": (trend.ema_indicator("close", window=12), 11),
    "macd_26": (trend.macd("close"), 25),
    "adx_14": (trend.adx("high", "low", "close"), 13),
    "adx_pos_14": (trend.adx_pos("high", "low", "close"), 13),
    "cci_20": (trend.cci("high", "low", "close"), 19),
    "rsi_14": (momentum.rsi("close"), 13),
    "stoch_14": (momentum.stoch("high", "low", "close"), 13),
    "williams_r_14": (momentum.williams_r("high", "low", "close"), 13),
    "roc_12": (momentum.roc("close"), 12),
    "kama_10": (momentum.kama("close"), 10),
    "atr_14": (volatility.average_true_range("high", "low", "close"), 13),
    "bollinger_hband_20": (volatility.bollinger_hband("close"), 19),
    "obv": (volume.on_balance_volume("close", "volume"), 0),
    "mfi_14": (volume.money_flow_index("high", "low", "close", "volume"), 13),
    "kyle_lambda_20": (ms.kyle_lambda("close", "volume", window=20), 39),
    "hurst_100": (ms.hurst_exponent("close"), 99),
    "hist_vol_20": (quant.historical_volatility("close"), 21),
    "hull_moving_average_9": (trend.hull_moving_average("close", window=9), 10),
    "supertrend_10": (trend.supertrend("high", "low", "close", window=10), 9),
    "elder_bull_power_13": (
        trend.elder_bull_power("high", "low", "close", window=13),
        12,
    ),
    "cmo_14": (momentum.cmo("close", window=14), 13),
    "fisher_transform_9": (momentum.fisher_transform("high", "low", window=9), 8),
    "kvo": (volume.klinger_volume_oscillator("high", "low", "close", "volume"), 54),
    "ewma_volatility_21": (quant.ewma_volatility("close", window=21), 21),
    "lee_ready_trade_sign": (ms.lee_ready_trade_sign("close"), 0),
    "shannon_entropy_50": (ms.shannon_entropy("close", window=50, n_bins=10), 49),
    "approx_entropy_30": (ms.approximate_entropy("close", window=30), 29),
    "max_drawdown_100": (quant.rolling_max_drawdown("close", window=100), 198),
    "skew_60": (quant.rolling_skew("close", window=60), 60),
    "kurtosis_60": (quant.rolling_kurtosis("close", window=60), 60),
    "cf_var_5_100": (quant.cornish_fisher_var("close", window=100), 100),
    "frac_diff_04_100": (quant.frac_diff("close", d=0.4, window=100), 99),
    "momentum_12_1": (quant.momentum_12_1("close", lookback=252, skip=21), 252),
    "beta_60": (quant.rolling_beta_to("close", "bench", window=60), 119),
    "idio_vol_60": (quant.idiosyncratic_vol("close", "bench", window=60), 119),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_warmup_is_null_then_always_valid(name):
    expr, first_valid = CASES[name]
    out = DF.select(expr.alias("v"))["v"].to_numpy()
    isnan = np.isnan(out)

    assert isnan[:first_valid].all(), (
        f"{name}: fabricated value inside the warm-up window "
        f"(expected rows 0..{first_valid - 1} null)"
    )
    assert not isnan[first_valid:].any(), (
        f"{name}: unexpected null/NaN after warm-up "
        f"(first at row {first_valid + int(np.argmax(isnan[first_valid:]))})"
    )
    assert np.isfinite(out[first_valid:]).all(), f"{name}: inf after warm-up"
