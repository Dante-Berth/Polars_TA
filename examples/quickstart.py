"""Quickstart: compute a full set of indicators on synthetic OHLCV data."""

import numpy as np
import polars as pl

from polars_ta import momentum, trend, volatility, volume, quant, others

# Build a synthetic OHLCV frame (swap in your own data here)
rng = np.random.default_rng(0)
n = 500
close = 100 + np.cumsum(rng.normal(0, 1, n))
df = pl.DataFrame(
    {
        "open": close + rng.normal(0, 0.5, n),
        "high": close + rng.uniform(0.1, 1.5, n),
        "low": close - rng.uniform(0.1, 1.5, n),
        "close": close,
        "volume": rng.uniform(1e4, 1e6, n),
    }
)

# All indicators are Polars expressions — compose them in one with_columns call
out = df.lazy().with_columns(
    momentum.rsi("close").alias("rsi_14"),
    momentum.stoch("high", "low", "close").alias("stoch_k"),
    trend.macd("close").alias("macd"),
    trend.macd_signal("close").alias("macd_signal"),
    trend.adx("high", "low", "close").alias("adx"),
    volatility.average_true_range("high", "low", "close").alias("atr_14"),
    volatility.bollinger_hband("close").alias("bb_high"),
    volatility.bollinger_lband("close").alias("bb_low"),
    volume.on_balance_volume("close", "volume").alias("obv"),
    volume.money_flow_index("high", "low", "close", "volume").alias("mfi"),
    quant.rolling_sharpe_ratio("close").alias("sharpe_63"),
    others.daily_return("close").alias("ret_pct"),
).collect()

print(out.tail(10))
