"""Multi-asset correctness: every indicator must be usable per-symbol via
``.over("symbol")`` and produce exactly the same values as computing each
symbol's frame separately — no state leaking across symbol boundaries.

This is the core promise of an expression-based TA library: one expression,
grouped execution. An indicator that silently mixes symbols (e.g. the first
bars of symbol B seeing symbol A's tail) invalidates every cross-sectional
backtest built on it.
"""

import numpy as np
import polars as pl
import pytest

from polars_ta import microstructure as ms
from polars_ta import momentum, quant, trend, volatility, volume


def _make_ohlcv(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pl.DataFrame(
        {
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "volume": rng.uniform(1e4, 1e5, n),
        }
    )


FRAME_A = _make_ohlcv(200, seed=1)
FRAME_B = _make_ohlcv(250, seed=2)
COMBINED = pl.concat(
    [
        FRAME_A.with_columns(pl.lit("A").alias("symbol")),
        FRAME_B.with_columns(pl.lit("B").alias("symbol")),
    ]
)

# One representative expression per computation style: pure rolling, ewm
# (infinite memory), cumulative, rolling_map, and the map_batches-based
# sequential kernels (kama, psar, vpin, hurst) — the ones most likely to
# ignore group boundaries.
INDICATORS = {
    "rsi": momentum.rsi("close"),
    "stoch": momentum.stoch("high", "low", "close"),
    "kama": momentum.kama("close"),
    "macd": trend.macd("close"),
    "adx": trend.adx("high", "low", "close"),
    "cci": trend.cci("high", "low", "close"),
    "psar": trend.psar("high", "low", "close"),
    "atr": volatility.average_true_range("high", "low", "close"),
    "bollinger_hband": volatility.bollinger_hband("close"),
    "obv": volume.on_balance_volume("close", "volume"),
    "mfi": volume.money_flow_index("high", "low", "close", "volume"),
    "kyle_lambda": ms.kyle_lambda("close", "volume", window=20),
    "roll_spread": ms.roll_spread("close", window=20),
    "vpin": ms.vpin("close", "volume", bucket_size=50_000),
    "hurst": ms.hurst_exponent("close"),
    "hull_moving_average": trend.hull_moving_average("close"),
    "supertrend": trend.supertrend("high", "low", "close"),
    "elder_bull_power": trend.elder_bull_power("high", "low", "close"),
    "cmo": momentum.cmo("close"),
    "fisher_transform": momentum.fisher_transform("high", "low"),
    "kvo": volume.klinger_volume_oscillator("high", "low", "close", "volume"),
    "ewma_volatility": quant.ewma_volatility("close"),
    "lee_ready_trade_sign": ms.lee_ready_trade_sign("close"),
    "shannon_entropy": ms.shannon_entropy("close", window=50, n_bins=10),
    "approximate_entropy": ms.approximate_entropy("close", window=30),
}


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_over_symbol_matches_per_symbol(name):
    expr = INDICATORS[name]
    grouped = COMBINED.with_columns(expr.over("symbol").alias("v"))["v"].to_numpy()
    separate = pl.concat(
        [FRAME_A.select(expr.alias("v")), FRAME_B.select(expr.alias("v"))]
    )["v"].to_numpy()

    both_nan = np.isnan(grouped) & np.isnan(separate)
    assert np.array_equal(np.isnan(grouped), np.isnan(separate)), (
        f"{name}: null pattern differs under .over('symbol') — state leaks "
        "across symbols"
    )
    diff = np.where(both_nan, 0.0, np.abs(grouped - separate))
    assert np.nanmax(diff, initial=0.0) < 1e-10, (
        f"{name}: values differ under .over('symbol')"
    )
