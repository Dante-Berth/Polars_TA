"""Property-based tests (Hypothesis) covering invariants every indicator must
satisfy, regardless of the specific numbers.

Three families of property:

1. **Length preservation** — an indicator returns exactly one value per input
   row (it's a column, not an aggregation).
2. **Null propagation** — indicators surface "not enough data yet" as nulls,
   and never leak NaN/inf into the output.
3. **Causality / no-lookahead** — the value at bar ``t`` must not depend on any
   data after ``t``. This is the single most important property for a TA
   library: an indicator that peeks into the future silently invalidates every
   backtest built on it. We check it by mutating the tail of the series and
   asserting the untouched prefix of the output is unchanged.
"""

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from polars_ta import microstructure as ms
from polars_ta import momentum, trend, volatility, volume

# A representative indicator per input signature, as (name, builder) where the
# builder takes the relevant column names and returns a single pl.Expr.
CLOSE_ONLY = {
    "rsi": lambda: momentum.rsi("close", window=14),
    "sma": lambda: trend.sma_indicator("close", window=10),
    "macd": lambda: trend.macd("close"),
    "roll_spread": lambda: ms.roll_spread("close", window=20),
    "variance_ratio": lambda: ms.variance_ratio("close", window=20),
    "half_life": lambda: ms.half_life("close", window=30),
}
HLC = {
    "atr": lambda: volatility.average_true_range("high", "low", "close", window=14),
    "adx": lambda: trend.adx("high", "low", "close", window=14),
}
CLOSE_VOLUME = {
    "obv": lambda: volume.on_balance_volume("close", "volume"),
    "kyle": lambda: ms.kyle_lambda("close", "volume", window=20),
    "hasbrouck": lambda: ms.hasbrouck_lambda("close", "volume", window=20),
}


def _make_df(prices: list[float], n: int) -> pl.DataFrame:
    """Build a valid OHLCV frame from a positive price path."""
    close = np.asarray(prices[:n], dtype=float)
    high = close + np.abs(close) * 0.001 + 0.01
    low = close - np.abs(close) * 0.001 - 0.01
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.linspace(1_000, 5_000, len(close))
    return pl.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


# Strictly positive, bounded price paths so log-returns stay well-defined.
price_lists = st.lists(
    st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
    min_size=120,
    max_size=400,
)


def _all_builders():
    yield from CLOSE_ONLY.items()
    yield from HLC.items()
    yield from CLOSE_VOLUME.items()


@settings(max_examples=40, deadline=None)
@given(prices=price_lists)
def test_length_preserved_and_no_naninf(prices):
    df = _make_df(prices, len(prices))
    for name, build in _all_builders():
        out = df.select(build().alias("v"))["v"]
        assert len(out) == df.height, f"{name} changed length"
        finite = out.drop_nulls()
        if finite.dtype.is_numeric():
            arr = finite.to_numpy()
            assert np.isfinite(arr).all(), f"{name} leaked NaN/inf"


@settings(max_examples=40, deadline=None)
@given(prices=price_lists, cut=st.integers(min_value=1, max_value=30))
def test_causality_no_lookahead(prices, cut):
    """Mutating the last ``cut`` bars must not change the earlier output."""
    n = len(prices)
    if n - cut < 90:  # keep enough history for the longest window used here
        return
    df = _make_df(prices, n)

    # A second frame identical up to n-cut, with the tail perturbed.
    perturbed = list(prices)
    for i in range(n - cut, n):
        perturbed[i] = perturbed[i] * 1.5 + 1.0
    df2 = _make_df(perturbed, n)

    keep = n - cut
    for name, build in _all_builders():
        a = df.select(build().alias("v"))["v"].to_numpy()[:keep]
        b = df2.select(build().alias("v"))["v"].to_numpy()[:keep]
        # Compare treating NaN positions as equal.
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.array_equal(both_nan, np.isnan(a)), (
            f"{name} null pattern differs before the cut — lookahead"
        )
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        assert np.nanmax(diff, initial=0.0) < 1e-9, (
            f"{name} value changed before the cut — lookahead"
        )
