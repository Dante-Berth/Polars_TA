"""The ``pl.col(...).ta.<indicator>()`` expression namespace.

Two guarantees are tested here:

1. **Parity** — every namespace method returns *byte-identical* output to the
   free function it wraps, for a representative call across every input
   signature (close-only, HLC, close+volume, OHLC, benchmark-based).
2. **Coverage** — the namespace exposes every eligible public indicator (so a
   newly added indicator is caught if it silently fails to register), and each
   registered method actually accepts the bound expression as its first
   argument.
"""

import numpy as np
import polars as pl
import pytest

import polars_ta
from polars_ta import microstructure as ms
from polars_ta import momentum, quant, trend, volatility, volume
from polars_ta.namespace import (
    _INDICATORS,
    _accepts_expr_first_arg,
    _collect_indicators,
)

N = 300


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, N))
    return pl.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + rng.uniform(0.1, 1.0, N),
            "low": close - rng.uniform(0.1, 1.0, N),
            "close": close,
            "volume": rng.uniform(1e4, 1e5, N),
            "benchmark": 100 + np.cumsum(rng.normal(0, 1, N)),
        }
    )


# (namespace expression builder, equivalent free-function expression). The
# calling expr is bound to each indicator's first positional argument.
PARITY_CASES = {
    "rsi": (
        lambda: pl.col("close").ta.rsi(14),
        lambda: momentum.rsi("close", 14),
    ),
    "macd": (
        lambda: pl.col("close").ta.macd(),
        lambda: trend.macd("close"),
    ),
    "ema": (
        lambda: pl.col("close").ta.ema_indicator(12),
        lambda: trend.ema_indicator("close", 12),
    ),
    "atr": (
        lambda: pl.col("high").ta.average_true_range("low", "close"),
        lambda: volatility.average_true_range("high", "low", "close"),
    ),
    "bollinger_hband": (
        lambda: pl.col("close").ta.bollinger_hband(),
        lambda: volatility.bollinger_hband("close"),
    ),
    "stoch": (
        lambda: pl.col("high").ta.stoch("low", "close"),
        lambda: momentum.stoch("high", "low", "close"),
    ),
    "obv": (
        lambda: pl.col("close").ta.on_balance_volume("volume"),
        lambda: volume.on_balance_volume("close", "volume"),
    ),
    "mfi": (
        lambda: pl.col("high").ta.money_flow_index("low", "close", "volume"),
        lambda: volume.money_flow_index("high", "low", "close", "volume"),
    ),
    "yang_zhang": (
        lambda: pl.col("open").ta.yang_zhang_volatility("high", "low", "close"),
        lambda: quant.yang_zhang_volatility("open", "high", "low", "close"),
    ),
    "kyle_lambda": (
        lambda: pl.col("close").ta.kyle_lambda("volume", 20),
        lambda: ms.kyle_lambda("close", "volume", 20),
    ),
    "roll_spread": (
        lambda: pl.col("close").ta.roll_spread(20),
        lambda: ms.roll_spread("close", 20),
    ),
    "rolling_cvar": (
        lambda: pl.col("close").ta.rolling_cvar(60),
        lambda: quant.rolling_cvar("close", 60),
    ),
    "rolling_beta_to": (
        lambda: pl.col("close").ta.rolling_beta_to("benchmark", 40),
        lambda: quant.rolling_beta_to("close", "benchmark", 40),
    ),
}


@pytest.mark.parametrize("name", sorted(PARITY_CASES))
def test_namespace_matches_free_function(df, name):
    ns_build, free_build = PARITY_CASES[name]
    a = df.select(ns_build().alias("v"))["v"].to_numpy().astype(float)
    b = df.select(free_build().alias("v"))["v"].to_numpy().astype(float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: null pattern differs"
    mask = ~np.isnan(a)
    # Byte-identical: the namespace is a pure dispatch layer, not a reimplement.
    assert np.array_equal(a[mask], b[mask]), f"{name}: values differ"


def test_namespace_accepts_col_name_argument(df):
    # The bound expression can itself be pl.col("close"); passing a plain
    # column name for the *other* inputs is the common path and must work.
    out = df.with_columns(pl.col("close").ta.rsi(14).alias("rsi"))
    assert out["rsi"].drop_nulls().len() > 0


def test_namespace_works_over_symbol():
    # A namespace call composes with .over("symbol") exactly like a free
    # function — no state leaks across groups.
    frames = []
    for sym, seed in (("A", 1), ("B", 2)):
        rg = np.random.default_rng(seed)
        c = 100 + np.cumsum(rg.normal(0, 1, 150))
        frames.append(pl.DataFrame({"close": c}).with_columns(pl.lit(sym).alias("s")))
    combined = pl.concat(frames)

    grouped = combined.with_columns(pl.col("close").ta.rsi(14).over("s").alias("r"))[
        "r"
    ].to_numpy()
    separate = pl.concat(
        [f.select(momentum.rsi("close", 14).alias("r")) for f in (frames[0], frames[1])]
    )["r"].to_numpy()
    both_nan = np.isnan(grouped) & np.isnan(separate)
    assert np.array_equal(np.isnan(grouped), np.isnan(separate))
    assert np.array_equal(grouped[~both_nan], separate[~both_nan])


def test_namespace_exposes_expected_indicators():
    # Spot-check that indicators from every module registered.
    for name in (
        "rsi",
        "macd",
        "average_true_range",
        "on_balance_volume",
        "yang_zhang_volatility",
        "roll_spread",
        "daily_return",
    ):
        assert hasattr(pl.col("close").ta, name), f"{name} missing from namespace"
    # The public tuple mirrors the registered set.
    assert set(polars_ta.TA_INDICATORS) == set(_INDICATORS)
    assert len(polars_ta.TA_INDICATORS) == len(_INDICATORS)


def test_namespace_excludes_cross_sectional_and_regime():
    # These take a generic value column / two signal expressions, not a price
    # series, so they are intentionally not on the namespace.
    for name in (
        "cross_sectional_zscore",
        "cross_sectional_rank",
        "regime_conditional_signal",
    ):
        assert not hasattr(pl.col("x").ta, name), f"{name} should not be exposed"


def test_every_registered_indicator_accepts_expr_first_arg():
    # The whole library now honors the documented str-or-Expr convention, so
    # every collected indicator must accept a bound expression as arg 1.
    rejected = [n for n, f in _INDICATORS.items() if not _accepts_expr_first_arg(f)]
    assert rejected == [], f"indicators rejecting an Expr first arg: {rejected}"


def test_accepts_expr_probe_rejects_unguarded_function():
    # A function that coerces its first arg with an unconditional pl.col()
    # (the exact anti-pattern the library was fixed to remove) must be flagged
    # by the probe, so a future regression is caught.
    def unguarded(close, window=14):
        return pl.col(close).rolling_mean(window_size=window)  # rejects an Expr

    def guarded(close, window=14):
        expr = pl.col(close) if isinstance(close, str) else close
        return expr.rolling_mean(window_size=window)

    assert _accepts_expr_first_arg(unguarded) is False
    assert _accepts_expr_first_arg(guarded) is True


def test_collect_indicators_is_deterministic():
    # Re-collecting yields the same mapping (guards against import-order flakiness).
    assert set(_collect_indicators()) == set(_INDICATORS)


def test_namespace_method_preserves_docstring():
    # Generated methods carry the free function's docstring for help()/tooling.
    assert TAExpr_rsi_doc() == momentum.rsi.__doc__


def TAExpr_rsi_doc():
    method = pl.col("close").ta.rsi
    return method.__doc__
