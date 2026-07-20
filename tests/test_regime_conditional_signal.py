"""Tests for quant.regime_conditional_signal — a compositional building
block (regime score + two signal expressions -> one switched signal), not a
single-input indicator, so it doesn't fit the price-indicator smoke/warmup/
multi-asset suites and gets its own file."""

import numpy as np
import polars as pl
import pytest

from polars_ta import quant, trend, volatility


def test_switches_at_threshold():
    df = pl.DataFrame(
        {"regime": [0.3, 0.5, 0.7], "above": [1.0, 1.0, 1.0], "below": [0.0, 0.0, 0.0]}
    )
    out = df.select(
        quant.regime_conditional_signal("regime", 0.5, "above", "below").alias("v")
    )["v"].to_list()
    # >= threshold (default above_or_equal=True) picks "above".
    assert out == [0.0, 1.0, 1.0]


def test_above_or_equal_false_excludes_the_boundary():
    df = pl.DataFrame({"regime": [0.5], "above": [1.0], "below": [0.0]})
    out = df.select(
        quant.regime_conditional_signal(
            "regime", 0.5, "above", "below", above_or_equal=False
        ).alias("v")
    )["v"].to_list()
    assert out == [0.0]


def test_null_regime_yields_null_output_not_a_fallback():
    df = pl.DataFrame(
        {
            "regime": [None, 0.6, 0.3],
            "above": [1.0, 2.0, 3.0],
            "below": [10.0, 20.0, 30.0],
        }
    )
    out = df.select(
        quant.regime_conditional_signal("regime", 0.5, "above", "below").alias("v")
    )["v"].to_list()
    assert out == [None, 2.0, 30.0]


def test_accepts_arbitrary_expressions_not_just_column_names():
    df = pl.DataFrame({"close": [100.0, 101.0, 99.0, 102.0]})
    regime = pl.col("close").diff(1).fill_null(0.0)  # arbitrary regime expr
    above = pl.lit(1.0)
    below = pl.lit(-1.0)
    out = df.select(
        quant.regime_conditional_signal(regime, 0.0, above, below).alias("v")
    )["v"].to_list()
    # diff: [0, 1, -2, 3] -> >=0 picks above (1.0), <0 picks below (-1.0)
    assert out == [1.0, 1.0, -1.0, 1.0]


def test_wired_to_hurst_ribbon_and_two_real_indicators():
    # The documented use case: trend-follow when H > 0.5, mean-revert
    # otherwise, arbitrated by hurst_ribbon.
    rng = np.random.default_rng(7)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pl.DataFrame({"close": close})

    out = df.with_columns(**quant.hurst_ribbon("close", scales=(16, 32, 64)))
    trend_sig = trend.ema_indicator("close", window=10) - trend.ema_indicator(
        "close", window=30
    )
    reversion_sig = volatility.bollinger_pband("close") - 0.5

    out = out.with_columns(
        quant.regime_conditional_signal(
            "h_ribbon_avg", 0.5, trend_sig, reversion_sig
        ).alias("composite")
    )

    out = out.with_columns(
        trend_sig.alias("trend_sig"), reversion_sig.alias("reversion_sig")
    )
    # Only compare rows where regime, trend_sig, and reversion_sig are all
    # defined — each has its own warm-up window.
    valid = out.filter(
        pl.col("h_ribbon_avg").is_not_null()
        & pl.col("trend_sig").is_not_null()
        & pl.col("reversion_sig").is_not_null()
    )
    assert len(valid) > 0
    expected = np.where(
        valid["h_ribbon_avg"].to_numpy() >= 0.5,
        valid["trend_sig"].to_numpy(),
        valid["reversion_sig"].to_numpy(),
    )
    actual = valid["composite"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_multi_asset_over_symbol():
    def mk(n, seed):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        return pl.DataFrame({"close": close})

    a, b = mk(200, 1), mk(250, 2)
    both = pl.concat(
        [
            a.with_columns(pl.lit("A").alias("sym")),
            b.with_columns(pl.lit("B").alias("sym")),
        ]
    )

    def build() -> pl.Expr:
        regime = quant.hurst_ribbon("close", scales=(16, 32, 64))["h_ribbon_avg"]
        return quant.regime_conditional_signal(
            regime, 0.5, pl.col("close").diff(5), -pl.col("close").diff(5)
        )

    grouped = both.with_columns(build().over("sym").alias("v"))["v"].to_numpy()
    separate = pl.concat([a.select(build().alias("v")), b.select(build().alias("v"))])[
        "v"
    ].to_numpy()

    both_nan = np.isnan(grouped) & np.isnan(separate)
    assert np.array_equal(np.isnan(grouped), np.isnan(separate))
    diff = np.where(both_nan, 0.0, np.abs(grouped - separate))
    assert np.nanmax(diff, initial=0.0) < 1e-10


@pytest.mark.parametrize("above_or_equal", [True, False])
def test_length_preserved(above_or_equal):
    df = pl.DataFrame(
        {"regime": [0.1, 0.9, None, 0.5], "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
    )
    out = df.select(
        quant.regime_conditional_signal(
            "regime", 0.5, "a", "b", above_or_equal=above_or_equal
        ).alias("v")
    )["v"]
    assert len(out) == 4
