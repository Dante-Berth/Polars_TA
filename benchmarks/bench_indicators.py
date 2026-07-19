"""Benchmark indicator throughput on large OHLCV frames.

Run with: uv run python benchmarks/bench_indicators.py
"""

import time

import numpy as np
import polars as pl

from polars_ta import momentum, trend, volatility, volume


def make_ohlcv(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pl.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + rng.uniform(0.1, 1.5, n),
            "low": close - rng.uniform(0.1, 1.5, n),
            "close": close,
            "volume": rng.uniform(1e4, 1e6, n),
        }
    )


def run_all(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        momentum.rsi("close").alias("rsi_14"),
        momentum.stoch("high", "low", "close").alias("stoch_k"),
        momentum.stochrsi("close").alias("stochrsi"),
        trend.macd("close").alias("macd"),
        trend.adx("high", "low", "close").alias("adx"),
        trend.cci("high", "low", "close").alias("cci"),
        volatility.average_true_range("high", "low", "close").alias("atr_14"),
        volatility.bollinger_hband("close").alias("bb_high"),
        volatility.keltner_channel_hband("high", "low", "close").alias("kc_high"),
        volume.on_balance_volume("close", "volume").alias("obv"),
        volume.money_flow_index("high", "low", "close", "volume").alias("mfi"),
        volume.chaikin_money_flow("high", "low", "close", "volume").alias("cmf"),
    )


def time_it(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    sizes = [10_000, 100_000, 1_000_000]
    print(
        f"{'rows':>10} | {'eager (s)':>10} | {'lazy (s)':>10} | {'streaming (s)':>13}"
    )
    print("-" * 55)
    for n in sizes:
        df = make_ohlcv(n)

        eager_time = time_it(
            lambda df=df: run_all(df.lazy()).collect(engine="in-memory")
        )
        lazy_time = time_it(lambda df=df: run_all(df.lazy()).collect())
        stream_time = time_it(
            lambda df=df: run_all(df.lazy()).collect(engine="streaming")
        )

        print(
            f"{n:>10} | {eager_time:>10.4f} | {lazy_time:>10.4f} | {stream_time:>13.4f}"
        )


if __name__ == "__main__":
    main()
