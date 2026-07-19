# Polars TA

Technical analysis indicators built on [Polars](https://pola.rs) expressions instead of pandas.

Every indicator is a plain `pl.Expr`, so it composes naturally with `.with_columns(...)`, works on both `DataFrame` and `LazyFrame`, and runs on Polars' multithreaded, vectorized engine and its [streaming engine](https://docs.pola.rs/user-guide/lazy/streaming/) for larger-than-memory data — no row-by-row Python loops, aside from a couple of genuinely recursive indicators (KAMA, PSAR) that use `map_batches`.

## Why Polars TA

- **Lazy by default.** Indicators are expressions, not eagerly computed columns — nothing runs until you `.collect()`.
- **One dependency.** Just `polars` + `numpy`, no pandas in the critical path.
- **Streaming-ready.** Every indicator is validated to produce identical output under Polars' streaming engine, so it scales to datasets larger than memory.
- **Numerically checked.** A subset of indicators is cross-validated against independent NumPy reference implementations in CI.

## Where to go next

- New to the library? Start with [Getting started](getting_started.md).
- Want the reasoning behind design choices (expressions, `fillna`, streaming)? See [Concepts](concepts.md).
- Need a recipe for a specific task? See [How-to guides](how_to_guides.md).
- Want to see indicators combined on real-shaped data? See [Examples](examples.md).
- Looking for a specific function signature? See the [API reference](api.md).
