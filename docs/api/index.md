# API reference

Every indicator is a plain `pl.Expr`. Call it as a free function
(`momentum.rsi("close", 14)`) or as an expression method
(`pl.col("close").ta.rsi(14)`) — same code, [pick whichever reads
better](../how_to_guides.md#call-indicators-as-plcoltaname).

The reference is split by what each family *measures*, so you can jump to the
question you're actually asking:

| If you want to know… | Look in |
|---|---|
| Is price accelerating or losing steam? (overbought/oversold, divergence) | [Momentum](momentum.md) |
| Which way is the trend, and how strong? | [Trend](trend.md) |
| How much is price moving — and is that expanding or contracting? | [Volatility](volatility.md) |
| Is volume confirming the move? Where is money flowing? | [Volume](volume.md) |
| Simple period / cumulative returns | [Returns](returns.md) |
| Risk sizing, drawdown, tail risk, factor exposure, regime detection | [Quant](quant.md) |
| Liquidity, order-flow toxicity, spread, mean-reversion speed | [Microstructure](microstructure.md) |
| Time-of-day / day-of-week / session effects | [Calendar](calendar.md) |
| Calling indicators as `pl.col(...).ta.<name>()` | [Expression namespace](namespace.md) |
| Building your own indicator, cleaning dirty data | [Utilities](utils.md) |

## Conventions that hold everywhere

- **Inputs** are a column name (`str`) *or* an existing `pl.Expr` — uniformly,
  across every indicator.
- **Warm-up is null, never faked.** An indicator needing `k` bars of history
  returns `null` for its first `k-1` rows, so a warm-up value never leaks into
  a backtest.
- **Per-symbol is free.** Append `.over("symbol")` on a multi-asset frame and
  every indicator computes independently per group, with no state crossing
  symbol boundaries.
- **`fillna: bool = False`** on every indicator forward-fills gaps when `True`.
- **Streaming-safe.** Every indicator produces identical output under
  `.collect(engine="streaming")`.
