# Volatility

**Volatility indicators answer: how much is price moving — and is that
movement expanding or contracting?** They drive position sizing (risk more when
quiet, less when wild), adaptive stops, and breakout detection (a volatility
squeeze often precedes a large move).

## Range & true-range measures

- **`average_true_range`** — the
  canonical volatility unit. True range accounts for gaps, so ATR is the natural
  scale for stops and for normalising signals across instruments. **Start
  here.**
- **`ulcer_index`** — a *downside*-only
  volatility: depth and duration of drawdowns, closer to felt risk than
  symmetric measures.

## Bands & channels (mean ± a volatility envelope)

- **Bollinger Bands** —
  `mavg ± k·σ`. Use `bollinger_wband`
  to detect squeezes (narrow band → pending breakout) and
  `bollinger_pband` as a 0–1
  mean-reversion signal (where price sits in the band).
- **Keltner Channel** — like
  Bollinger but banded by **ATR** instead of standard deviation, so it reacts to
  gaps rather than closing dispersion. Bollinger-inside-Keltner is the classic
  squeeze setup.
- **Donchian Channel** — the
  rolling high/low envelope; the basis of breakout systems (the "Turtle"
  channel).

Each band family exposes the high / low / mid bands plus a **width** (`_wband`,
for squeeze detection) and a **position** (`_pband`, 0–1 within the band), and
the `_indicator` variants flag band touches.

!!! tip "OHLC volatility estimators live in Quant"
    Range-based estimators that squeeze more information out of each bar —
    Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, EWMA/RiskMetrics —
    are in [Quant](quant.md#volatility-estimators), since they're aimed at
    volatility *forecasting / risk* rather than charting.

---

::: polars_ta.volatility
