# Microstructure

**How the market is trading underneath the price** — liquidity, the cost of
crossing the spread, whether order flow looks *informed*, and how fast a series
reverts. These are standard on institutional desks and largely absent from
retail TA libraries, because they need bar-level volume classification,
autocovariance of returns, or scaling-law fits rather than a simple rolling
window. All are validated against real BTCUSDT data.

## Spread & transaction cost (from prices alone)

Recover the effective bid-ask spread without quote data:

- **`roll_spread`** — Roll (1984), from
  the negative autocovariance that bid-ask bounce induces in price changes.
- **`corwin_schultz_spread`** —
  the modern high-low estimator; more robust on OHLC bars than Roll.
- **`effective_spread`** — distance
  of the trade price from the mid.

## Price impact & liquidity

How much price moves per unit of order flow — thinner market, steeper impact:

- **`kyle_lambda`** — Kyle (1985): the
  workhorse linear price-impact measure for sizing orders against liquidity.
- **`hasbrouck_lambda`** — impact in
  log-price / √dollar-volume space, matching the empirical square-root impact
  law.

## Order-flow toxicity & trade direction

- **`vpin`** — Volume-Synchronised Probability
  of Informed Trading: order-flow imbalance in *volume time*, the canonical
  flow-toxicity warning (it spiked ahead of the 2010 Flash Crash).
- **`lee_ready_trade_sign`** —
  classifies each trade as buy- or sell-initiated (the input to signed-flow
  measures).

## Regime & mean-reversion diagnostics

Decide whether to run momentum or mean-reversion, and how fast reversion is:

- **`hurst_exponent`** — full R/S
  Hurst: persistent (>0.5) vs. mean-reverting (<0.5). (For a fast multi-scale
  version, see [`quant.hurst_ribbon`](quant.md#regime-detection-signal-conditioning).)
- **`half_life`** — how many bars a
  mean-reverting series takes to close half the gap to its mean (from an OU fit).
  The "how fast does it revert?" number for stat-arb.
- **`variance_ratio`** — Lo-MacKinlay
  test: is this a random walk (=1), trending (>1), or mean-reverting (<1)?
- **`shannon_entropy` /
  `approximate_entropy`** —
  distributional complexity and pattern predictability; complementary to Hurst
  (they don't care about direction, only structure).

!!! note "These describe conditions, not signals"
    Microstructure tools are for *classifying the environment* — is it liquid,
    is flow toxic, is it trending or reverting — and gating strategy choice,
    rather than firing buy/sell signals directly.

---

::: polars_ta.microstructure
