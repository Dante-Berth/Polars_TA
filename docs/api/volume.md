# Volume

**Volume indicators answer: is the price move backed by real participation, or
is it hollow?** Volume is the fuel behind price. These tools combine price and
volume to reveal accumulation vs. distribution and to flag moves that lack
conviction (a breakout on thin volume is suspect).

## Cumulative flow (accumulation vs. distribution)

- **`on_balance_volume`** — adds volume on
  up days, subtracts on down days. The simplest running tally of buying vs.
  selling pressure. **Start here.**
- **`acc_dist_index`** — like OBV but weights
  each bar by *where* it closed in its range, so a strong close counts more than
  a weak one.
- **`volume_price_trend` /
  `negative_volume_index`** — VPT
  scales the flow by the size of the return; NVI tracks what "smart money" does
  on *quiet* (low-volume) days.

## Flow oscillators & pressure

- **`chaikin_money_flow`** — accumulation/
  distribution as a bounded oscillator over a window; positive = buying
  pressure.
- **`money_flow_index`** — a
  *volume-weighted RSI* (0–100): overbought/oversold that also requires volume
  to confirm.
- **`force_index`** — combines the *size* of a
  move with its volume to gauge the power behind it.
- **`klinger_volume_oscillator`** —
  a long/short-term volume-force difference aimed at spotting reversals.

## Movement efficiency & fair price

- **`ease_of_movement` /
  `sma_ease_of_movement`** — how far
  price moved *per unit of volume*: big moves on light volume score high (price
  moves "easily").
- **`volume_weighted_average_price`**
  — VWAP, the execution benchmark: the average price *weighted by volume*, i.e.
  where the bulk of trading actually happened.

!!! tip "Confirm, don't lead"
    Volume tools are best as *confirmation*: a price breakout with rising OBV/CMF
    is trustworthy; the same breakout with falling volume flow often fails.

---

::: polars_ta.volume
