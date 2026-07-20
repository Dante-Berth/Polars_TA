# Momentum

**Momentum indicators answer one question: is the current move *accelerating*
or *running out of fuel*?** They're oscillators — bounded or centred measures
of the *speed* of price change rather than its direction — so they shine at
spotting exhaustion (overbought/oversold) and divergence (price makes a new
high, momentum doesn't) rather than at telling you the trend.

## Which one to reach for

- **`rsi`** — the default overbought/oversold gauge
  (0–100, Wilder-smoothed). Start here.
- **`stoch` / `stoch_signal`**
  — where price closes within its recent high-low range; faster and twitchier
  than RSI, good in rangebound markets.
- **`stochrsi`** — the stochastic *of* RSI: even
  more sensitive, for very short horizons.
- **`williams_r`** — the stochastic's mirror
  image (−100 to 0); same information, different convention.
- **`tsi` / `cmo`** —
  double-smoothed / range-normalised momentum that filter noise better than raw
  RSI.
- **`ppo` / `pvo`** — MACD
  expressed in *percent*, so momentum is comparable across instruments at very
  different price levels (price PPO; volume PVO).
- **`roc`** — the rawest momentum: plain percent
  change over `n` bars.
- **`kama`** — an *adaptive* moving average that
  speeds up in trends and flattens in chop; part MA, part momentum filter.
- **`awesome_oscillator` /
  `fisher_transform` /
  `ultimate_oscillator`** —
  specialised variants (median-price momentum; a Gaussianising transform that
  sharpens turning points; a multi-timeframe blend that resists false
  divergences).

!!! tip "Momentum ≠ direction"
    An oscillator being "overbought" is not a sell signal in a strong uptrend —
    it can stay pinned for a long time. Pair momentum with a
    [trend](trend.md) or [regime](quant.md) filter before acting on extremes.

---

::: polars_ta.momentum
