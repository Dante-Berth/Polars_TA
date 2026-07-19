# Examples

## Quickstart: a full indicator bundle

The runnable version of this lives at [`examples/quickstart.py`](https://github.com/Dante-Berth/Polars_TA/blob/main/examples/quickstart.py):

```python
--8<-- "examples/quickstart.py"
```

Run it with:

```bash
uv run python examples/quickstart.py
```

## More indicator combinations

### Trend + volatility regime filter

Combine ADX (trend strength) with Bollinger Band width (volatility) to flag "trending and volatile" periods:

```python
from polars_ta import trend, volatility

out = df.with_columns(
    trend.adx("high", "low", "close").alias("adx"),
    volatility.bollinger_wband("close").alias("bb_width"),
).with_columns(
    ((pl.col("adx") > 25) & (pl.col("bb_width") > pl.col("bb_width").rolling_mean(20)))
    .alias("trending_and_volatile")
)
```

### Volume-confirmed momentum

Require both RSI momentum and Money Flow Index (volume-weighted RSI-analogue) to agree:

```python
from polars_ta import momentum, volume

out = df.with_columns(
    momentum.rsi("close").alias("rsi"),
    volume.money_flow_index("high", "low", "close", "volume").alias("mfi"),
).with_columns(
    ((pl.col("rsi") > 70) & (pl.col("mfi") > 80)).alias("overbought_confirmed")
)
```
