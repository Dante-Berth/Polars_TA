# Expression namespace (`.ta`)

Importing `polars_ta` registers a `.ta` accessor on every Polars expression, so
indicators read like native Polars methods:

```python
import polars as pl
import polars_ta  # registers .ta on import

df.with_columns(
    pl.col("close").ta.rsi(14).alias("rsi"),
    pl.col("high").ta.average_true_range("low", "close").alias("atr"),
)
```

The expression you call `.ta` on is bound to the indicator's **first input**
(`close` for most, `high` for the high-anchored ones); remaining columns are
passed as arguments. It's the *same code* as the module-level functions — a
byte-for-byte-identical dispatch layer — so `.over(...)`, streaming and `fillna`
all work through it. `polars_ta.TA_INDICATORS` lists every exposed name.

Cross-sectional and regime-composite helpers, which take a generic value column
or two pre-built signals rather than a single price series, stay
free-function-only.

See the [how-to guide](../how_to_guides.md#call-indicators-as-plcoltaname)
for the full calling convention.

---

::: polars_ta.namespace
