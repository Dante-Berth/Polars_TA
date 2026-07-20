# Utilities

Shared building blocks for *composing* and *cleaning* — the tools you use around
the indicators rather than the indicators themselves.

- **`BaseIndicator`** — the reusable primitives
  every indicator is built from (`sma`, `ema`, `true_range`, `check_fillna`,
  `get_min_max`). Use these when you [build a custom
  indicator](../how_to_guides.md#build-a-custom-indicator-on-top-of-existing-ones)
  so it inherits the same warm-up and `fillna` behaviour as the built-ins.
- **`DataCleaner`** — detect and repair the
  `NaN`/`inf`/`null`/absurdly-large values that real market data is full of
  (`dropna`, `get_invalid_indices`, `approximate_invalid_values`) *before* they
  cascade through a rolling window and null out a whole column.

!!! tip "Clean first, compute second"
    A single bad tick inside a rolling window can poison every value that window
    touches. Run `DataCleaner` on raw feeds before computing indicators — see
    the [cleaning how-to](../how_to_guides.md#clean-invalid-values-before-computing-indicators).

---

::: polars_ta.utils
