"""Polars TA — Technical Analysis library built on Polars expressions.

Every indicator returns a lazy ``pl.Expr`` that you can plug directly into
``df.with_columns(...)``, so computations stay vectorized and parallel.
"""

from polars_ta import (
    calendar,
    microstructure,
    momentum,
    others,
    quant,
    trend,
    volatility,
    volume,
)

# Registers the ``.ta`` expression namespace (pl.col("close").ta.rsi(...)) as a
# side effect of import. Imported last so every indicator module it wraps is
# already defined. See polars_ta.namespace for the calling convention.
from polars_ta import namespace as _namespace  # noqa: E402
from polars_ta.utils import BaseIndicator, DataCleaner

__version__ = "0.2.0"

#: Indicator names reachable via ``pl.col(...).ta.<name>()``.
TA_INDICATORS = _namespace.INDICATOR_NAMES

__all__ = [
    "momentum",
    "trend",
    "volatility",
    "volume",
    "quant",
    "microstructure",
    "calendar",
    "others",
    "BaseIndicator",
    "DataCleaner",
    "TA_INDICATORS",
    "__version__",
]
