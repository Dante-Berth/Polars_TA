"""Polars TA — Technical Analysis library built on Polars expressions.

Every indicator returns a lazy ``pl.Expr`` that you can plug directly into
``df.with_columns(...)``, so computations stay vectorized and parallel.
"""

from polars_ta import (
    microstructure,
    momentum,
    others,
    quant,
    trend,
    volatility,
    volume,
)
from polars_ta.utils import BaseIndicator, DataCleaner

__version__ = "0.2.0"

__all__ = [
    "momentum",
    "trend",
    "volatility",
    "volume",
    "quant",
    "microstructure",
    "others",
    "BaseIndicator",
    "DataCleaner",
    "__version__",
]
