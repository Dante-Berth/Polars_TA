"""Shared expression primitives used across the indicator modules.

These are internal helpers (not part of the public API). They exist to keep the
rolling covariance / variance / regression-slope and log-return patterns —
which recur across several microstructure and quant indicators — in one place,
so a fix to the numerics applies everywhere at once.
"""

import polars as pl


def as_expr(x: str | pl.Expr) -> pl.Expr:
    """Coerce a column name or expression into an expression."""
    return pl.col(x) if isinstance(x, str) else x


def log_return(close: str | pl.Expr, periods: int = 1) -> pl.Expr:
    """Log return over ``periods`` bars, guarded against non-positive prices.

    A price <= 0 makes ``log`` undefined (``-inf``/``NaN``); those bars are
    turned into nulls so the return series stays well-defined rather than
    poisoning every downstream rolling window with ``NaN``.
    """
    close = as_expr(close)
    safe = pl.when(close > 0).then(close).otherwise(None)
    return (safe / safe.shift(periods)).log()


def rolling_cov(a: str | pl.Expr, b: str | pl.Expr, window: int) -> pl.Expr:
    """Rolling (population) covariance of two expressions over ``window`` bars."""
    a, b = as_expr(a), as_expr(b)
    a_dm = a - a.rolling_mean(window_size=window)
    b_dm = b - b.rolling_mean(window_size=window)
    return (a_dm * b_dm).rolling_mean(window_size=window)


def rolling_var(a: str | pl.Expr, window: int) -> pl.Expr:
    """Rolling (population) variance of an expression over ``window`` bars."""
    return rolling_cov(a, a, window)


def rolling_beta(y: str | pl.Expr, x: str | pl.Expr, window: int) -> pl.Expr:
    """Rolling OLS slope of ``y`` on ``x`` — cov(x, y) / var(x).

    The variance is guarded so a flat window (zero variance) yields null rather
    than a divide-by-zero, matching how every indicator signals "undefined
    here" with a null.
    """
    var = rolling_var(x, window)
    safe_var = pl.when(var == 0).then(None).otherwise(var)
    return rolling_cov(x, y, window) / safe_var
