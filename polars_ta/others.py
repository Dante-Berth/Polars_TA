"""Return-based indicators (daily return, log return, cumulative return)."""

import polars as pl

from polars_ta.utils import BaseIndicator


def daily_return(close: str | pl.Expr, fillna: bool = False) -> pl.Expr:
    """Daily percentage return, in percent."""
    close = pl.col(close) if isinstance(close, str) else close
    dr = (close / close.shift(1) - 1.0) * 100.0
    return BaseIndicator.check_fillna(dr, fillna, value=0)


def daily_log_return(close: str | pl.Expr, fillna: bool = False) -> pl.Expr:
    """Daily logarithmic return, in percent."""
    close = pl.col(close) if isinstance(close, str) else close
    dlr = (close / close.shift(1)).log() * 100.0
    return BaseIndicator.check_fillna(dlr, fillna, value=0)


def cumulative_return(close: str | pl.Expr, fillna: bool = False) -> pl.Expr:
    """Cumulative return since the first observation, in percent."""
    close = pl.col(close) if isinstance(close, str) else close
    cr = (close / close.first() - 1.0) * 100.0
    return BaseIndicator.check_fillna(cr, fillna, value=0)
