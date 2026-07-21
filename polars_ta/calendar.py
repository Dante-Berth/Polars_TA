"""Calendar / seasonality feature expressions.

Every other module in this library takes a price or volume column as input.
This module is the odd one out: it takes a `pl.Datetime` (or `pl.Date`)
column and derives features from the *calendar*, not the market data —
day-of-week, hour-of-day, time-since-session-open, and similar. These are
mundane compared to the rest of the toolkit but a real gap in a full feature
pipeline: seasonality effects (day-of-week volume patterns, intraday
volatility curves) are common enough in practice that most desks compute
some version of these.

Timestamps must already be a proper `pl.Datetime`/`pl.Date` column — convert
epoch integers first with `pl.from_epoch(col, time_unit=...)`. This module
intentionally does not hardcode market-specific trading-session windows
(e.g. FX Asian/London/NY hours): those are UTC-hour conventions that vary by
instrument and would be silently wrong for half of any given user's data.
`hour_of_day` / `minute_of_day` give you the building blocks to define your
own session boundaries per instrument.
"""

import polars as pl

from polars_ta._internal import as_expr


def _as_datetime_expr(timestamp: str | pl.Expr) -> pl.Expr:
    return as_expr(timestamp)


def day_of_week(timestamp: str | pl.Expr) -> pl.Expr:
    """Day of week as an integer, Monday=0 .. Sunday=6."""
    ts = _as_datetime_expr(timestamp)
    return (ts.dt.weekday() - 1).alias("day_of_week")


def is_weekend(timestamp: str | pl.Expr) -> pl.Expr:
    """True for Saturday/Sunday. Meaningless (always False) on 24/7 markets
    like crypto, but relevant for equities/FX data that excludes weekends."""
    ts = _as_datetime_expr(timestamp)
    return (ts.dt.weekday() >= 6).alias("is_weekend")


def hour_of_day(timestamp: str | pl.Expr) -> pl.Expr:
    """Hour of day in the timestamp's own timezone/offset, `[0, 23]`."""
    ts = _as_datetime_expr(timestamp)
    return ts.dt.hour().alias("hour_of_day")


def minute_of_day(timestamp: str | pl.Expr) -> pl.Expr:
    """Minutes since midnight, `[0, 1439]` — a finer-grained alternative to
    `hour_of_day` for defining custom intraday session boundaries."""
    ts = _as_datetime_expr(timestamp)
    # dt.hour()/dt.minute() return Int8; the product overflows i8 (e.g.
    # 23 * 60 = 1380), so widen before multiplying.
    hour = ts.dt.hour().cast(pl.Int32)
    minute = ts.dt.minute().cast(pl.Int32)
    return (hour * 60 + minute).alias("minute_of_day")


def time_since_midnight(timestamp: str | pl.Expr, unit: str = "m") -> pl.Expr:
    """Elapsed time since the start of the timestamp's own calendar day.

    `unit` is `"s"`, `"m"` (default), or `"h"`.
    """
    if unit not in ("s", "m", "h"):
        raise ValueError('"unit" must be "s", "m", or "h"')
    ts = _as_datetime_expr(timestamp)
    # dt.hour()/dt.minute()/dt.second() return Int8; widen before
    # multiplying by 3600/60 to avoid silent overflow.
    hour = ts.dt.hour().cast(pl.Int32)
    minute = ts.dt.minute().cast(pl.Int32)
    second = ts.dt.second().cast(pl.Int32)
    seconds = hour * 3600 + minute * 60 + second
    divisor = {"s": 1, "m": 60, "h": 3600}[unit]
    return (seconds / divisor).alias(f"time_since_midnight_{unit}")


def bars_since_session_open(session_col: str | pl.Expr) -> pl.Expr:
    """Bar count elapsed since the start of the current session, where a
    "session" is any grouping column that changes value at each session
    boundary — a date column for a daily session, or a custom session-id
    column for intraday sessions with gaps.

    This counts bars, not wall-clock time, so it's correct regardless of
    whether bars are evenly spaced. Apply with `.over(session_col)` so the
    counter restarts at zero for every session:

        calendar.bars_since_session_open("date").over("date")
    """
    session = as_expr(session_col)
    return pl.int_range(pl.len()).over(session).alias("bars_since_session_open")


def month_of_year(timestamp: str | pl.Expr) -> pl.Expr:
    """Month as an integer, `[1, 12]` — for detecting monthly seasonality
    (e.g. "January effect"-style patterns)."""
    ts = _as_datetime_expr(timestamp)
    return ts.dt.month().alias("month_of_year")


def is_month_end(timestamp: str | pl.Expr, window_days: int = 3) -> pl.Expr:
    """True within `window_days` of the end of the calendar month —
    approximates the month-end rebalancing window that drives elevated
    volume/volatility at many institutional desks."""
    ts = _as_datetime_expr(timestamp)
    days_in_month = ts.dt.month_end().dt.day()
    return (days_in_month - ts.dt.day() < window_days).alias("is_month_end")
