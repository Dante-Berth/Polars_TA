"""Tests for polars_ta.calendar — the one module whose input is a timestamp
column rather than price/volume, so it gets its own file instead of being
folded into the price-indicator smoke/warmup/multi-asset suites."""

import polars as pl
import pytest

from polars_ta import calendar


def _ts_df(*timestamps: str) -> pl.DataFrame:
    return pl.DataFrame({"ts": timestamps}).with_columns(pl.col("ts").str.to_datetime())


def test_day_of_week_monday_is_zero():
    # 2024-01-01 is a Monday.
    df = _ts_df("2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-07 00:00:00")
    out = df.select(calendar.day_of_week("ts").alias("v"))["v"].to_list()
    assert out == [0, 1, 6]  # Mon, Tue, Sun


def test_is_weekend():
    df = _ts_df(
        "2024-01-05 12:00:00",  # Friday
        "2024-01-06 12:00:00",  # Saturday
        "2024-01-07 12:00:00",  # Sunday
    )
    out = df.select(calendar.is_weekend("ts").alias("v"))["v"].to_list()
    assert out == [False, True, True]


def test_hour_and_minute_of_day():
    df = _ts_df("2024-01-01 09:30:00", "2024-01-01 23:59:00")
    out = df.select(
        calendar.hour_of_day("ts").alias("h"),
        calendar.minute_of_day("ts").alias("m"),
    )
    assert out["h"].to_list() == [9, 23]
    assert out["m"].to_list() == [9 * 60 + 30, 23 * 60 + 59]


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("s", 3600.0), ("m", 60.0), ("h", 1.0)],
)
def test_time_since_midnight_units(unit, expected):
    df = _ts_df("2024-01-01 01:00:00")
    out = df.select(calendar.time_since_midnight("ts", unit=unit).alias("v"))[
        "v"
    ].to_list()
    assert out == [expected]


def test_time_since_midnight_late_hour_no_overflow():
    # Regression guard: dt.hour()/minute()/second() are Int8; hour*3600
    # overflows i8 unless widened first (23*3600 wraps in 8-bit arithmetic).
    df = _ts_df("2024-01-01 23:59:59")
    out = df.select(calendar.time_since_midnight("ts", unit="s").alias("v"))[
        "v"
    ].to_list()
    assert out == [23 * 3600 + 59 * 60 + 59]


def test_minute_of_day_late_hour_no_overflow():
    df = _ts_df("2024-01-01 23:59:00")
    out = df.select(calendar.minute_of_day("ts").alias("v"))["v"].to_list()
    assert out == [23 * 60 + 59]


def test_time_since_midnight_invalid_unit_raises():
    with pytest.raises(ValueError):
        calendar.time_since_midnight("ts", unit="days")


def test_month_of_year():
    df = _ts_df("2024-01-15 00:00:00", "2024-12-15 00:00:00")
    out = df.select(calendar.month_of_year("ts").alias("v"))["v"].to_list()
    assert out == [1, 12]


def test_is_month_end():
    # January 2024 has 31 days.
    df = _ts_df(
        "2024-01-27 00:00:00",  # 4 days from end -> False
        "2024-01-29 00:00:00",  # 2 days from end -> True
        "2024-01-31 00:00:00",  # last day -> True
    )
    out = df.select(calendar.is_month_end("ts", window_days=3).alias("v"))[
        "v"
    ].to_list()
    assert out == [False, True, True]


def test_bars_since_session_open_resets_per_session():
    df = pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
        }
    )
    out = df.with_columns(
        calendar.bars_since_session_open("date").over("date").alias("v")
    )["v"].to_list()
    assert out == [0, 1, 2, 0, 1]


def test_bars_since_session_open_uneven_bar_spacing():
    # Bar count, not wall-clock time: gaps in the timestamps don't matter,
    # only which session each row belongs to.
    df = pl.DataFrame(
        {
            "ts": ["09:00", "09:01", "09:45", "10:00"],
            "session": ["A", "A", "A", "B"],
        }
    )
    out = df.with_columns(
        calendar.bars_since_session_open("session").over("session").alias("v")
    )["v"].to_list()
    assert out == [0, 1, 2, 0]


def test_accepts_expr_not_just_column_name():
    df = _ts_df("2024-01-01 00:00:00")
    out_by_name = df.select(calendar.day_of_week("ts").alias("v"))["v"]
    out_by_expr = df.select(calendar.day_of_week(pl.col("ts")).alias("v"))["v"]
    assert out_by_name.to_list() == out_by_expr.to_list()


def test_real_fixture_bars_per_day_matches_bar_interval():
    # tests/fixtures/btcusdt_5m_sample.arrow is 5-minute bars: a full day
    # should show 287 as the max bars-since-open (288 bars, 0-indexed).
    df = pl.read_ipc("tests/fixtures/btcusdt_5m_sample.arrow")
    df = df.with_columns(
        pl.from_epoch("timestamp_open", time_unit="ms").alias("ts")
    ).with_columns(pl.col("ts").dt.date().alias("date"))
    out = df.with_columns(
        calendar.bars_since_session_open("date").over("date").alias("bso")
    )
    max_per_day = out.group_by("date").agg(pl.col("bso").max())["bso"]
    # Every full day (all but the first/last partial day) has 287.
    assert max_per_day.max() == 287
