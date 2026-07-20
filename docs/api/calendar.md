# Calendar / seasonality

The one family whose input is a **timestamp**, not price or volume. Markets have
clocks: Monday behaves differently from Friday, the first hour differently from
lunchtime, month-end differently from mid-month. These turn a datetime column
into features that capture those recurring effects.

- **`day_of_week` /
  `is_weekend`** — weekday effects and the
  crypto weekend regime.
- **`hour_of_day` /
  `minute_of_day` /
  `time_since_midnight`** — intraday
  position, the raw material for your own session definitions.
- **`month_of_year` /
  `is_month_end`** — seasonal and month-end
  rebalancing effects.
- **`bars_since_session_open`** —
  how far into the session each bar is; apply with `.over(session_id)` so the
  count restarts each session.

!!! note "No hardcoded market sessions"
    This module deliberately doesn't bake in market-specific session windows
    (FX Asian/London/NY hours vary by instrument and are UTC-hour conventions).
    `hour_of_day` / `minute_of_day` give you the primitives to define your own —
    see the [calendar how-to](../how_to_guides.md#add-calendarseasonality-features-to-a-feature-pipeline).

---

::: polars_ta.calendar
