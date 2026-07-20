"""Plot a regime-conditional trend/mean-reversion composite signal on real
BTCUSDT 5m data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 3-panel figure:

    1. Price
    2. Multi-scale Hurst ribbon average (the regime score) with the 0.5
       trending/mean-reverting boundary
    3. The two candidate signals (EMA-cross trend signal, Bollinger %B
       mean-reversion signal) and the single composite signal that
       quant.regime_conditional_signal switches between them

Run with:
    uv run python examples/plot_regime_conditional_signal.py

Saves regime_conditional_signal.png next to this script, and a copy into
docs/assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from polars_ta import quant, trend, volatility

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "regime_conditional_signal.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "regime_conditional_signal.png"

# Okabe-Ito colorblind-safe qualitative palette (fixed order, never cycled).
INK = "#2c3e50"
BLUE = "#0072b2"
ORANGE = "#e69f00"
GREEN = "#009e73"
VERMILLION = "#d55e00"
GRID = {"linestyle": "--", "alpha": 0.3}


def main() -> None:
    df = pl.read_ipc(FIXTURE)

    ribbon = quant.hurst_ribbon("close", scales=(16, 32, 64))
    # Normalize the trend signal (an EMA-cross price difference, naturally in
    # the hundreds for BTCUSDT) against its own rolling volatility so it
    # lands on a comparable scale to the reversion signal (a Bollinger %B
    # deviation, naturally in [-0.5, 0.5]) — the composite line would
    # otherwise be dominated by whichever raw signal happens to have larger
    # natural units, which says nothing about which regime is "more right".
    raw_trend = trend.ema_indicator("close", window=10) - trend.ema_indicator(
        "close", window=30
    )
    trend_signal = raw_trend / volatility.average_true_range(
        "high", "low", "close", window=30
    )
    reversion_signal = (volatility.bollinger_pband("close") - 0.5) * 4

    out = (
        df.with_columns(**ribbon)
        .with_columns(
            trend_signal.alias("trend_signal"),
            reversion_signal.alias("reversion_signal"),
        )
        .with_columns(
            quant.regime_conditional_signal(
                "h_ribbon_avg", 0.5, "trend_signal", "reversion_signal"
            ).alias("composite")
        )
    )

    x = list(range(out.height))
    h_avg = out["h_ribbon_avg"].to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    # --- Panel 1: price, shaded by which regime/signal is active ---
    close = out["close"].to_numpy()
    ax1.plot(x, close, color=INK, linewidth=1.0, label="Close")
    trending = h_avg >= 0.5
    ax1.fill_between(
        x,
        close.min(),
        close.max(),
        where=trending,
        color=GREEN,
        alpha=0.08,
        label="Trend-follow active (H ≥ 0.5)",
    )
    ax1.fill_between(
        x,
        close.min(),
        close.max(),
        where=~trending,
        color=ORANGE,
        alpha=0.08,
        label="Mean-revert active (H < 0.5)",
    )
    ax1.set_title(
        "BTCUSDT 5m — Regime-Conditional Composite Signal (polars_ta)",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, **GRID)
    ax1.legend(loc="upper left", fontsize=8)

    # --- Panel 2: regime score ---
    ax2.plot(x, h_avg, color=BLUE, linewidth=0.9, label="Hurst ribbon avg")
    ax2.axhline(0.5, color=INK, linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("Hurst exponent")
    ax2.grid(True, **GRID)
    ax2.legend(loc="upper left", fontsize=8)

    # --- Panel 3: the two candidate signals + the composite ---
    ax3.plot(
        x,
        out["trend_signal"],
        color=GREEN,
        linewidth=0.6,
        alpha=0.5,
        label="Trend signal (EMA 10-30 cross)",
    )
    ax3.plot(
        x,
        out["reversion_signal"],
        color=ORANGE,
        linewidth=0.6,
        alpha=0.5,
        label="Reversion signal (Bollinger %B - 0.5)",
    )
    ax3.plot(
        x,
        out["composite"],
        color=VERMILLION,
        linewidth=1.1,
        label="Composite (regime_conditional_signal)",
    )
    ax3.axhline(0, color=INK, linewidth=0.6, alpha=0.4)
    ax3.set_ylabel("Signal value")
    ax3.grid(True, **GRID)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_xlabel("Bar index (5-minute bars)")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    DOCS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()
