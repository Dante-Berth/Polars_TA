"""Plot the trend & volume toolkit on real BTCUSDT 5m data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 4-panel figure:

    1. Price with the Ichimoku cloud (senkou span A/B)
    2. ADX (14) trend strength with +DI / -DI
    3. Aroon up / down
    4. On-Balance Volume

Run with:
    uv run python examples/plot_trend_volume.py

Saves trend_volume.png next to this script, and a copy into docs/assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from polars_ta import trend, volume

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "trend_volume.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "trend_volume.png"

# Okabe-Ito colorblind-safe qualitative palette (fixed order, never cycled).
INK = "#2c3e50"
BLUE = "#0072b2"
ORANGE = "#e69f00"
GREEN = "#009e73"
VERMILLION = "#d55e00"
PURPLE = "#cc79a7"
GRID = {"linestyle": "--", "alpha": 0.3}


def main() -> None:
    df = pl.read_ipc(FIXTURE)

    out = df.with_columns(
        trend.ichimoku_a("high", "low").alias("ichimoku_a"),
        trend.ichimoku_b("high", "low").alias("ichimoku_b"),
        trend.adx("high", "low", "close", window=14).alias("adx"),
        trend.adx_pos("high", "low", "close", window=14).alias("adx_pos"),
        trend.adx_neg("high", "low", "close", window=14).alias("adx_neg"),
        trend.aroon_up("high", "low", window=25).alias("aroon_up"),
        trend.aroon_down("high", "low", window=25).alias("aroon_down"),
        volume.on_balance_volume("close", "volume").alias("obv"),
    )

    x = list(range(out.height))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]},
    )

    # --- Panel 1: price + Ichimoku cloud ---
    ax1.plot(x, out["close"], color=INK, linewidth=1.0, label="Close")
    span_a = out["ichimoku_a"]
    span_b = out["ichimoku_b"]
    ax1.plot(x, span_a, color=GREEN, linewidth=0.8, alpha=0.8, label="Senkou span A")
    ax1.plot(
        x, span_b, color=VERMILLION, linewidth=0.8, alpha=0.8, label="Senkou span B"
    )
    ax1.fill_between(
        x, span_a, span_b, where=(span_a >= span_b), color=GREEN, alpha=0.12
    )
    ax1.fill_between(
        x, span_a, span_b, where=(span_a < span_b), color=VERMILLION, alpha=0.12
    )
    ax1.set_title(
        "BTCUSDT 5m — Trend & Volume (polars_ta)",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, **GRID)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    # --- Panel 2: ADX + directional indicators ---
    ax2.plot(x, out["adx"], color=INK, linewidth=1.2, label="ADX (14)")
    ax2.plot(x, out["adx_pos"], color=GREEN, linewidth=0.9, alpha=0.9, label="+DI")
    ax2.plot(x, out["adx_neg"], color=VERMILLION, linewidth=0.9, alpha=0.9, label="-DI")
    ax2.axhline(25, color=INK, linestyle="--", alpha=0.5, label="Trend threshold (25)")
    ax2.set_ylabel("ADX")
    ax2.grid(True, **GRID)
    ax2.legend(loc="upper left", ncol=4, fontsize=8)

    # --- Panel 3: Aroon oscillator (up - down) ---
    # Over 5000 bars the raw up/down lines whip between 0 and 100 too fast to
    # read; the oscillator (up - down), lightly smoothed, shows the same
    # up/down dominance as a single legible series shaded by sign.
    aroon_osc = (out["aroon_up"] - out["aroon_down"]).rolling_mean(
        window_size=10, min_samples=1
    )
    ax3.fill_between(
        x, aroon_osc, 0, where=(aroon_osc >= 0), color=GREEN, alpha=0.4
    )
    ax3.fill_between(
        x, aroon_osc, 0, where=(aroon_osc < 0), color=PURPLE, alpha=0.4
    )
    ax3.plot(x, aroon_osc, color=INK, linewidth=0.6, label="Aroon oscillator (25)")
    ax3.axhline(0, color=INK, linewidth=0.6, alpha=0.5)
    ax3.set_ylabel("Aroon osc.")
    ax3.set_ylim(-100, 100)
    ax3.grid(True, **GRID)
    ax3.legend(loc="upper left", fontsize=8)

    # --- Panel 4: On-Balance Volume ---
    ax4.fill_between(x, out["obv"], color=BLUE, alpha=0.3, label="OBV")
    ax4.plot(x, out["obv"], color=BLUE, linewidth=0.9)
    ax4.set_ylabel("OBV")
    ax4.grid(True, **GRID)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.set_xlabel("Bar index (5-minute bars)")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    DOCS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()
