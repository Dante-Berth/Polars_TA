"""Plot the well-known retail indicators on real BTCUSDT 5m data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 4-panel figure showing the classic
technical-analysis toolkit computed with polars_ta:

    1. Price with Bollinger Bands (20, 2) and a 50-bar SMA
    2. RSI (14) with the conventional 30/70 overbought/oversold guides
    3. MACD (12/26) line, signal, and histogram
    4. Average True Range (14)

Run with:
    uv run python examples/plot_classic_indicators.py

Saves classic_indicators.png next to this script, and a copy into
docs/assets/ so the documentation site can embed it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from polars_ta import momentum, trend, volatility

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "classic_indicators.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "classic_indicators.png"

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
        volatility.bollinger_mavg("close", window=20).alias("bb_mid"),
        volatility.bollinger_hband("close", window=20, window_dev=2).alias("bb_hi"),
        volatility.bollinger_lband("close", window=20, window_dev=2).alias("bb_lo"),
        trend.sma_indicator("close", window=50).alias("sma_50"),
        momentum.rsi("close", window=14).alias("rsi"),
        trend.macd("close").alias("macd"),
        trend.macd_signal("close").alias("macd_signal"),
        trend.macd_diff("close").alias("macd_hist"),
        volatility.average_true_range("high", "low", "close", window=14).alias("atr"),
    )

    x = list(range(out.height))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]},
    )

    # --- Panel 1: price + Bollinger Bands + SMA ---
    ax1.plot(x, out["close"], color=INK, linewidth=1.0, label="Close")
    ax1.plot(x, out["sma_50"], color=ORANGE, linewidth=1.2, label="SMA (50)")
    ax1.plot(x, out["bb_mid"], color=BLUE, linewidth=0.9, label="Bollinger mid (20)")
    ax1.fill_between(
        x,
        out["bb_lo"],
        out["bb_hi"],
        color=BLUE,
        alpha=0.12,
        label="Bollinger band (20, 2)",
    )
    ax1.set_title(
        "BTCUSDT 5m — Classic Indicators (polars_ta)",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, **GRID)
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    # --- Panel 2: RSI ---
    ax2.plot(x, out["rsi"], color=PURPLE, linewidth=1.0, label="RSI (14)")
    ax2.axhline(
        70, color=VERMILLION, linestyle="--", alpha=0.6, label="Overbought (70)"
    )
    ax2.axhline(30, color=GREEN, linestyle="--", alpha=0.6, label="Oversold (30)")
    ax2.fill_between(x, 30, 70, color=INK, alpha=0.04)
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.grid(True, **GRID)
    ax2.legend(loc="upper left", ncol=3, fontsize=8)

    # --- Panel 3: MACD ---
    hist = out["macd_hist"].to_numpy()
    ax3.bar(
        x,
        hist,
        color=[GREEN if h >= 0 else VERMILLION for h in hist],
        alpha=0.5,
        width=1.0,
        label="Histogram",
    )
    ax3.plot(x, out["macd"], color=BLUE, linewidth=1.0, label="MACD (12/26)")
    ax3.plot(x, out["macd_signal"], color=ORANGE, linewidth=1.0, label="Signal (9)")
    ax3.axhline(0, color=INK, linewidth=0.6, alpha=0.5)
    ax3.set_ylabel("MACD")
    ax3.grid(True, **GRID)
    ax3.legend(loc="upper left", ncol=3, fontsize=8)

    # --- Panel 4: ATR ---
    ax4.fill_between(x, out["atr"], color=VERMILLION, alpha=0.35, label="ATR (14)")
    ax4.plot(x, out["atr"], color=VERMILLION, linewidth=0.9)
    ax4.set_ylabel("ATR")
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
