"""Plot the newest indicator batch on real BTCUSDT 5m data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 4-panel figure covering both the
retail-standard and microstructure/quant additions:

    1. Price with SuperTrend and Hull Moving Average overlaid
    2. Elder Ray (Bull Power / Bear Power) vs. Chande Momentum Oscillator
    3. Fisher Transform and Klinger Volume Oscillator
    4. EWMA volatility vs. the flat-window historical volatility, and the
       Lee-Ready trade-side proxy

Run with:
    uv run python examples/plot_new_indicators.py

Saves new_indicators.png next to this script, and a copy into docs/assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from polars_ta import microstructure as ms
from polars_ta import momentum, quant, trend, volume

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "new_indicators.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "new_indicators.png"

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
        trend.supertrend("high", "low", "close", window=10, multiplier=3.0).alias(
            "supertrend"
        ),
        trend.hull_moving_average("close", window=20).alias("hma"),
        trend.elder_bull_power("high", "low", "close").alias("bull_power"),
        trend.elder_bear_power("high", "low", "close").alias("bear_power"),
        momentum.cmo("close").alias("cmo"),
        momentum.fisher_transform("high", "low").alias("fisher"),
        volume.klinger_volume_oscillator("high", "low", "close", "volume").alias(
            "kvo"
        ),
        quant.ewma_volatility("close").alias("ewma_vol"),
        quant.historical_volatility("close").alias("hist_vol"),
        ms.lee_ready_trade_sign("close").alias("trade_sign"),
    )

    x = list(range(out.height))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]},
    )

    # --- Panel 1: price + SuperTrend + HMA ---
    close = out["close"]
    st = out["supertrend"]
    ax1.plot(x, close, color=INK, linewidth=1.0, label="Close", alpha=0.6)
    up = np.where(close.to_numpy() >= st.to_numpy(), st.to_numpy(), np.nan)
    down = np.where(close.to_numpy() < st.to_numpy(), st.to_numpy(), np.nan)
    ax1.plot(x, up, color=GREEN, linewidth=1.2, label="SuperTrend (uptrend)")
    ax1.plot(x, down, color=VERMILLION, linewidth=1.2, label="SuperTrend (downtrend)")
    ax1.plot(x, out["hma"], color=PURPLE, linewidth=0.9, label="Hull MA (20)")
    ax1.set_title(
        "BTCUSDT 5m — New Indicators (polars_ta)",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, **GRID)
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    # --- Panel 2: Elder Ray + CMO ---
    ax2.fill_between(
        x, out["bull_power"], 0, color=GREEN, alpha=0.3, label="Bull Power"
    )
    ax2.fill_between(
        x, out["bear_power"], 0, color=VERMILLION, alpha=0.3, label="Bear Power"
    )
    ax2.axhline(0, color=INK, linewidth=0.6, alpha=0.5)
    ax2.set_ylabel("Elder Ray")
    ax2b = ax2.twinx()
    ax2b.plot(x, out["cmo"], color=BLUE, linewidth=0.8, label="CMO (14)")
    ax2b.axhline(50, color=BLUE, linestyle=":", linewidth=0.6, alpha=0.5)
    ax2b.axhline(-50, color=BLUE, linestyle=":", linewidth=0.6, alpha=0.5)
    ax2b.set_ylabel("CMO", color=BLUE)
    ax2.grid(True, **GRID)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # --- Panel 3: Fisher Transform + KVO ---
    ax3.plot(x, out["fisher"], color=ORANGE, linewidth=0.9, label="Fisher Transform")
    ax3.axhline(0, color=INK, linewidth=0.6, alpha=0.5)
    ax3.set_ylabel("Fisher Transform", color=ORANGE)
    ax3b = ax3.twinx()
    ax3b.plot(x, out["kvo"], color=PURPLE, linewidth=0.7, alpha=0.8, label="KVO")
    ax3b.set_ylabel("Klinger Volume Osc.", color=PURPLE)
    ax3.grid(True, **GRID)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # --- Panel 4: EWMA vol vs historical vol, with trade-sign markers ---
    ax4.plot(
        x, out["hist_vol"], color=BLUE, linewidth=0.8, alpha=0.6, label="Hist. vol (21)"
    )
    ax4.plot(
        x, out["ewma_vol"], color=VERMILLION, linewidth=1.0, label="EWMA vol (λ=0.94)"
    )
    ax4.set_ylabel("Annualized volatility")
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
