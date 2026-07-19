"""Plot the microstructure / liquidity toolkit on real BTCUSDT 5m data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 4-panel figure of the professional-desk
liquidity features:

    1. Price
    2. Bid-ask spread estimators: Roll vs Corwin-Schultz
    3. Kyle's lambda (price impact per unit signed volume)
    4. Half-life of mean reversion

Run with:
    uv run python examples/plot_liquidity.py

Saves liquidity.png next to this script, and a copy into docs/assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from polars_ta import microstructure as ms

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "liquidity.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "liquidity.png"

# Okabe-Ito colorblind-safe qualitative palette (fixed order, never cycled).
INK = "#2c3e50"
BLUE = "#0072b2"
ORANGE = "#e69f00"
GREEN = "#009e73"
VERMILLION = "#d55e00"
GRID = {"linestyle": "--", "alpha": 0.3}


def main() -> None:
    df = pl.read_ipc(FIXTURE)

    out = df.with_columns(
        ms.roll_spread("close", window=20).alias("roll"),
        # Corwin-Schultz is a *fractional* spread; rescale to price units
        # (× close) so it shares a single honest axis with Roll's spread.
        (ms.corwin_schultz_spread("high", "low", window=20) * pl.col("close")).alias(
            "corwin"
        ),
        ms.kyle_lambda("close", "volume", window=50).alias("kyle"),
        ms.half_life("close", window=60).alias("half_life"),
    )

    x = list(range(out.height))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]},
    )

    # --- Panel 1: price ---
    ax1.plot(x, out["close"], color=INK, linewidth=1.0, label="Close")
    ax1.set_title(
        "BTCUSDT 5m — Liquidity & Microstructure (polars_ta)",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, **GRID)
    ax1.legend(loc="upper left", fontsize=8)

    # --- Panel 2: spread estimators ---
    ax2.plot(x, out["roll"], color=BLUE, linewidth=0.7, alpha=0.7, label="Roll spread")
    ax2.plot(
        x,
        out["corwin"],
        color=ORANGE,
        linewidth=1.1,
        alpha=0.95,
        label="Corwin-Schultz spread",
    )
    ax2.set_ylabel("Spread (USDT)")
    ax2.grid(True, **GRID)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)

    # --- Panel 3: Kyle's lambda (price impact) ---
    kyle = out["kyle"]
    ax3.fill_between(x, kyle, color=VERMILLION, alpha=0.3, label="Kyle's lambda")
    ax3.plot(x, kyle, color=VERMILLION, linewidth=0.8)
    ax3.set_ylabel("Kyle's lambda")
    ax3.grid(True, **GRID)
    ax3.legend(loc="upper left", fontsize=8)

    # --- Panel 4: half-life of mean reversion ---
    # When reversion is weak the OU half-life explodes toward infinity; cap the
    # view at 300 bars so the fast-reverting structure stays legible (spikes
    # above the cap are clipped, not dropped).
    hl = out["half_life"]
    ax4.plot(x, hl, color=GREEN, linewidth=0.8, label="Half-life (bars, capped 300)")
    ax4.set_ylabel("Half-life (bars)")
    ax4.set_ylim(0, 300)
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
