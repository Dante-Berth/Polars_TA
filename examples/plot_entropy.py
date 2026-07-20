"""Plot the entropy-based regime/complexity indicators on real BTCUSDT 5m
data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 3-panel figure:

    1. Price
    2. Shannon entropy (distributional concentration) vs. the Hurst ribbon
       (directional persistence) — two complementary regime signals
    3. Approximate entropy (pattern predictability)

Run with:
    uv run python examples/plot_entropy.py

Saves entropy.png next to this script, and a copy into docs/assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from polars_ta import microstructure as ms
from polars_ta import quant

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "entropy.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "entropy.png"

# Okabe-Ito colorblind-safe qualitative palette (fixed order, never cycled).
INK = "#2c3e50"
BLUE = "#0072b2"
ORANGE = "#e69f00"
GREEN = "#009e73"
GRID = {"linestyle": "--", "alpha": 0.3}


def main() -> None:
    df = pl.read_ipc(FIXTURE)

    out = df.with_columns(
        ms.shannon_entropy("close", window=50, n_bins=10).alias("shannon"),
        ms.approximate_entropy("close", window=30).alias("apen"),
        **quant.hurst_ribbon("close", scales=(16, 32, 64)),
    )

    x = list(range(out.height))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    # --- Panel 1: price ---
    ax1.plot(x, out["close"], color=INK, linewidth=1.0, label="Close")
    ax1.set_title(
        "BTCUSDT 5m — Entropy-Based Regime Indicators (polars_ta)",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, **GRID)
    ax1.legend(loc="upper left", fontsize=8)

    # --- Panel 2: Shannon entropy vs Hurst ribbon ---
    ax2.plot(
        x, out["shannon"], color=ORANGE, linewidth=0.9, label="Shannon entropy (50)"
    )
    ax2.set_ylabel("Shannon entropy [0, 1]", color=ORANGE)
    ax2.set_ylim(0, 1.05)
    ax2b = ax2.twinx()
    ax2b.plot(
        x, out["h_ribbon_avg"], color=BLUE, linewidth=0.9, label="Hurst ribbon avg"
    )
    ax2b.axhline(0.5, color=BLUE, linestyle=":", linewidth=0.7, alpha=0.6)
    ax2b.set_ylabel("Hurst exponent", color=BLUE)
    ax2.grid(True, **GRID)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # --- Panel 3: Approximate entropy ---
    apen = out["apen"]
    ax3.fill_between(x, apen, color=GREEN, alpha=0.3)
    ax3.plot(x, apen, color=GREEN, linewidth=0.8, label="Approximate entropy (30)")
    ax3.set_ylabel("Approximate entropy")
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
