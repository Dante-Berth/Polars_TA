"""Plot a professional-desk regime dashboard on real BTCUSDT 5m data.

Uses tests/fixtures/btcusdt_5m_sample.arrow (5000 real Binance BTCUSDT
5-minute bars, Dec 2025) to produce a 3-panel dashboard: price, multi-scale
Hurst ribbon (trend vs mean-reversion regime), and Yang-Zhang volatility +
VPIN (order-flow toxicity).

Run with:
    uv run python examples/plot_regime_dashboard.py

Saves regime_dashboard.png next to this script, and a copy into
docs/assets/ so the documentation site can embed it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from polars_ta import microstructure, quant

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_PATH = Path(__file__).parent / "regime_dashboard.png"
DOCS_OUT_PATH = ROOT / "docs" / "assets" / "regime_dashboard.png"


def main() -> None:
    df = pl.read_ipc(FIXTURE)

    out = df.with_columns(
        quant.yang_zhang_volatility("open", "high", "low", "close", window=48).alias(
            "yz_vol"
        ),
        microstructure.vpin("close", "volume", bucket_size=500, window=20).alias(
            "vpin"
        ),
        **quant.hurst_ribbon("close", scales=(16, 32, 64)),
    )

    x = range(out.height)
    close = out["close"]
    h_avg = out["h_ribbon_avg"]
    yz_vol = out["yz_vol"]
    vpin = out["vpin"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    # --- Panel 1: price ---
    ax1.plot(x, close, color="#2c3e50", linewidth=1.0)
    ax1.set_title(
        "BTCUSDT 5m — Regime Dashboard (polars_ta)", fontsize=15, fontweight="bold"
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # --- Panel 2: Hurst ribbon (trend vs mean-reversion regime) ---
    # Smooth the regime average so the shading reads as a regime, not noise.
    h_avg_smooth = h_avg.fill_null(0.5).rolling_mean(window_size=50, min_samples=1)
    ax2.axhline(
        0.5, color="#e74c3c", linestyle="--", alpha=0.6, label="Random walk (0.5)"
    )
    ax2.fill_between(
        x,
        0.3,
        0.7,
        where=(h_avg_smooth > 0.5),
        color="#2ecc71",
        alpha=0.15,
        label="Trending regime",
    )
    ax2.fill_between(
        x,
        0.3,
        0.7,
        where=(h_avg_smooth <= 0.5),
        color="#9b59b6",
        alpha=0.15,
        label="Mean-reverting regime",
    )
    ax2.plot(x, h_avg, color="#2980b9", linewidth=0.8, alpha=0.9, label="H-ribbon avg")
    ax2.set_ylabel("Hurst exponent", color="#2980b9")
    ax2.set_ylim(0.3, 0.7)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)

    # --- Panel 3: Yang-Zhang volatility + VPIN ---
    ax3.fill_between(x, yz_vol, color="#e67e22", alpha=0.4, label="Yang-Zhang vol")
    ax3.set_ylabel("YZ volatility (annualized)", color="#e67e22")
    ax3.grid(True, linestyle="--", alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.plot(x, vpin, color="#c0392b", linewidth=1.2, label="VPIN")
    ax3b.set_ylabel("VPIN (order-flow toxicity)", color="#c0392b")
    ax3b.set_ylim(0, 1)

    lines3 = ax3.get_lines() + ax3b.get_lines()
    ax3.legend(lines3, [line_.get_label() for line_ in lines3], loc="upper left")

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
