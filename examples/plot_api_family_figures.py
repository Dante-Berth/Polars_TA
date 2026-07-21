"""Generate one representative figure per API-reference family.

Each family page under docs/api/ opens with a small illustrative chart so the
human reader can *see* what that family of indicators does before reading the
reference. This script computes a handful of indicators from each family on the
real BTCUSDT 5-minute fixture and saves one PNG per family into
docs/assets/api/.

Run with:
    uv run python examples/plot_api_family_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to write PNGs
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from polars_ta import (  # noqa: E402
    calendar,
    momentum,
    others,
    quant,
    trend,
    volatility,
    volume,
)

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "btcusdt_5m_sample.arrow"
OUT_DIR = ROOT / "docs" / "assets" / "api"

# Okabe-Ito colorblind-safe qualitative palette (matches the other examples).
INK = "#2c3e50"
BLUE = "#0072b2"
ORANGE = "#e69f00"
GREEN = "#009e73"
VERMILLION = "#d55e00"
PURPLE = "#cc79a7"
SKY = "#56b4e9"
GRID = {"linestyle": "--", "alpha": 0.3}


def _style(ax, title, ylabel=None, legend=True):
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(**GRID)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def momentum_fig(df, x):
    out = df.with_columns(
        momentum.rsi("close", window=14).alias("rsi"),
        momentum.stoch("high", "low", "close").alias("stoch"),
    )
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    a1.plot(x, out["close"], color=INK, linewidth=0.9, label="Close")
    _style(a1, "Price", "USDT")
    a2.plot(x, out["rsi"], color=BLUE, linewidth=1.0, label="RSI (14)")
    a2.plot(x, out["stoch"], color=ORANGE, linewidth=0.8, alpha=0.8, label="Stoch %K")
    a2.axhline(70, color=VERMILLION, **GRID)
    a2.axhline(30, color=GREEN, **GRID)
    _style(a2, "Momentum oscillators — overbought (>70) / oversold (<30)", "0–100")
    fig.suptitle(
        "Momentum", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left"
    )
    _save(fig, "momentum.png")


def trend_fig(df, x):
    out = df.with_columns(
        trend.ema_indicator("close", window=20).alias("ema20"),
        trend.sma_indicator("close", window=50).alias("sma50"),
        trend.macd("close").alias("macd"),
        trend.macd_signal("close").alias("sig"),
        trend.macd_diff("close").alias("hist"),
        trend.adx("high", "low", "close").alias("adx"),
    )
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    a1.plot(x, out["close"], color=INK, linewidth=0.9, label="Close")
    a1.plot(x, out["ema20"], color=BLUE, linewidth=1.0, label="EMA (20)")
    a1.plot(x, out["sma50"], color=ORANGE, linewidth=1.0, label="SMA (50)")
    _style(a1, "Price with moving averages", "USDT")
    a2.plot(x, out["macd"], color=BLUE, linewidth=1.0, label="MACD")
    a2.plot(x, out["sig"], color=VERMILLION, linewidth=0.9, label="Signal")
    a2.bar(x, out["hist"], color=GREEN, alpha=0.4, width=1.0, label="Histogram")
    _style(a2, "MACD (12/26/9)")
    a3.plot(x, out["adx"], color=PURPLE, linewidth=1.1, label="ADX (14)")
    a3.axhline(25, color=INK, **GRID)
    _style(a3, "ADX — trend strength (>25 = trending)")
    fig.suptitle("Trend", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    _save(fig, "trend.png")


def volatility_fig(df, x):
    out = df.with_columns(
        volatility.bollinger_mavg("close", window=20).alias("mid"),
        volatility.bollinger_hband("close", window=20, window_dev=2).alias("hi"),
        volatility.bollinger_lband("close", window=20, window_dev=2).alias("lo"),
        volatility.average_true_range("high", "low", "close").alias("atr"),
    )
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    a1.plot(x, out["close"], color=INK, linewidth=0.9, label="Close")
    a1.plot(x, out["mid"], color=BLUE, linewidth=0.8, label="Bollinger mid (20)")
    a1.fill_between(
        x, out["lo"], out["hi"], color=BLUE, alpha=0.13, label="Bollinger band (20, 2)"
    )
    _style(a1, "Price with a volatility envelope", "USDT")
    a2.plot(x, out["atr"], color=VERMILLION, linewidth=1.0, label="ATR (14)")
    _style(a2, "Average True Range — how much price moves per bar", "USDT")
    fig.suptitle(
        "Volatility", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left"
    )
    _save(fig, "volatility.png")


def volume_fig(df, x):
    out = df.with_columns(
        volume.on_balance_volume("close", "volume").alias("obv"),
        volume.money_flow_index("high", "low", "close", "volume").alias("mfi"),
    )
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1, 2]}
    )
    a1.plot(x, out["close"], color=INK, linewidth=0.9, label="Close")
    _style(a1, "Price", "USDT")
    a2.bar(x, out["volume"], color=SKY, width=1.0, label="Volume")
    _style(a2, "Volume")
    ln1 = a3.plot(x, out["obv"], color=GREEN, linewidth=1.0, label="OBV")
    a3b = a3.twinx()
    ln2 = a3b.plot(
        x, out["mfi"], color=ORANGE, linewidth=0.8, alpha=0.8, label="MFI (14)"
    )
    a3b.set_ylabel("MFI", fontsize=9)
    _style(a3, "On-Balance Volume (flow) + Money Flow Index", legend=False)
    lns = ln1 + ln2
    a3.legend(lns, [ln.get_label() for ln in lns], loc="upper left", fontsize=8)
    fig.suptitle("Volume", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    _save(fig, "volume.png")


def returns_fig(df, x):
    out = df.with_columns(
        others.daily_return("close").alias("ret"),
        others.cumulative_return("close").alias("cum"),
    )
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    a1.plot(x, out["cum"], color=BLUE, linewidth=1.1, label="Cumulative return (%)")
    a1.axhline(0, color=INK, **GRID)
    _style(a1, "Cumulative return — the equity-curve view", "%")
    a2.plot(x, out["ret"], color=VERMILLION, linewidth=0.6, label="Per-bar return (%)")
    a2.axhline(0, color=INK, **GRID)
    _style(a2, "Per-bar simple return", "%")
    fig.suptitle(
        "Returns", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left"
    )
    _save(fig, "returns.png")


def quant_fig(df, x):
    ribbon = quant.hurst_ribbon("close", scales=(16, 32, 64))
    out = df.with_columns(
        quant.yang_zhang_volatility("open", "high", "low", "close").alias("yz"),
        quant.rolling_max_drawdown("close", window=200).alias("mdd"),
        ribbon["h_ribbon_avg"],
    )
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    a1.plot(x, out["close"], color=INK, linewidth=0.9, label="Close")
    _style(a1, "Price", "USDT")
    a2.plot(x, out["yz"], color=PURPLE, linewidth=1.0, label="Yang-Zhang vol")
    a2.plot(x, out["mdd"], color=VERMILLION, linewidth=0.9, label="Max drawdown (200)")
    _style(a2, "Risk: OHLC volatility & rolling drawdown")
    a3.plot(
        x, out["h_ribbon_avg"], color=GREEN, linewidth=1.1, label="Hurst ribbon avg"
    )
    a3.axhline(0.5, color=INK, **GRID)
    _style(a3, "Regime: Hurst > 0.5 trending, < 0.5 mean-reverting")
    fig.suptitle("Quant", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    _save(fig, "quant.png")


def microstructure_fig(df, x):
    from polars_ta import microstructure as ms

    out = df.with_columns(
        ms.roll_spread("close", window=50).alias("roll"),
        ms.kyle_lambda("close", "volume", window=50).alias("kyle"),
        ms.hurst_exponent("close", window=200).alias("hurst"),
    )
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    a1.plot(x, out["close"], color=INK, linewidth=0.9, label="Close")
    _style(a1, "Price", "USDT")
    ln1 = a2.plot(x, out["roll"], color=BLUE, linewidth=0.9, label="Roll spread (50)")
    a2b = a2.twinx()
    ln2 = a2b.plot(
        x, out["kyle"], color=ORANGE, linewidth=0.8, alpha=0.8, label="Kyle λ (50)"
    )
    a2b.set_ylabel("Kyle λ", fontsize=9)
    _style(a2, "Liquidity: implied spread & price impact", legend=False)
    lns = ln1 + ln2
    a2.legend(lns, [ln.get_label() for ln in lns], loc="upper left", fontsize=8)
    a3.plot(x, out["hurst"], color=GREEN, linewidth=1.1, label="Hurst R/S (200)")
    a3.axhline(0.5, color=INK, **GRID)
    _style(a3, "Regime: Hurst exponent (R/S)")
    fig.suptitle(
        "Microstructure", fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left"
    )
    _save(fig, "microstructure.png")


def calendar_fig(df, x):
    out = df.with_columns(
        pl.from_epoch("timestamp_open", time_unit="ms").alias("ts"),
    ).with_columns(
        calendar.hour_of_day("ts").alias("hour"),
        others.daily_return("close").alias("ret"),
    )
    # Average absolute per-bar move by hour of day — an intraday seasonality read.
    by_hour = (
        out.group_by("hour")
        .agg(pl.col("ret").abs().mean().alias("avg_abs_ret"))
        .sort("hour")
        .drop_nulls()
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(by_hour["hour"], by_hour["avg_abs_ret"], color=SKY, width=0.8)
    ax.set_xlabel("Hour of day (UTC)", fontsize=9)
    _style(
        ax,
        "Average absolute return by hour of day — intraday seasonality",
        "%",
        legend=False,
    )
    fig.suptitle(
        "Calendar / seasonality",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.01,
        ha="left",
    )
    _save(fig, "calendar.png")


FIGURES = {
    "momentum": momentum_fig,
    "trend": trend_fig,
    "volatility": volatility_fig,
    "volume": volume_fig,
    "returns": returns_fig,
    "quant": quant_fig,
    "microstructure": microstructure_fig,
    "calendar": calendar_fig,
}


def main() -> None:
    df = pl.read_ipc(FIXTURE)
    x = list(range(df.height))
    for name, fn in FIGURES.items():
        print(f"=== {name} ===")
        fn(df, x)
    print(f"\nAll family figures written to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
