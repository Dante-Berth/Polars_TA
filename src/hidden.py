import math
import polars as pl


def garman_klass_volatility(
    open_price: str, high: str, low: str, close: str, window: int = 20
) -> pl.Expr:
    o, h, l, c = pl.col(open_price), pl.col(high), pl.col(low), pl.col(close)
    log_hl = (h / l).log().pow(2)
    log_co = (c / o).log().pow(2)
    gk_variance = (0.5 * log_hl) - ((2 * math.log(2) - 1) * log_co)
    return gk_variance.rolling_mean(window_size=window).sqrt()


def rolling_z_score(close: str, window: int = 20) -> pl.Expr:
    c = pl.col(close)
    rolling_mean = c.rolling_mean(window_size=window)
    rolling_std = c.rolling_std(window_size=window)
    safe_std = pl.when(rolling_std == 0).then(1.0).otherwise(rolling_std)
    return (c - rolling_mean) / safe_std


def vol_adjusted_momentum(close: str, window: int = 20) -> pl.Expr:
    c = pl.col(close)
    ret = (c / c.shift(window)) - 1.0
    daily_ret = c.pct_change()
    vol = daily_ret.rolling_std(window_size=window)
    safe_vol = pl.when(vol == 0).then(1.0).otherwise(vol)
    return ret / safe_vol


def micro_price_proxy(high: str, low: str, close: str, volume: str) -> pl.Expr:
    h, l, c, v = pl.col(high), pl.col(low), pl.col(close), pl.col(volume)
    typical_price = (h + l + c) / 3.0
    log_v = pl.when(v <= 1).then(1.0).otherwise(v.log())
    return typical_price * log_v


def rolling_sharpe_ratio(
    close: str, window: int = 63, risk_free_rate: float = 0.0
) -> pl.Expr:
    """Annualized Rolling Sharpe Ratio (Assuming 252 trading days)"""
    ret = pl.col(close).pct_change()

    # Assuming daily data, we annualize by multiplying by sqrt(252)
    rolling_mean = ret.rolling_mean(window_size=window) - (risk_free_rate / 252)
    rolling_std = ret.rolling_std(window_size=window)

    safe_std = pl.when(rolling_std == 0).then(1.0).otherwise(rolling_std)
    sharpe = (rolling_mean / safe_std) * math.sqrt(252)

    return sharpe.alias(f"sharpe_{window}")


def rolling_sortino_ratio(
    close: str, window: int = 63, risk_free_rate: float = 0.0
) -> pl.Expr:
    """Annualized Rolling Sortino Ratio (Penalizes only downside volatility)"""
    ret = pl.col(close).pct_change()

    downside_ret = pl.when(ret < 0).then(ret).otherwise(0.0)

    rolling_mean = ret.rolling_mean(window_size=window) - (risk_free_rate / 252)
    downside_std = downside_ret.rolling_std(window_size=window)

    safe_down_std = pl.when(downside_std == 0).then(1.0).otherwise(downside_std)
    sortino = (rolling_mean / safe_down_std) * math.sqrt(252)

    return sortino.alias(f"sortino_{window}")


def historical_volatility(close: str, window: int = 21) -> pl.Expr:
    """Annualized Historical Volatility (Close-to-Close)"""
    c = pl.col(close)
    log_ret = (c / c.shift(1)).log()

    # Standard deviation of log returns * sqrt(252 trading days)
    hv = log_ret.rolling_std(window_size=window) * math.sqrt(252)
    return hv.alias(f"hist_vol_{window}")


def amihud_illiquidity(close: str, volume: str, window: int = 21) -> pl.Expr:
    """Rolling Amihud Illiquidity (Price impact per dollar traded)"""
    c = pl.col(close)
    v = pl.col(volume)

    daily_ret_abs = c.pct_change().abs()
    dollar_volume = c * v

    safe_dollar_vol = pl.when(dollar_volume == 0).then(None).otherwise(dollar_volume)

    # Ratio of absolute return to dollar volume, smoothed over a window
    amihud = (daily_ret_abs / safe_dollar_vol).rolling_mean(window_size=window)

    # Multiply by 10^6 to make the numbers human-readable (standard practice)
    return (amihud * 1_000_000).alias(f"amihud_illiq_{window}")
