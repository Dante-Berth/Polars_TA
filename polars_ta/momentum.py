import numpy as np
import polars as pl

from polars_ta._internal import as_expr
from polars_ta.utils import BaseIndicator

# (Assuming BaseIndicator is already defined from our previous steps)


class MomentumIndicators:
    # ---------------------------------------------------------
    # Relative Strength Index (RSI)
    # ---------------------------------------------------------
    @staticmethod
    def rsi(close: str | pl.Expr, window: int = 14, fillna: bool = False) -> pl.Expr:
        close = as_expr(close)
        min_periods = 1 if fillna else window

        diff = close.diff(1)
        up_direction = pl.when(diff > 0).then(diff).otherwise(0.0)
        down_direction = pl.when(diff < 0).then(-diff).otherwise(0.0)

        # RSI typically uses Wilder's smoothing (alpha = 1 / window)
        emaup = up_direction.ewm_mean(
            alpha=1.0 / window, adjust=False, min_samples=min_periods
        )
        emadn = down_direction.ewm_mean(
            alpha=1.0 / window, adjust=False, min_samples=min_periods
        )

        relative_strength = emaup / emadn
        rsi_val = (
            pl.when(emadn == 0)
            .then(100.0)
            .otherwise(100.0 - (100.0 / (1.0 + relative_strength)))
        )

        return BaseIndicator.check_fillna(rsi_val, fillna, value=50)

    # ---------------------------------------------------------
    # True Strength Index (TSI)
    # ---------------------------------------------------------
    @staticmethod
    def tsi(
        close: str | pl.Expr,
        window_slow: int = 25,
        window_fast: int = 13,
        fillna: bool = False,
    ) -> pl.Expr:
        close = as_expr(close)
        min_periods_r = 1 if fillna else window_slow
        min_periods_s = 1 if fillna else window_fast

        diff_close = close.diff(1)

        smoothed = diff_close.ewm_mean(
            span=window_slow, adjust=False, min_samples=min_periods_r
        ).ewm_mean(span=window_fast, adjust=False, min_samples=min_periods_s)

        smoothed_abs = (
            diff_close.abs()
            .ewm_mean(span=window_slow, adjust=False, min_samples=min_periods_r)
            .ewm_mean(span=window_fast, adjust=False, min_samples=min_periods_s)
        )

        tsi_val = (smoothed / smoothed_abs) * 100.0
        return BaseIndicator.check_fillna(tsi_val, fillna, value=0)

    # ---------------------------------------------------------
    # Ultimate Oscillator (UO)
    # ---------------------------------------------------------
    @staticmethod
    def ultimate_oscillator(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window1: int = 7,
        window2: int = 14,
        window3: int = 28,
        weight1: float = 4.0,
        weight2: float = 2.0,
        weight3: float = 1.0,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close = (
            as_expr(high),
            as_expr(low),
            as_expr(close),
        )

        prev_close = close.shift(1)
        true_range = BaseIndicator.true_range(high, low, prev_close)

        min_low_or_pc = pl.min_horizontal([low, prev_close])
        buying_pressure = close - min_low_or_pc

        avg_s = buying_pressure.rolling_sum(
            window_size=window1, min_samples=1 if fillna else window1
        ) / true_range.rolling_sum(
            window_size=window1, min_samples=1 if fillna else window1
        )

        avg_m = buying_pressure.rolling_sum(
            window_size=window2, min_samples=1 if fillna else window2
        ) / true_range.rolling_sum(
            window_size=window2, min_samples=1 if fillna else window2
        )

        avg_l = buying_pressure.rolling_sum(
            window_size=window3, min_samples=1 if fillna else window3
        ) / true_range.rolling_sum(
            window_size=window3, min_samples=1 if fillna else window3
        )

        uo = (
            100.0
            * ((weight1 * avg_s) + (weight2 * avg_m) + (weight3 * avg_l))
            / (weight1 + weight2 + weight3)
        )
        return BaseIndicator.check_fillna(uo, fillna, value=50)

    # ---------------------------------------------------------
    # Stochastic Oscillator
    # ---------------------------------------------------------
    @staticmethod
    def stoch(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close = (
            as_expr(high),
            as_expr(low),
            as_expr(close),
        )
        min_periods = 1 if fillna else window

        smin = low.rolling_min(window_size=window, min_samples=min_periods)
        smax = high.rolling_max(window_size=window, min_samples=min_periods)

        stoch_k = 100.0 * (close - smin) / (smax - smin)
        return BaseIndicator.check_fillna(stoch_k, fillna, value=50)

    @staticmethod
    def stoch_signal(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        smooth_window: int = 3,
        fillna: bool = False,
    ) -> pl.Expr:
        stoch_k = MomentumIndicators.stoch(high, low, close, window, fillna)
        stoch_d = stoch_k.rolling_mean(
            window_size=smooth_window, min_samples=1 if fillna else smooth_window
        )
        return BaseIndicator.check_fillna(stoch_d, fillna, value=50)

    # ---------------------------------------------------------
    # Kaufman's Adaptive Moving Average (KAMA)
    # ---------------------------------------------------------
    @staticmethod
    def kama(
        close: str | pl.Expr,
        window: int = 10,
        pow1: int = 2,
        pow2: int = 30,
        fillna: bool = False,
    ) -> pl.Expr:
        close = as_expr(close)
        min_periods = 1 if fillna else window

        # 1. Calculate the Dynamic Smoothing Constant in Polars
        vol = (close - close.shift(1)).abs()
        er_num = (close - close.shift(window)).abs()
        er_den = vol.rolling_sum(window_size=window, min_samples=min_periods)

        # Guard only the true zero-denominator case; a *null* denominator means
        # "warm-up, not enough data yet" and must stay null (an `otherwise(0.0)`
        # here would fabricate a smoothing constant and start the KAMA
        # recursion at bar 0 with no history).
        efficiency_ratio = pl.when(er_den == 0).then(0.0).otherwise(er_num / er_den)

        fast_alpha = 2.0 / (pow1 + 1.0)
        slow_alpha = 2.0 / (pow2 + 1.0)
        smoothing_constant = (
            efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha
        ).pow(2)

        # 2. Iterate the recursive KAMA calculation via NumPy in map_batches
        def _calc_kama(struct_s: pl.Series) -> pl.Series:
            df = struct_s.struct.unnest()
            c_arr = df["close"].to_numpy()
            sc_arr = df["sc"].to_numpy()
            n = len(c_arr)

            kama = np.full(n, np.nan)
            first_value = True

            for i in range(n):
                if np.isnan(sc_arr[i]) or sc_arr[i] is None:
                    continue
                if first_value:
                    kama[i] = c_arr[i]
                    first_value = False
                else:
                    kama[i] = kama[i - 1] + sc_arr[i] * (c_arr[i] - kama[i - 1])

            # Warm-up rows are *null* ("not enough data yet"), never NaN.
            return pl.Series(kama).fill_nan(None)

        expr = pl.struct(
            [close.alias("close"), smoothing_constant.alias("sc")]
        ).map_batches(_calc_kama, returns_scalar=False)
        # Note: ta lib fills KAMA na values with the close price
        kama_val = (
            pl.when(expr.is_null() | expr.is_nan()).then(close).otherwise(expr)
            if fillna
            else expr
        )
        return kama_val

    # ---------------------------------------------------------
    # Rate of Change (ROC)
    # ---------------------------------------------------------
    @staticmethod
    def roc(close: str | pl.Expr, window: int = 12, fillna: bool = False) -> pl.Expr:
        close = as_expr(close)
        roc_val = ((close - close.shift(window)) / close.shift(window)) * 100.0
        return BaseIndicator.check_fillna(roc_val, fillna, value=0)

    # ---------------------------------------------------------
    # Awesome Oscillator (AO)
    # ---------------------------------------------------------
    @staticmethod
    def awesome_oscillator(
        high: str | pl.Expr,
        low: str | pl.Expr,
        window1: int = 5,
        window2: int = 34,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low = (
            as_expr(high),
            as_expr(low),
        )
        median_price = 0.5 * (high + low)

        sma_fast = median_price.rolling_mean(
            window_size=window1, min_samples=1 if fillna else window1
        )
        sma_slow = median_price.rolling_mean(
            window_size=window2, min_samples=1 if fillna else window2
        )

        ao = sma_fast - sma_slow
        return BaseIndicator.check_fillna(ao, fillna, value=0)

    # ---------------------------------------------------------
    # Williams %R
    # ---------------------------------------------------------
    @staticmethod
    def williams_r(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        lbp: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close = (
            as_expr(high),
            as_expr(low),
            as_expr(close),
        )
        min_periods = 1 if fillna else lbp

        highest_high = high.rolling_max(window_size=lbp, min_samples=min_periods)
        lowest_low = low.rolling_min(window_size=lbp, min_samples=min_periods)

        wr = -100.0 * (highest_high - close) / (highest_high - lowest_low)
        return BaseIndicator.check_fillna(wr, fillna, value=-50)

    # ---------------------------------------------------------
    # Stochastic RSI
    # ---------------------------------------------------------
    @staticmethod
    def stochrsi(
        close: str | pl.Expr, window: int = 14, fillna: bool = False
    ) -> pl.Expr:
        close = as_expr(close)
        rsi_val = MomentumIndicators.rsi(close, window, fillna)

        lowest_low_rsi = rsi_val.rolling_min(
            window_size=window, min_samples=1 if fillna else window
        )
        highest_high_rsi = rsi_val.rolling_max(
            window_size=window, min_samples=1 if fillna else window
        )

        stochrsi_val = (rsi_val - lowest_low_rsi) / (highest_high_rsi - lowest_low_rsi)
        return BaseIndicator.check_fillna(stochrsi_val, fillna, value=0)

    @staticmethod
    def stochrsi_k(
        close: str | pl.Expr, window: int = 14, smooth1: int = 3, fillna: bool = False
    ) -> pl.Expr:
        stochrsi_val = MomentumIndicators.stochrsi(close, window, fillna)
        stochrsi_k_val = stochrsi_val.rolling_mean(
            window_size=smooth1, min_samples=1 if fillna else smooth1
        )
        return BaseIndicator.check_fillna(stochrsi_k_val, fillna, value=0)

    @staticmethod
    def stochrsi_d(
        close: str | pl.Expr,
        window: int = 14,
        smooth1: int = 3,
        smooth2: int = 3,
        fillna: bool = False,
    ) -> pl.Expr:
        stochrsi_k_val = MomentumIndicators.stochrsi_k(close, window, smooth1, fillna)
        stochrsi_d_val = stochrsi_k_val.rolling_mean(
            window_size=smooth2, min_samples=1 if fillna else smooth2
        )
        return BaseIndicator.check_fillna(stochrsi_d_val, fillna, value=0)

    # ---------------------------------------------------------
    # Percentage Price Oscillator (PPO)
    # ---------------------------------------------------------
    @staticmethod
    def ppo(
        close: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        fillna: bool = False,
    ) -> pl.Expr:
        close = as_expr(close)
        ema_fast = BaseIndicator.ema(close, window_fast, fillna)
        ema_slow = BaseIndicator.ema(close, window_slow, fillna)

        ppo_val = ((ema_fast - ema_slow) / ema_slow) * 100.0
        return BaseIndicator.check_fillna(ppo_val, fillna, value=0)

    @staticmethod
    def ppo_signal(
        close: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        ppo_val = MomentumIndicators.ppo(close, window_slow, window_fast, fillna)
        ppo_sig = BaseIndicator.ema(ppo_val, window_sign, fillna)
        return BaseIndicator.check_fillna(ppo_sig, fillna, value=0)

    @staticmethod
    def ppo_hist(
        close: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        ppo_val = MomentumIndicators.ppo(close, window_slow, window_fast, fillna)
        ppo_sig = MomentumIndicators.ppo_signal(
            close, window_slow, window_fast, window_sign, fillna
        )
        ppo_hist_val = ppo_val - ppo_sig
        return BaseIndicator.check_fillna(ppo_hist_val, fillna, value=0)

    # ---------------------------------------------------------
    # Percentage Volume Oscillator (PVO)
    # ---------------------------------------------------------
    @staticmethod
    def pvo(
        volume: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        fillna: bool = False,
    ) -> pl.Expr:
        volume = as_expr(volume)
        ema_fast = BaseIndicator.ema(volume, window_fast, fillna)
        ema_slow = BaseIndicator.ema(volume, window_slow, fillna)

        pvo_val = ((ema_fast - ema_slow) / ema_slow) * 100.0
        return BaseIndicator.check_fillna(pvo_val, fillna, value=0)

    @staticmethod
    def pvo_signal(
        volume: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        pvo_val = MomentumIndicators.pvo(volume, window_slow, window_fast, fillna)
        pvo_sig = BaseIndicator.ema(pvo_val, window_sign, fillna)
        return BaseIndicator.check_fillna(pvo_sig, fillna, value=0)

    @staticmethod
    def pvo_hist(
        volume: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        pvo_val = MomentumIndicators.pvo(volume, window_slow, window_fast, fillna)
        pvo_sig = MomentumIndicators.pvo_signal(
            volume, window_slow, window_fast, window_sign, fillna
        )
        pvo_hist_val = pvo_val - pvo_sig
        return BaseIndicator.check_fillna(pvo_hist_val, fillna, value=0)

    # ---------------------------------------------------------
    # Chande Momentum Oscillator (CMO)
    # ---------------------------------------------------------
    @staticmethod
    def cmo(close: str | pl.Expr, window: int = 14, fillna: bool = False) -> pl.Expr:
        """Chande Momentum Oscillator: 100 * (sum(up) - sum(down)) / (sum(up)
        + sum(down)) over the window, unlike RSI's smoothed averages."""
        close = as_expr(close)
        min_periods = 1 if fillna else window

        diff = close.diff(1)
        up = pl.when(diff > 0).then(diff).otherwise(0.0)
        down = pl.when(diff < 0).then(-diff).otherwise(0.0)

        sum_up = up.rolling_sum(window_size=window, min_samples=min_periods)
        sum_down = down.rolling_sum(window_size=window, min_samples=min_periods)
        total = sum_up + sum_down

        cmo = pl.when(total == 0).then(0.0).otherwise(100 * (sum_up - sum_down) / total)
        return BaseIndicator.check_fillna(cmo, fillna, value=0)

    # ---------------------------------------------------------
    # Fisher Transform
    # ---------------------------------------------------------
    @staticmethod
    def fisher_transform(
        high: str | pl.Expr,
        low: str | pl.Expr,
        window: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        """Ehlers' Fisher Transform: maps a bounded price-position oscillator
        through `atanh` to produce sharper, more Gaussian-distributed turning
        points than the underlying stochastic-style oscillator.

        Ehlers' original formula EMA-smooths the normalized price position
        (`value = 0.33 * 2*(...) + 0.67 * value[1]`) *before* the `atanh`
        step, and smooths the Fisher output itself the same way. Skipping
        that damping — feeding the raw, unsmoothed position straight into
        `atanh` — saturates the output near the clip boundary on real,
        noisy data (price sits at the rolling high/low far more often than
        the idealized derivation assumes), producing a square-wave-like
        series instead of the intended smooth oscillator. `map_batches`
        carries the two damped recursions; everything upstream is vectorized.
        """
        high = as_expr(high)
        low = as_expr(low)
        min_periods = 1 if fillna else window

        hl2 = (high + low) / 2.0
        lowest = hl2.rolling_min(window_size=window, min_samples=min_periods)
        highest = hl2.rolling_max(window_size=window, min_samples=min_periods)

        price_range = highest - lowest
        raw = (
            pl.when(price_range == 0)
            .then(0.0)
            .otherwise(2.0 * (hl2 - lowest) / price_range - 1.0)
        )

        def _calc_fisher(s: pl.Series) -> pl.Series:
            raw_arr = s.to_numpy()
            n = len(raw_arr)
            value = 0.0
            fish = 0.0
            out = np.full(n, np.nan)
            for i in range(n):
                r = raw_arr[i]
                if np.isnan(r):
                    continue
                value = 0.33 * r + 0.67 * value
                value = min(max(value, -0.999), 0.999)
                fish = 0.5 * np.log((1 + value) / (1 - value)) + 0.5 * fish
                out[i] = fish
            return pl.Series(out).fill_nan(None)

        fisher = raw.map_batches(_calc_fisher, returns_scalar=False)
        return BaseIndicator.check_fillna(fisher, fillna, value=0)


# ==============================================================================
# TOP-LEVEL API WRAPPERS
# ==============================================================================


def rsi(close, window=14, fillna=False) -> pl.Expr:
    return MomentumIndicators.rsi(close, window, fillna)


def tsi(close, window_slow=25, window_fast=13, fillna=False) -> pl.Expr:
    return MomentumIndicators.tsi(close, window_slow, window_fast, fillna)


def ultimate_oscillator(
    high,
    low,
    close,
    window1=7,
    window2=14,
    window3=28,
    weight1=4.0,
    weight2=2.0,
    weight3=1.0,
    fillna=False,
) -> pl.Expr:
    return MomentumIndicators.ultimate_oscillator(
        high, low, close, window1, window2, window3, weight1, weight2, weight3, fillna
    )


def stoch(high, low, close, window=14, smooth_window=3, fillna=False) -> pl.Expr:
    return MomentumIndicators.stoch(high, low, close, window, fillna)


def stoch_signal(high, low, close, window=14, smooth_window=3, fillna=False) -> pl.Expr:
    return MomentumIndicators.stoch_signal(
        high, low, close, window, smooth_window, fillna
    )


def williams_r(high, low, close, lbp=14, fillna=False) -> pl.Expr:
    return MomentumIndicators.williams_r(high, low, close, lbp, fillna)


def awesome_oscillator(high, low, window1=5, window2=34, fillna=False) -> pl.Expr:
    return MomentumIndicators.awesome_oscillator(high, low, window1, window2, fillna)


def kama(close, window=10, pow1=2, pow2=30, fillna=False) -> pl.Expr:
    return MomentumIndicators.kama(close, window, pow1, pow2, fillna)


def roc(close, window=12, fillna=False) -> pl.Expr:
    return MomentumIndicators.roc(close, window, fillna)


def stochrsi(close, window=14, smooth1=3, smooth2=3, fillna=False) -> pl.Expr:
    return MomentumIndicators.stochrsi(close, window, fillna)


def stochrsi_k(close, window=14, smooth1=3, smooth2=3, fillna=False) -> pl.Expr:
    return MomentumIndicators.stochrsi_k(close, window, smooth1, fillna)


def stochrsi_d(close, window=14, smooth1=3, smooth2=3, fillna=False) -> pl.Expr:
    return MomentumIndicators.stochrsi_d(close, window, smooth1, smooth2, fillna)


def ppo(close, window_slow=26, window_fast=12, window_sign=9, fillna=False) -> pl.Expr:
    return MomentumIndicators.ppo(close, window_slow, window_fast, fillna)


def ppo_signal(
    close, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pl.Expr:
    return MomentumIndicators.ppo_signal(
        close, window_slow, window_fast, window_sign, fillna
    )


def ppo_hist(
    close, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pl.Expr:
    return MomentumIndicators.ppo_hist(
        close, window_slow, window_fast, window_sign, fillna
    )


def pvo(volume, window_slow=26, window_fast=12, window_sign=9, fillna=False) -> pl.Expr:
    return MomentumIndicators.pvo(volume, window_slow, window_fast, fillna)


def pvo_signal(
    volume, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pl.Expr:
    return MomentumIndicators.pvo_signal(
        volume, window_slow, window_fast, window_sign, fillna
    )


def pvo_hist(
    volume, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pl.Expr:
    return MomentumIndicators.pvo_hist(
        volume, window_slow, window_fast, window_sign, fillna
    )


def cmo(close, window=14, fillna=False) -> pl.Expr:
    return MomentumIndicators.cmo(close, window, fillna)


def fisher_transform(high, low, window=9, fillna=False) -> pl.Expr:
    return MomentumIndicators.fisher_transform(high, low, window, fillna)
