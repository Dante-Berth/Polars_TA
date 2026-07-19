import typing as tp
import polars as pl
from polars_ta.utils import BaseIndicator


class VolumeIndicators:
    # ---------------------------------------------------------
    # Accumulation/Distribution Index (ADI)
    # ---------------------------------------------------------
    @staticmethod
    def acc_dist_index(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        volume: str | pl.Expr,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close, volume = (
            pl.col(high),
            pl.col(low),
            pl.col(close),
            pl.col(volume),
        )

        # Avoid division by zero
        hl_diff = high - low
        safe_hl_diff = pl.when(hl_diff == 0).then(1).otherwise(hl_diff)

        clv = ((close - low) - (high - close)) / safe_hl_diff
        adi = (clv * volume).cum_sum()

        return BaseIndicator.check_fillna(adi, fillna, value=0)

    # ---------------------------------------------------------
    # On-Balance Volume (OBV)
    # ---------------------------------------------------------
    @staticmethod
    def on_balance_volume(
        close: str | pl.Expr, volume: str | pl.Expr, fillna: bool = False
    ) -> pl.Expr:
        close, volume = pl.col(close), pl.col(volume)

        # Vectorized equivalent of np.where
        obv_step = pl.when(close < close.shift(1)).then(-volume).otherwise(volume)
        obv = obv_step.cum_sum()

        return BaseIndicator.check_fillna(obv, fillna, value=0)

    # ---------------------------------------------------------
    # Chaikin Money Flow (CMF)
    # ---------------------------------------------------------
    @staticmethod
    def chaikin_money_flow(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        volume: str | pl.Expr,
        window: int = 20,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close, volume = (
            pl.col(high),
            pl.col(low),
            pl.col(close),
            pl.col(volume),
        )
        min_periods = 1 if fillna else window

        hl_diff = high - low
        safe_hl_diff = pl.when(hl_diff == 0).then(1).otherwise(hl_diff)

        mfv = (((close - low) - (high - close)) / safe_hl_diff) * volume

        cmf = mfv.rolling_sum(
            window_size=window, min_periods=min_periods
        ) / volume.rolling_sum(window_size=window, min_periods=min_periods)
        return BaseIndicator.check_fillna(cmf, fillna, value=0)

    # ---------------------------------------------------------
    # Force Index (FI)
    # ---------------------------------------------------------
    @staticmethod
    def force_index(
        close: str | pl.Expr,
        volume: str | pl.Expr,
        window: int = 13,
        fillna: bool = False,
    ) -> pl.Expr:
        close, volume = pl.col(close), pl.col(volume)
        fi_series = (close - close.shift(1)) * volume
        fi = BaseIndicator.ema(fi_series, window, fillna)
        return BaseIndicator.check_fillna(fi, fillna, value=0)

    # ---------------------------------------------------------
    # Ease of Movement (EoM, EMV)
    # ---------------------------------------------------------
    @staticmethod
    def ease_of_movement(
        high: str | pl.Expr,
        low: str | pl.Expr,
        volume: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, volume = pl.col(high), pl.col(low), pl.col(volume)

        emv = ((high.diff() + low.diff()) * (high - low)) / (2 * volume)
        emv = emv * 100000000

        return BaseIndicator.check_fillna(emv, fillna, value=0)

    @staticmethod
    def sma_ease_of_movement(
        high: str | pl.Expr,
        low: str | pl.Expr,
        volume: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        emv = VolumeIndicators.ease_of_movement(high, low, volume, window, fillna)
        min_periods = 1 if fillna else window
        sma_emv = emv.rolling_mean(window_size=window, min_periods=min_periods)
        return BaseIndicator.check_fillna(sma_emv, fillna, value=0)

    # ---------------------------------------------------------
    # Volume-Price Trend (VPT)
    # ---------------------------------------------------------
    @staticmethod
    def volume_price_trend(
        close: str | pl.Expr,
        volume: str | pl.Expr,
        fillna: bool = False,
        smoothing_factor: tp.Optional[int] = None,
        dropnans: bool = False,
    ) -> pl.Expr:
        close, volume = pl.col(close), pl.col(volume)

        # pct_change is equivalent to (close / close.shift(1)) - 1
        pct_change = (close / close.shift(1)) - 1
        vpt = (pct_change * volume).cum_sum()

        if smoothing_factor:
            min_periods = 1 if fillna else smoothing_factor
            vpt = vpt.rolling_mean(
                window_size=smoothing_factor, min_periods=min_periods
            )

        # Note: dropnans is ignored here because Polars Expr must return the same length
        # as the DataFrame when used in .with_columns(). Drop nulls at the DataFrame level.
        return BaseIndicator.check_fillna(vpt, fillna, value=0)

    # ---------------------------------------------------------
    # Negative Volume Index (NVI)
    # ---------------------------------------------------------
    @staticmethod
    def negative_volume_index(
        close: str | pl.Expr, volume: str | pl.Expr, fillna: bool = False
    ) -> pl.Expr:
        close, volume = pl.col(close), pl.col(volume)

        pct_change = (close / close.shift(1)) - 1
        vol_decrease = volume < volume.shift(1)

        # Replace the iterative loop with a cumulative product
        multiplier = pl.when(vol_decrease).then(1.0 + pct_change).otherwise(1.0)
        nvi = 1000.0 * multiplier.cum_prod()

        return BaseIndicator.check_fillna(nvi, fillna, value=1000)

    # ---------------------------------------------------------
    # Money Flow Index (MFI)
    # ---------------------------------------------------------
    @staticmethod
    def money_flow_index(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        volume: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close, volume = (
            pl.col(high),
            pl.col(low),
            pl.col(close),
            pl.col(volume),
        )
        min_periods = 1 if fillna else window

        typical_price = (high + low + close) / 3.0

        up_down = (
            pl.when(typical_price > typical_price.shift(1))
            .then(1)
            .when(typical_price < typical_price.shift(1))
            .then(-1)
            .otherwise(0)
        )

        mfr = typical_price * volume * up_down

        # Pre-split positive and negative flows to use highly optimized rolling_sum
        pos_mf = pl.when(mfr > 0).then(mfr).otherwise(0.0)
        neg_mf = pl.when(mfr < 0).then(-mfr).otherwise(0.0)

        n_pos_mf = pos_mf.rolling_sum(window_size=window, min_periods=min_periods)
        n_neg_mf = neg_mf.rolling_sum(window_size=window, min_periods=min_periods)

        # Protect against division by zero
        safe_neg_mf = pl.when(n_neg_mf == 0).then(1.0).otherwise(n_neg_mf)

        mfi_ratio = n_pos_mf / safe_neg_mf
        mfi = 100.0 - (100.0 / (1.0 + mfi_ratio))

        # If neg_mf was 0, MFI should be 100
        mfi = pl.when(n_neg_mf == 0).then(100.0).otherwise(mfi)

        return BaseIndicator.check_fillna(mfi, fillna, value=50)

    # ---------------------------------------------------------
    # Volume Weighted Average Price (VWAP)
    # ---------------------------------------------------------
    @staticmethod
    def volume_weighted_average_price(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        volume: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close, volume = (
            pl.col(high),
            pl.col(low),
            pl.col(close),
            pl.col(volume),
        )
        min_periods = 1 if fillna else window

        typical_price = (high + low + close) / 3.0
        typical_price_volume = typical_price * volume

        total_pv = typical_price_volume.rolling_sum(
            window_size=window, min_periods=min_periods
        )
        total_volume = volume.rolling_sum(window_size=window, min_periods=min_periods)

        vwap = total_pv / total_volume
        return BaseIndicator.check_fillna(vwap, fillna, value=0)


# ==============================================================================
# TOP-LEVEL API WRAPPERS
# ==============================================================================


def acc_dist_index(high, low, close, volume, fillna=False) -> pl.Expr:
    return VolumeIndicators.acc_dist_index(high, low, close, volume, fillna)


def on_balance_volume(close, volume, fillna=False) -> pl.Expr:
    return VolumeIndicators.on_balance_volume(close, volume, fillna)


def chaikin_money_flow(high, low, close, volume, window=20, fillna=False) -> pl.Expr:
    return VolumeIndicators.chaikin_money_flow(high, low, close, volume, window, fillna)


def force_index(close, volume, window=13, fillna=False) -> pl.Expr:
    return VolumeIndicators.force_index(close, volume, window, fillna)


def ease_of_movement(high, low, volume, window=14, fillna=False) -> pl.Expr:
    return VolumeIndicators.ease_of_movement(high, low, volume, window, fillna)


def sma_ease_of_movement(high, low, volume, window=14, fillna=False) -> pl.Expr:
    return VolumeIndicators.sma_ease_of_movement(high, low, volume, window, fillna)


def volume_price_trend(
    close,
    volume,
    fillna=False,
    smoothing_factor: tp.Optional[int] = None,
    dropnans: bool = False,
) -> pl.Expr:
    return VolumeIndicators.volume_price_trend(
        close, volume, fillna, smoothing_factor, dropnans
    )


def negative_volume_index(close, volume, fillna=False) -> pl.Expr:
    return VolumeIndicators.negative_volume_index(close, volume, fillna)


def money_flow_index(high, low, close, volume, window=14, fillna=False) -> pl.Expr:
    return VolumeIndicators.money_flow_index(high, low, close, volume, window, fillna)


def volume_weighted_average_price(
    high, low, close, volume, window=14, fillna=False
) -> pl.Expr:
    return VolumeIndicators.volume_weighted_average_price(
        high, low, close, volume, window, fillna
    )
