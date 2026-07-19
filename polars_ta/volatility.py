import polars as pl

from polars_ta.utils import BaseIndicator


class VolatilityIndicators:
    # ---------------------------------------------------------
    # Average True Range (ATR)
    # ---------------------------------------------------------
    @staticmethod
    def average_true_range(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """Average True Range (ATR)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close

        min_periods = 1 if fillna else window

        close_shift = close.shift(1)
        true_range = BaseIndicator.true_range(high, low, close_shift)

        # Wilder's smoothing is equivalent to an EMA with alpha = 1 / window
        atr = true_range.ewm_mean(
            alpha=1.0 / window, adjust=False, min_samples=min_periods
        )
        return BaseIndicator.check_fillna(atr, fillna, value=0)

    # ---------------------------------------------------------
    # Bollinger Bands (BB)
    # ---------------------------------------------------------
    @staticmethod
    def _bb_components(
        close: pl.Expr, window: int, window_dev: int, fillna: bool
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        min_periods = 1 if fillna else window
        mavg = close.rolling_mean(window_size=window, min_samples=min_periods)
        # Note: ta uses ddof=0 for its standard deviation
        mstd = close.rolling_std(window_size=window, min_samples=min_periods, ddof=0)
        hband = mavg + (window_dev * mstd)
        lband = mavg - (window_dev * mstd)
        return mavg, hband, lband

    @staticmethod
    def bollinger_mavg(
        close: str | pl.Expr, window: int = 20, fillna: bool = False
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        mavg, _, _ = VolatilityIndicators._bb_components(close, window, 2, fillna)
        return BaseIndicator.check_fillna(mavg, fillna, value=-1)

    @staticmethod
    def bollinger_hband(
        close: str | pl.Expr,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        _, hband, _ = VolatilityIndicators._bb_components(
            close, window, window_dev, fillna
        )
        return BaseIndicator.check_fillna(hband, fillna, value=-1)

    @staticmethod
    def bollinger_lband(
        close: str | pl.Expr,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        _, _, lband = VolatilityIndicators._bb_components(
            close, window, window_dev, fillna
        )
        return BaseIndicator.check_fillna(lband, fillna, value=-1)

    @staticmethod
    def bollinger_wband(
        close: str | pl.Expr,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        mavg, hband, lband = VolatilityIndicators._bb_components(
            close, window, window_dev, fillna
        )
        wband = ((hband - lband) / mavg) * 100
        return BaseIndicator.check_fillna(wband, fillna, value=0)

    @staticmethod
    def bollinger_pband(
        close: str | pl.Expr,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        _, hband, lband = VolatilityIndicators._bb_components(
            close, window, window_dev, fillna
        )

        band_diff = hband - lband
        safe_diff = pl.when(band_diff == 0).then(None).otherwise(band_diff)
        pband = (close - lband) / safe_diff
        return BaseIndicator.check_fillna(pband, fillna, value=0)

    @staticmethod
    def bollinger_hband_indicator(
        close: str | pl.Expr,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        _, hband, _ = VolatilityIndicators._bb_components(
            close, window, window_dev, fillna
        )
        indicator = pl.when(close > hband).then(1.0).otherwise(0.0)
        return BaseIndicator.check_fillna(indicator, fillna, value=0)

    @staticmethod
    def bollinger_lband_indicator(
        close: str | pl.Expr,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ) -> pl.Expr:
        close = pl.col(close) if isinstance(close, str) else close
        _, _, lband = VolatilityIndicators._bb_components(
            close, window, window_dev, fillna
        )
        indicator = pl.when(close < lband).then(1.0).otherwise(0.0)
        return BaseIndicator.check_fillna(indicator, fillna, value=0)

    # ---------------------------------------------------------
    # Keltner Channel (KC)
    # ---------------------------------------------------------
    @staticmethod
    def _kc_components(
        high: pl.Expr,
        low: pl.Expr,
        close: pl.Expr,
        window: int,
        window_atr: int,
        original_version: bool,
        multiplier: int,
        fillna: bool,
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        min_periods = 1 if fillna else window

        if original_version:
            tp = ((high + low + close) / 3.0).rolling_mean(
                window_size=window, min_samples=min_periods
            )
            tp_high = (((4 * high) - (2 * low) + close) / 3.0).rolling_mean(
                window_size=window, min_samples=1
            )
            tp_low = (((-2 * high) + (4 * low) + close) / 3.0).rolling_mean(
                window_size=window, min_samples=1
            )
        else:
            tp = close.ewm_mean(span=window, adjust=False, min_samples=min_periods)
            atr = VolatilityIndicators.average_true_range(
                high, low, close, window_atr, fillna
            )
            tp_high = tp + (multiplier * atr)
            tp_low = tp - (multiplier * atr)

        return tp, tp_high, tp_low

    @staticmethod
    def keltner_channel_mband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        tp, _, _ = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        return BaseIndicator.check_fillna(tp, fillna, value=-1)

    @staticmethod
    def keltner_channel_hband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        _, tp_high, _ = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        return BaseIndicator.check_fillna(tp_high, fillna, value=-1)

    @staticmethod
    def keltner_channel_lband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        _, _, tp_low = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        return BaseIndicator.check_fillna(tp_low, fillna, value=-1)

    @staticmethod
    def keltner_channel_wband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        tp, tp_high, tp_low = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        wband = ((tp_high - tp_low) / tp) * 100
        return BaseIndicator.check_fillna(wband, fillna, value=0)

    @staticmethod
    def keltner_channel_pband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        _, tp_high, tp_low = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        pband = (close - tp_low) / (tp_high - tp_low)
        return BaseIndicator.check_fillna(pband, fillna, value=0)

    @staticmethod
    def keltner_channel_hband_indicator(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        _, tp_high, _ = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        indicator = pl.when(close > tp_high).then(1.0).otherwise(0.0)
        return BaseIndicator.check_fillna(indicator, fillna, value=0)

    @staticmethod
    def keltner_channel_lband_indicator(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
        multiplier: int = 2,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        _, _, tp_low = VolatilityIndicators._kc_components(
            high, low, close, window, window_atr, original_version, multiplier, fillna
        )
        indicator = pl.when(close < tp_low).then(1.0).otherwise(0.0)
        return BaseIndicator.check_fillna(indicator, fillna, value=0)

    # ---------------------------------------------------------
    # Donchian Channel (DC)
    # ---------------------------------------------------------
    @staticmethod
    def _dc_components(
        high: pl.Expr, low: pl.Expr, window: int, fillna: bool
    ) -> tuple[pl.Expr, pl.Expr]:
        min_periods = 1 if fillna else window
        hband = high.rolling_max(window_size=window, min_samples=min_periods)
        lband = low.rolling_min(window_size=window, min_samples=min_periods)
        return hband, lband

    @staticmethod
    def donchian_channel_hband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        offset: int = 0,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low = pl.col(high), pl.col(low)
        hband, _ = VolatilityIndicators._dc_components(high, low, window, fillna)
        hband = BaseIndicator.check_fillna(hband, fillna, value=-1)
        return hband.shift(offset) if offset != 0 else hband

    @staticmethod
    def donchian_channel_lband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        offset: int = 0,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low = pl.col(high), pl.col(low)
        _, lband = VolatilityIndicators._dc_components(high, low, window, fillna)
        lband = BaseIndicator.check_fillna(lband, fillna, value=-1)
        return lband.shift(offset) if offset != 0 else lband

    @staticmethod
    def donchian_channel_mband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 10,
        offset: int = 0,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low = pl.col(high), pl.col(low)
        hband, lband = VolatilityIndicators._dc_components(high, low, window, fillna)
        mband = ((hband - lband) / 2.0) + lband
        mband = BaseIndicator.check_fillna(mband, fillna, value=-1)
        return mband.shift(offset) if offset != 0 else mband

    @staticmethod
    def donchian_channel_wband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 10,
        offset: int = 0,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        hband, lband = VolatilityIndicators._dc_components(high, low, window, fillna)
        mavg = close.rolling_mean(
            window_size=window, min_samples=1 if fillna else window
        )
        wband = ((hband - lband) / mavg) * 100
        wband = BaseIndicator.check_fillna(wband, fillna, value=0)
        return wband.shift(offset) if offset != 0 else wband

    @staticmethod
    def donchian_channel_pband(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 10,
        offset: int = 0,
        fillna: bool = False,
    ) -> pl.Expr:
        high, low, close = pl.col(high), pl.col(low), pl.col(close)
        hband, lband = VolatilityIndicators._dc_components(high, low, window, fillna)
        pband = (close - lband) / (hband - lband)
        pband = BaseIndicator.check_fillna(pband, fillna, value=0)
        return pband.shift(offset) if offset != 0 else pband

    # ---------------------------------------------------------
    # Ulcer Index (UI)
    # ---------------------------------------------------------
    @staticmethod
    def ulcer_index(
        close: str | pl.Expr, window: int = 14, fillna: bool = False
    ) -> pl.Expr:
        """Ulcer Index (UI) - fully vectorized!"""
        close = pl.col(close) if isinstance(close, str) else close

        ui_max = close.rolling_max(window_size=window, min_samples=1)
        r_i = 100 * (close - ui_max) / ui_max

        # Instead of a slow python function applied across a rolling window,
        # we can mathematically vectorize the root-mean-square
        r_i_squared = r_i.pow(2)
        ulcer_idx = (
            r_i_squared.rolling_sum(window_size=window, min_samples=1) / window
        ).sqrt()

        return BaseIndicator.check_fillna(ulcer_idx, fillna, value=0)


# ==============================================================================
# TOP-LEVEL API WRAPPERS
# ==============================================================================


def average_true_range(high, low, close, window=14, fillna=False) -> pl.Expr:
    return VolatilityIndicators.average_true_range(high, low, close, window, fillna)


def bollinger_mavg(close, window=20, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_mavg(close, window, fillna)


def bollinger_hband(close, window=20, window_dev=2, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_hband(close, window, window_dev, fillna)


def bollinger_lband(close, window=20, window_dev=2, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_lband(close, window, window_dev, fillna)


def bollinger_wband(close, window=20, window_dev=2, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_wband(close, window, window_dev, fillna)


def bollinger_pband(close, window=20, window_dev=2, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_pband(close, window, window_dev, fillna)


def bollinger_hband_indicator(close, window=20, window_dev=2, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_hband_indicator(
        close, window, window_dev, fillna
    )


def bollinger_lband_indicator(close, window=20, window_dev=2, fillna=False) -> pl.Expr:
    return VolatilityIndicators.bollinger_lband_indicator(
        close, window, window_dev, fillna
    )


def keltner_channel_mband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_mband(
        high, low, close, window, window_atr, fillna, original_version
    )


def keltner_channel_hband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_hband(
        high, low, close, window, window_atr, fillna, original_version
    )


def keltner_channel_lband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_lband(
        high, low, close, window, window_atr, fillna, original_version
    )


def keltner_channel_wband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_wband(
        high, low, close, window, window_atr, fillna, original_version
    )


def keltner_channel_pband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_pband(
        high, low, close, window, window_atr, fillna, original_version
    )


def keltner_channel_hband_indicator(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_hband_indicator(
        high, low, close, window, window_atr, fillna, original_version
    )


def keltner_channel_lband_indicator(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
) -> pl.Expr:
    return VolatilityIndicators.keltner_channel_lband_indicator(
        high, low, close, window, window_atr, fillna, original_version
    )


def donchian_channel_hband(
    high, low, close, window=20, offset=0, fillna=False
) -> pl.Expr:
    return VolatilityIndicators.donchian_channel_hband(
        high, low, close, window, offset, fillna
    )


def donchian_channel_lband(
    high, low, close, window=20, offset=0, fillna=False
) -> pl.Expr:
    return VolatilityIndicators.donchian_channel_lband(
        high, low, close, window, offset, fillna
    )


def donchian_channel_mband(
    high, low, close, window=10, offset=0, fillna=False
) -> pl.Expr:
    return VolatilityIndicators.donchian_channel_mband(
        high, low, close, window, offset, fillna
    )


def donchian_channel_wband(
    high, low, close, window=10, offset=0, fillna=False
) -> pl.Expr:
    return VolatilityIndicators.donchian_channel_wband(
        high, low, close, window, offset, fillna
    )


def donchian_channel_pband(
    high, low, close, window=10, offset=0, fillna=False
) -> pl.Expr:
    return VolatilityIndicators.donchian_channel_pband(
        high, low, close, window, offset, fillna
    )


def ulcer_index(close, window=14, fillna=False) -> pl.Expr:
    return VolatilityIndicators.ulcer_index(close, window, fillna)
