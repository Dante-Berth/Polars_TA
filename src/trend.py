import polars as pl
import numpy as np
from src.utils import BaseIndicator


class TrendIndicators:
    """Trend Indicators translated to Polars Expressions."""

    # ---------------------------------------------------------
    # Aroon Indicator
    # ---------------------------------------------------------
    @staticmethod
    def aroon_up(
        high: str | pl.Expr, window: int = 25, fillna: bool = False
    ) -> pl.Expr:
        """Aroon Up Channel"""
        high = pl.col(high) if isinstance(high, str) else high
        min_periods = 1 if fillna else window + 1

        # rolling_map allows us to run the argmax logic on each window slice
        expr = high.rolling_map(
            lambda s: float(s.to_numpy().argmax()) / window * 100,
            window_size=window + 1,
            min_periods=min_periods,
        )
        return BaseIndicator.check_fillna(expr, fillna, value=0)

    @staticmethod
    def aroon_down(
        low: str | pl.Expr, window: int = 25, fillna: bool = False
    ) -> pl.Expr:
        """Aroon Down Channel"""
        low = pl.col(low) if isinstance(low, str) else low
        min_periods = 1 if fillna else window + 1

        expr = low.rolling_map(
            lambda s: float(s.to_numpy().argmin()) / window * 100,
            window_size=window + 1,
            min_periods=min_periods,
        )
        return BaseIndicator.check_fillna(expr, fillna, value=0)

    @staticmethod
    def aroon_indicator(
        high: str | pl.Expr, low: str | pl.Expr, window: int = 25, fillna: bool = False
    ) -> pl.Expr:
        """Aroon Indicator (Up - Down)"""
        up = TrendIndicators.aroon_up(high, window, fillna)
        down = TrendIndicators.aroon_down(low, window, fillna)
        diff = up - down
        return BaseIndicator.check_fillna(diff, fillna, value=0)

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------
    @staticmethod
    def macd(
        close: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        fillna: bool = False,
    ) -> pl.Expr:
        """MACD Line"""
        close = pl.col(close) if isinstance(close, str) else close
        ema_fast = BaseIndicator.ema(close, window_fast, fillna)
        ema_slow = BaseIndicator.ema(close, window_slow, fillna)
        macd_line = ema_fast - ema_slow
        return BaseIndicator.check_fillna(macd_line, fillna, value=0)

    @staticmethod
    def macd_signal(
        close: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        """MACD Signal Line"""
        macd_line = TrendIndicators.macd(close, window_slow, window_fast, fillna)
        signal_line = BaseIndicator.ema(macd_line, window_sign, fillna)
        return BaseIndicator.check_fillna(signal_line, fillna, value=0)

    @staticmethod
    def macd_diff(
        close: str | pl.Expr,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        """MACD Histogram"""
        macd_line = TrendIndicators.macd(close, window_slow, window_fast, fillna)
        signal_line = TrendIndicators.macd_signal(
            close, window_slow, window_fast, window_sign, fillna
        )
        diff_line = macd_line - signal_line
        return BaseIndicator.check_fillna(diff_line, fillna, value=0)

    # ---------------------------------------------------------
    # Standard Moving Averages
    # ---------------------------------------------------------
    @staticmethod
    def ema_indicator(
        close: str | pl.Expr, window: int = 14, fillna: bool = False
    ) -> pl.Expr:
        """Exponential Moving Average (EMA)"""
        return BaseIndicator.ema(close, window, fillna)

    @staticmethod
    def sma_indicator(
        close: str | pl.Expr, window: int, fillna: bool = False
    ) -> pl.Expr:
        """Simple Moving Average (SMA)"""
        return BaseIndicator.sma(close, window, fillna)

    # ---------------------------------------------------------
    # WMA Indicator
    # ---------------------------------------------------------
    @staticmethod
    def wma_indicator(
        close: str | pl.Expr, window: int = 9, fillna: bool = False
    ) -> pl.Expr:
        """Weighted Moving Average (WMA)"""
        close = pl.col(close) if isinstance(close, str) else close

        # Pre-calculate weights array exactly as the original does
        weights = np.array(
            [i * 2 / (window * (window + 1)) for i in range(1, window + 1)]
        )

        # Apply the weighted dot product over the rolling window
        expr = close.rolling_map(
            lambda s: np.dot(s.to_numpy(), weights), window_size=window
        )
        return BaseIndicator.check_fillna(expr, fillna, value=0)

    # ---------------------------------------------------------
    # TRIX Indicator
    # ---------------------------------------------------------
    @staticmethod
    def trix(close: str | pl.Expr, window: int = 15, fillna: bool = False) -> pl.Expr:
        """Trix (TRIX) - Triple exponentially smoothed moving average percent change"""
        close = pl.col(close) if isinstance(close, str) else close

        ema1 = BaseIndicator.ema(close, window, fillna)
        ema2 = BaseIndicator.ema(ema1, window, fillna)
        ema3 = BaseIndicator.ema(ema2, window, fillna)

        # Original: ema3.shift(1, fill_value=ema3.mean())
        ema3_mean = ema3.mean()
        shifted_ema3 = ema3.shift(1).fill_null(ema3_mean)

        trix_expr = ((ema3 - shifted_ema3) / shifted_ema3) * 100

        return BaseIndicator.check_fillna(trix_expr, fillna, value=0)

    @staticmethod
    def mass_index(
        high: str | pl.Expr,
        low: str | pl.Expr,
        window_fast: int = 9,
        window_slow: int = 25,
        fillna: bool = False,
    ) -> pl.Expr:
        """Mass Index (MI)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low

        min_periods = 1 if fillna else window_slow

        amplitude = high - low
        ema1 = BaseIndicator.ema(amplitude, window_fast, fillna)
        ema2 = BaseIndicator.ema(ema1, window_fast, fillna)

        mass = ema1 / ema2
        mass_idx = mass.rolling_sum(window_size=window_slow, min_periods=min_periods)

        return BaseIndicator.check_fillna(mass_idx, fillna, value=0)

    # ---------------------------------------------------------
    # Ichimoku Kinkō Hyō
    # ---------------------------------------------------------
    @staticmethod
    def _ichimoku_line(
        high: pl.Expr, low: pl.Expr, window: int, fillna: bool
    ) -> pl.Expr:
        """Helper method to calculate Ichimoku lines (Conv, Base, Span B)"""
        min_periods = 1 if fillna else window
        roll_max = high.rolling_max(window_size=window, min_periods=min_periods)
        roll_min = low.rolling_min(window_size=window, min_periods=min_periods)
        return 0.5 * (roll_max + roll_min)

    @staticmethod
    def ichimoku_conversion_line(
        high: str | pl.Expr, low: str | pl.Expr, window1: int = 9, fillna: bool = False
    ) -> pl.Expr:
        """Tenkan-sen (Conversion Line)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low

        conv = TrendIndicators._ichimoku_line(high, low, window1, fillna)
        return BaseIndicator.check_fillna(conv, fillna, value=-1)

    @staticmethod
    def ichimoku_base_line(
        high: str | pl.Expr, low: str | pl.Expr, window2: int = 26, fillna: bool = False
    ) -> pl.Expr:
        """Kijun-sen (Base Line)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low

        base = TrendIndicators._ichimoku_line(high, low, window2, fillna)
        return BaseIndicator.check_fillna(base, fillna, value=-1)

    @staticmethod
    def ichimoku_a(
        high: str | pl.Expr,
        low: str | pl.Expr,
        window1: int = 9,
        window2: int = 26,
        visual: bool = False,
        fillna: bool = False,
    ) -> pl.Expr:
        """Senkou Span A (Leading Span A)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low

        conv = TrendIndicators._ichimoku_line(high, low, window1, fillna)
        base = TrendIndicators._ichimoku_line(high, low, window2, fillna)

        spana = 0.5 * (conv + base)
        if visual:
            spana = spana.shift(window2).fill_null(spana.mean())

        return BaseIndicator.check_fillna(spana, fillna, value=-1)

    @staticmethod
    def ichimoku_b(
        high: str | pl.Expr,
        low: str | pl.Expr,
        window2: int = 26,
        window3: int = 52,
        visual: bool = False,
        fillna: bool = False,
    ) -> pl.Expr:
        """Senkou Span B (Leading Span B)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low

        # Span B is calculated using the longest window (window3) but shifted by window2 if visual
        spanb = TrendIndicators._ichimoku_line(high, low, window3, fillna)

        if visual:
            spanb = spanb.shift(window2).fill_null(spanb.mean())

        return BaseIndicator.check_fillna(spanb, fillna, value=-1)

    # ---------------------------------------------------------
    # KST Oscillator
    # ---------------------------------------------------------
    @staticmethod
    def kst(
        close: str | pl.Expr,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        window1: int = 10,
        window2: int = 10,
        window3: int = 10,
        window4: int = 15,
        fillna: bool = False,
    ) -> pl.Expr:
        """Know Sure Thing (KST)"""
        close = pl.col(close) if isinstance(close, str) else close

        def _rocma(r: int, w: int) -> pl.Expr:
            """Helper to calculate the Smoothed Rate of Change"""
            min_p = 1 if fillna else w
            shifted_close = close.shift(r).fill_null(close.mean())
            roc = (close - shifted_close) / shifted_close
            return roc.rolling_mean(window_size=w, min_periods=min_p)

        rocma1 = _rocma(roc1, window1)
        rocma2 = _rocma(roc2, window2)
        rocma3 = _rocma(roc3, window3)
        rocma4 = _rocma(roc4, window4)

        kst_val = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
        return BaseIndicator.check_fillna(kst_val, fillna, value=0)

    @staticmethod
    def kst_sig(
        close: str | pl.Expr,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        window1: int = 10,
        window2: int = 10,
        window3: int = 10,
        window4: int = 15,
        nsig: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        """Signal Line Know Sure Thing (KST)"""
        kst_val = TrendIndicators.kst(
            close, roc1, roc2, roc3, roc4, window1, window2, window3, window4, fillna
        )
        kst_sig_val = kst_val.rolling_mean(window_size=nsig, min_periods=1)
        return BaseIndicator.check_fillna(kst_sig_val, fillna, value=0)

    @staticmethod
    def kst_diff(
        close: str | pl.Expr,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        window1: int = 10,
        window2: int = 10,
        window3: int = 10,
        window4: int = 15,
        nsig: int = 9,
        fillna: bool = False,
    ) -> pl.Expr:
        """Diff Know Sure Thing (KST)"""
        kst_val = TrendIndicators.kst(
            close, roc1, roc2, roc3, roc4, window1, window2, window3, window4, fillna
        )
        kst_sig_val = TrendIndicators.kst_sig(
            close,
            roc1,
            roc2,
            roc3,
            roc4,
            window1,
            window2,
            window3,
            window4,
            nsig,
            fillna,
        )

        kst_diff_val = kst_val - kst_sig_val
        return BaseIndicator.check_fillna(kst_diff_val, fillna, value=0)


class ComplexTrendIndicators:
    # ---------------------------------------------------------
    # Detrended Price Oscillator (DPO)
    # ---------------------------------------------------------
    @staticmethod
    def dpo(close: str | pl.Expr, window: int = 20, fillna: bool = False) -> pl.Expr:
        """Detrended Price Oscillator (DPO)"""
        close = pl.col(close) if isinstance(close, str) else close
        min_periods = 1 if fillna else window

        # Shift back by (window / 2) + 1
        shift_val = int((0.5 * window) + 1)
        shifted_close = close.shift(shift_val).fill_null(close.mean())
        sma = close.rolling_mean(window_size=window, min_periods=min_periods)

        dpo_val = shifted_close - sma
        return BaseIndicator.check_fillna(dpo_val, fillna, value=0)

    # ---------------------------------------------------------
    # Commodity Channel Index (CCI)
    # ---------------------------------------------------------
    @staticmethod
    def cci(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 20,
        constant: float = 0.015,
        fillna: bool = False,
    ) -> pl.Expr:
        """Commodity Channel Index (CCI)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close
        min_periods = 1 if fillna else window

        typical_price = (high + low + close) / 3.0
        tp_sma = typical_price.rolling_mean(window_size=window, min_periods=min_periods)

        # Polars rolling_map for Mean Absolute Deviation (MAD)
        mad = typical_price.rolling_map(
            lambda s: np.mean(np.abs(s - np.mean(s))),
            window_size=window,
            min_periods=min_periods,
        )

        cci_val = (typical_price - tp_sma) / (constant * mad)
        return BaseIndicator.check_fillna(cci_val, fillna, value=0)

    # ---------------------------------------------------------
    # Vortex Indicator (VI)
    # ---------------------------------------------------------
    @staticmethod
    def vortex_pos(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """+VI (Positive Vortex Indicator)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close
        min_periods = 1 if fillna else window

        close_shift = close.shift(1).fill_null(close.mean())
        true_range = BaseIndicator.true_range(high, low, close_shift)

        trn = true_range.rolling_sum(window_size=window, min_periods=min_periods)
        vmp = (high - low.shift(1)).abs()

        vip = vmp.rolling_sum(window_size=window, min_periods=min_periods) / trn
        return BaseIndicator.check_fillna(vip, fillna, value=1)

    @staticmethod
    def vortex_neg(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """-VI (Negative Vortex Indicator)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close
        min_periods = 1 if fillna else window

        close_shift = close.shift(1).fill_null(close.mean())
        true_range = BaseIndicator.true_range(high, low, close_shift)

        trn = true_range.rolling_sum(window_size=window, min_periods=min_periods)
        vmm = (low - high.shift(1)).abs()

        vin = vmm.rolling_sum(window_size=window, min_periods=min_periods) / trn
        return BaseIndicator.check_fillna(vin, fillna, value=1)

    @staticmethod
    def vortex_diff(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """Diff VI"""
        vip = ComplexTrendIndicators.vortex_pos(high, low, close, window, fillna)
        vin = ComplexTrendIndicators.vortex_neg(high, low, close, window, fillna)
        vid = vip - vin
        return BaseIndicator.check_fillna(vid, fillna, value=0)

    # ---------------------------------------------------------
    # Average Directional Movement Index (ADX)
    # ---------------------------------------------------------
    @staticmethod
    def _adx_components(
        high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Helper to compute Directional Movement and True Range via Wilder's Smoothing (EMA)"""
        up = high - high.shift(1)
        down = low.shift(1) - low

        # Calculate +DM and -DM
        pos_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
        neg_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)

        tr = BaseIndicator.true_range(high, low, close.shift(1))

        # Wilder's Smoothing = EMA with alpha = 1 / window
        alpha = 1.0 / window
        smoothed_tr = tr.ewm_mean(alpha=alpha, adjust=False)
        smoothed_pos_dm = pos_dm.ewm_mean(alpha=alpha, adjust=False)
        smoothed_neg_dm = neg_dm.ewm_mean(alpha=alpha, adjust=False)

        return smoothed_pos_dm, smoothed_neg_dm, smoothed_tr

    @staticmethod
    def adx_pos(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """+DI"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close

        pos_dm, _, tr = ComplexTrendIndicators._adx_components(high, low, close, window)
        dip = 100 * (pos_dm / tr)
        return BaseIndicator.check_fillna(dip, fillna, value=20)

    @staticmethod
    def adx_neg(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """-DI"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close

        _, neg_dm, tr = ComplexTrendIndicators._adx_components(high, low, close, window)
        din = 100 * (neg_dm / tr)
        return BaseIndicator.check_fillna(din, fillna, value=20)

    @staticmethod
    def adx(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        window: int = 14,
        fillna: bool = False,
    ) -> pl.Expr:
        """Average Directional Index (ADX)"""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close

        dip = ComplexTrendIndicators.adx_pos(high, low, close, window, fillna)
        din = ComplexTrendIndicators.adx_neg(high, low, close, window, fillna)

        dx = 100 * (dip - din).abs() / (dip + din)
        # ADX is the smoothed moving average of DX
        adx_val = dx.ewm_mean(alpha=1.0 / window, adjust=False)
        return BaseIndicator.check_fillna(adx_val, fillna, value=20)

    # ---------------------------------------------------------
    # Parabolic SAR (PSAR)
    # ---------------------------------------------------------
    @staticmethod
    def psar(
        high: str | pl.Expr,
        low: str | pl.Expr,
        close: str | pl.Expr,
        step: float = 0.02,
        max_step: float = 0.20,
        fillna: bool = False,
    ) -> pl.Expr:
        """Parabolic SAR computed using map_batches for stateful execution."""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        close = pl.col(close) if isinstance(close, str) else close

        def _calc_psar(struct_s: pl.Series) -> pl.Series:
            """Internal NumPy loop to handle the FSM logic of PSAR."""
            df = struct_s.struct.unnest()
            h_arr = df["high"].to_numpy()
            l_arr = df["low"].to_numpy()
            c_arr = df["close"].to_numpy()

            n = len(c_arr)
            if n < 2:
                return pl.Series(c_arr)

            psar = np.copy(c_arr)
            up_trend = True
            af = step
            up_trend_high = h_arr[0]
            down_trend_low = l_arr[0]

            for i in range(2, n):
                reversal = False
                max_high = h_arr[i]
                min_low = l_arr[i]

                if up_trend:
                    psar[i] = psar[i - 1] + (af * (up_trend_high - psar[i - 1]))
                    if min_low < psar[i]:
                        reversal = True
                        psar[i] = up_trend_high
                        down_trend_low = min_low
                        af = step
                    else:
                        if max_high > up_trend_high:
                            up_trend_high = max_high
                            af = min(af + step, max_step)
                        low1, low2 = l_arr[i - 1], l_arr[i - 2]
                        if low2 < psar[i]:
                            psar[i] = low2
                        elif low1 < psar[i]:
                            psar[i] = low1
                else:
                    psar[i] = psar[i - 1] - (af * (psar[i - 1] - down_trend_low))
                    if max_high > psar[i]:
                        reversal = True
                        psar[i] = down_trend_low
                        up_trend_high = max_high
                        af = step
                    else:
                        if min_low < down_trend_low:
                            down_trend_low = min_low
                            af = min(af + step, max_step)
                        high1, high2 = h_arr[i - 1], h_arr[i - 2]
                        if high2 > psar[i]:
                            psar[i] = high2
                        elif high1 > psar[i]:
                            psar[i] = high1

                up_trend = up_trend != reversal

            return pl.Series(psar)

        # Pack columns into a struct and pass to map_batches
        expr = pl.struct(
            [high.alias("high"), low.alias("low"), close.alias("close")]
        ).map_batches(_calc_psar, returns_scalar=False)
        return BaseIndicator.check_fillna(expr, fillna, value=-1)

    # ---------------------------------------------------------
    # Schaff Trend Cycle (STC)
    # ---------------------------------------------------------
    @staticmethod
    def stc(
        close: str | pl.Expr,
        window_slow: int = 50,
        window_fast: int = 23,
        cycle: int = 10,
        smooth1: int = 3,
        smooth2: int = 3,
        fillna: bool = False,
    ) -> pl.Expr:
        """Schaff Trend Cycle (STC)"""
        close = pl.col(close) if isinstance(close, str) else close
        min_periods_cycle = 1 if fillna else cycle

        # 1. MACD Line
        ema_fast = BaseIndicator.ema(close, window_fast, fillna)
        ema_slow = BaseIndicator.ema(close, window_slow, fillna)
        macd = ema_fast - ema_slow

        # 2. Stochastic of MACD
        macd_min = macd.rolling_min(window_size=cycle, min_periods=min_periods_cycle)
        macd_max = macd.rolling_max(window_size=cycle, min_periods=min_periods_cycle)

        # Guard against division by zero in stochastic calculation
        macd_range = macd_max - macd_min
        macd_range = pl.when(macd_range == 0).then(1).otherwise(macd_range)
        stoch_k = 100 * (macd - macd_min) / macd_range

        # 3. Smoothed Stochastic
        stoch_d = BaseIndicator.ema(stoch_k, smooth1, fillna)

        # 4. Stochastic of Smoothed Stochastic
        stoch_d_min = stoch_d.rolling_min(
            window_size=cycle, min_periods=min_periods_cycle
        )
        stoch_d_max = stoch_d.rolling_max(
            window_size=cycle, min_periods=min_periods_cycle
        )

        stoch_d_range = stoch_d_max - stoch_d_min
        stoch_d_range = pl.when(stoch_d_range == 0).then(1).otherwise(stoch_d_range)
        stoch_kd = 100 * (stoch_d - stoch_d_min) / stoch_d_range

        # 5. Final STC (Smoothed again)
        stc_val = BaseIndicator.ema(stoch_kd, smooth2, fillna)

        return BaseIndicator.check_fillna(stc_val, fillna, value=0)


def ema_indicator(close, window=12, fillna=False) -> pl.Expr:
    return TrendIndicators.ema_indicator(close, window, fillna)


def sma_indicator(close, window=12, fillna=False) -> pl.Expr:
    return TrendIndicators.sma_indicator(close, window, fillna)


def wma_indicator(close, window=9, fillna=False) -> pl.Expr:
    return TrendIndicators.wma_indicator(close, window, fillna)


def macd(close, window_slow=26, window_fast=12, fillna=False) -> pl.Expr:
    return TrendIndicators.macd(close, window_slow, window_fast, fillna)


def macd_signal(
    close, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pl.Expr:
    return TrendIndicators.macd_signal(
        close, window_slow, window_fast, window_sign, fillna
    )


def macd_diff(
    close, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pl.Expr:
    return TrendIndicators.macd_diff(
        close, window_slow, window_fast, window_sign, fillna
    )


def adx(high, low, close, window=14, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.adx(high, low, close, window, fillna)


def adx_pos(high, low, close, window=14, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.adx_pos(high, low, close, window, fillna)


def adx_neg(high, low, close, window=14, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.adx_neg(high, low, close, window, fillna)


def vortex_indicator_pos(high, low, close, window=14, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.vortex_pos(high, low, close, window, fillna)


def vortex_indicator_neg(high, low, close, window=14, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.vortex_neg(high, low, close, window, fillna)


def trix(close, window=15, fillna=False) -> pl.Expr:
    return TrendIndicators.trix(close, window, fillna)


def mass_index(high, low, window_fast=9, window_slow=25, fillna=False) -> pl.Expr:
    return TrendIndicators.mass_index(high, low, window_fast, window_slow, fillna)


def cci(high, low, close, window=20, constant=0.015, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.cci(high, low, close, window, constant, fillna)


def dpo(close, window=20, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.dpo(close, window, fillna)


def kst(
    close,
    roc1=10,
    roc2=15,
    roc3=20,
    roc4=30,
    window1=10,
    window2=10,
    window3=10,
    window4=15,
    fillna=False,
) -> pl.Expr:
    return TrendIndicators.kst(
        close, roc1, roc2, roc3, roc4, window1, window2, window3, window4, fillna
    )


def kst_sig(
    close,
    roc1=10,
    roc2=15,
    roc3=20,
    roc4=30,
    window1=10,
    window2=10,
    window3=10,
    window4=15,
    nsig=9,
    fillna=False,
) -> pl.Expr:
    return TrendIndicators.kst_sig(
        close, roc1, roc2, roc3, roc4, window1, window2, window3, window4, nsig, fillna
    )


def stc(
    close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=False
) -> pl.Expr:
    return ComplexTrendIndicators.stc(
        close, window_slow, window_fast, cycle, smooth1, smooth2, fillna
    )


def ichimoku_conversion_line(
    high, low, window1=9, window2=26, visual=False, fillna=False
) -> pl.Expr:
    return TrendIndicators.ichimoku_conversion_line(high, low, window1, fillna)


def ichimoku_base_line(
    high, low, window1=9, window2=26, visual=False, fillna=False
) -> pl.Expr:
    return TrendIndicators.ichimoku_base_line(high, low, window2, fillna)


def ichimoku_a(high, low, window1=9, window2=26, visual=False, fillna=False) -> pl.Expr:
    return TrendIndicators.ichimoku_a(high, low, window1, window2, visual, fillna)


def ichimoku_b(
    high, low, window2=26, window3=52, visual=False, fillna=False
) -> pl.Expr:
    return TrendIndicators.ichimoku_b(high, low, window2, window3, visual, fillna)


def aroon_up(high, low, window=25, fillna=False) -> pl.Expr:
    return TrendIndicators.aroon_up(high, window, fillna)


def aroon_down(high, low, window=25, fillna=False) -> pl.Expr:
    return TrendIndicators.aroon_down(low, window, fillna)


def psar(high, low, close, step=0.02, max_step=0.20, fillna=False) -> pl.Expr:
    return ComplexTrendIndicators.psar(high, low, close, step, max_step, fillna)
