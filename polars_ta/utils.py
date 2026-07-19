import math

import polars as pl
import polars.selectors as cs


class BaseIndicator:
    """Utility functions for the Polars TA library."""

    @staticmethod
    def check_fillna(expr: pl.Expr | str, fillna: bool, value: int = 0) -> pl.Expr:
        """
        Check if fillna flag is True and fill gaps.
        Replaces inf/-inf with nulls before filling.
        """
        expr = pl.col(expr) if isinstance(expr, str) else expr

        if not fillna:
            return expr

        # Replace Inf, -Inf, and NaN with Polars Null
        clean_expr = (
            pl.when(expr.is_infinite() | expr.is_nan()).then(None).otherwise(expr)
        )

        if value == -1:
            # ffill().bfill() equivalent in Polars
            return clean_expr.forward_fill().backward_fill()
        else:
            # ffill().fillna(value) equivalent in Polars
            return clean_expr.forward_fill().fill_null(value)

    @staticmethod
    def true_range(
        high: pl.Expr | str, low: pl.Expr | str, prev_close: pl.Expr | str
    ) -> pl.Expr:
        """Calculate the True Range using horizontal aggregation."""
        high = pl.col(high) if isinstance(high, str) else high
        low = pl.col(low) if isinstance(low, str) else low
        prev_close = pl.col(prev_close) if isinstance(prev_close, str) else prev_close

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        return pl.max_horizontal([tr1, tr2, tr3])

    @staticmethod
    def sma(expr: pl.Expr | str, periods: int, fillna: bool = False) -> pl.Expr:
        """Simple Moving Average"""
        expr = pl.col(expr) if isinstance(expr, str) else expr
        min_periods = 1 if fillna else periods
        return expr.rolling_mean(window_size=periods, min_samples=min_periods)

    @staticmethod
    def ema(expr: pl.Expr | str, periods: int, fillna: bool = False) -> pl.Expr:
        """Exponential Moving Average"""
        expr = pl.col(expr) if isinstance(expr, str) else expr
        min_periods = 1 if fillna else periods
        return expr.ewm_mean(span=periods, adjust=False, min_samples=min_periods)

    @staticmethod
    def get_min_max(
        expr1: pl.Expr | str, expr2: pl.Expr | str, function: str = "min"
    ) -> pl.Expr:
        """Find min or max value between two series for each index."""
        expr1 = pl.col(expr1) if isinstance(expr1, str) else expr1
        expr2 = pl.col(expr2) if isinstance(expr2, str) else expr2

        if function == "min":
            return pl.min_horizontal([expr1, expr2])
        elif function == "max":
            return pl.max_horizontal([expr1, expr2])
        else:
            raise ValueError('"function" variable value should be "min" or "max"')


class DataCleaner:
    """Methods to handle, track, and heal invalid data before applying TA indicators."""

    @staticmethod
    def _build_invalid_mask(df: pl.DataFrame) -> pl.Expr:
        """
        Helper method: Builds a mask that flags rows containing
        NaN, Null, Infinity, or excessively large numbers.
        """
        num_cols = df.select(cs.numeric()).columns
        is_invalid_mask = pl.lit(False)
        big_number = math.exp(709)

        for col in num_cols:
            col_is_bad = (
                (pl.col(col) >= big_number)
                | pl.col(col).is_nan()
                | pl.col(col).is_null()
                | pl.col(col).is_infinite()
            )
            is_invalid_mask = is_invalid_mask | col_is_bad

        return is_invalid_mask

    @staticmethod
    def dropna(df: pl.DataFrame) -> pl.DataFrame:
        """
        Drop rows with nulls, NaNs, or excessively large numbers
        in numeric columns (safe alternative to the original ta library).
        """
        invalid_mask = DataCleaner._build_invalid_mask(df)
        # Keep rows where invalid_mask is False
        return df.filter(~invalid_mask)

    @staticmethod
    def get_invalid_indices(df: pl.DataFrame) -> list[int]:
        """
        Returns a list of integer row indices where invalid data exists.
        Useful for logging or inspecting anomalies.
        """
        invalid_mask = DataCleaner._build_invalid_mask(df)

        bad_indices = (
            df.with_row_index("row_idx")
            .filter(invalid_mask)
            .get_column("row_idx")
            .to_list()
        )
        return bad_indices

    @staticmethod
    def approximate_invalid_values(df: pl.DataFrame) -> pl.DataFrame:
        """
        Replaces invalid values with Polars Nulls, then approximates them
        using linear interpolation and forward-filling based on past values.
        """
        num_cols = df.select(cs.numeric()).columns
        big_number = math.exp(709)

        exprs = []
        for col in num_cols:
            clean_col = (
                pl.when(
                    (pl.col(col) >= big_number)
                    | pl.col(col).is_nan()
                    | pl.col(col).is_infinite()
                )
                .then(None)
                .otherwise(pl.col(col))
            )

            # Interpolate (linear line) and forward_fill (carry last good value)
            imputed_col = clean_col.interpolate().forward_fill()
            exprs.append(imputed_col.alias(col))

        return df.with_columns(exprs)
