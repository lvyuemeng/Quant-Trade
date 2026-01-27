import polars as pl

from quant_trade.utils.logger import log


class DataCleaning:
    """
    Module for cleaning and preprocessing raw financial data.
    """

    @staticmethod
    def clean_market_data(df: pl.DataFrame) -> pl.DataFrame:
        """
        Basic cleaning for daily market data.
        - Handle missing values
        - Ensure data types
        - Remove duplicates
        """
        if df.is_empty():
            return df

        log.info("Cleaning market data")

        # 1. Remove duplicates
        initial_len = len(df)
        df = df.unique(
            subset=["date", "ts_code"] if "ts_code" in df.columns else ["date"]
        )
        if len(df) < initial_len:
            log.info(f"Removed {initial_len - len(df)} duplicate rows")

        # 2. Sort by date
        df = df.sort("date")

        # 3. Fill missing values (forward fill for prices, 0 for volume)
        # In Polars, we use fill_null
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).forward_fill())

        if "volume" in df.columns:
            df = df.with_columns(pl.col("volume").fill_null(0))

        return df

    @staticmethod
    def align_fundamental_data(df: pl.DataFrame) -> pl.DataFrame:
        """
        Align fundamental data to avoid look-ahead bias.
        In China A-shares, announcement_date should be used.
        """
        # This is a placeholder for more complex logic described in the plan
        log.info("Aligning fundamental data (placeholder)")
        return df
