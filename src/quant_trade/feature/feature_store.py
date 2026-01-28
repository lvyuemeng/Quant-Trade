from pathlib import Path

import polars as pl

from quant_trade.utils.logger import log


class FeatureStore:
    """
    Simple file-based feature store for local development.
    Transitions to Redis/ClickHouse in later stages.
    """

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"

        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def save_raw_data(self, df: pl.DataFrame, name: str):
        """Save raw data to parquet."""
        path = self.raw_path / f"{name}.parquet"
        log.info(f"Saving raw data to {path}")
        df.write_parquet(path)

    def load_raw_data(self, name: str) -> pl.DataFrame:
        """Load raw data from parquet."""
        path = self.raw_path / f"{name}.parquet"
        if not path.exists():
            log.warning(f"Raw data {path} not found")
            return pl.DataFrame()
        return pl.read_parquet(path)

    def save_features(self, df: pl.DataFrame, name: str):
        """Save processed features to parquet."""
        path = self.processed_path / f"{name}.parquet"
        log.info(f"Saving features to {path}")
        df.write_parquet(path)

    def load_features(self, name: str) -> pl.DataFrame:
        """Load processed features from parquet."""
        path = self.processed_path / f"{name}.parquet"
        if not path.exists():
            log.warning(f"Features {path} not found")
            return pl.DataFrame()
        return pl.read_parquet(path)
