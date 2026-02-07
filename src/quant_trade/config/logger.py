import sys
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger


def setup_logger(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    """Configure logger based on the configuration file"""
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    rotation = "100 MB"
    retention = "30 days"
    log_path = "logs/quant_{time}.log"

    logger.remove()
    logger.add(sys.stderr, level=level, format=log_format)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path, rotation=rotation, retention=retention, level=level, format=log_format
    )

    return logger


def debug_null_profile(df: pl.DataFrame) -> pl.DataFrame:
    total = len(df)

    if total == 0:
        log.warning("DEBUG: DataFrame is EMPTY")
        return pl.DataFrame()

    return (
        pl.DataFrame(
            {
                "column": df.columns,
                "non_null": [df[c].count() for c in df.columns],
                "null": [df[c].null_count() for c in df.columns],
            }
        )
        .with_columns(
            [
                (pl.col("non_null") / total).alias("non_null_ratio"),
                (pl.col("null") / total).alias("null_ratio"),
            ]
        )
        .sort("non_null")
    )


log = setup_logger()
