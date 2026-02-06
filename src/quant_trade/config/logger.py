import sys
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger


def setup_logger(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    """Configure logger based on the configuration file"""
    # try:
    #     with open(config_path) as f:
    #         config = yaml.safe_load(f)
    #         log_config = config.get("logging", {})
    # except FileNotFoundError:
    #     log_config = {}

    # level = log_config.get("level", "INFO")
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    # log_format = log_config.get(
    #     "format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    # )
    rotation = "100 MB"
    retention = "30 days"
    log_path = "logs/quant_{time}.log"
    # rotation = log_config.get("rotation", "100 MB")
    # retention = log_config.get("retention", "30 days")
    # log_path = log_config.get("path", "logs/ashare_quant_{time}.log")

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(sys.stderr, level=level, format=log_format)

    # Add file handler
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path, rotation=rotation, retention=retention, level=level, format=log_format
    )

    return logger


@staticmethod
def debug_null_profile(df: pl.DataFrame) -> pl.DataFrame:
    total = len(df)
    if total == 0:
        log.warning("DEBUG: DataFrame is EMPTY")
        return pl.DataFrame()

    total = len(df)
    if total == 0:
        log.warning("DEBUG: DataFrame is EMPTY")
        return pl.DataFrame()

    stats_list = []
    for col_name in df.columns:
        col_data = df[col_name]
        non_null = col_data.count()
        null = col_data.null_count()
        stats_list.append(
            {
                "column": col_name,
                "non_null": non_null,
                "null": null,
                "non_null_ratio": non_null / total if total > 0 else 0,
                "null_ratio": null / total if total > 0 else 0,
            }
        )

    stats = pl.DataFrame(stats_list).sort("non_null")
    return stats


log = setup_logger()
