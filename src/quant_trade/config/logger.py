import sys
from pathlib import Path
from typing import Literal

from loguru import logger


def setup_logger(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    """
    Configure logger based on the configuration file
    """
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


log = setup_logger()
