"""test fixtures."""

import sys

from quant_trade.config.logger import log


def pytest_configure(config):
    """Configure loguru to show debug logs during tests"""
    log.add(
        sys.stderr,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | TEST | {level} | {module}:{function}:{line} | {message}",
    )
