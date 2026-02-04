"""test fixtures."""

import sys

import polars as pl

from quant_trade.config.logger import log


def pytest_configure(config):
    """Configure loguru to show debug logs during tests"""
    log.add(
        sys.stderr,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | TEST | {level} | {module}:{function}:{line} | {message}",
    )


def smoke_configure():
    """Configure polars for smoke tests"""
    pl.Config(tbl_cols=20, tbl_rows=20)
    log.add(
        sys.stderr,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | TEST | {level} | {module}:{function}:{line} | {message}",
    )
