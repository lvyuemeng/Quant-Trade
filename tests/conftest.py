"""test fixtures."""


from quant_trade.config.logger import setup_logger


def pytest_configure(config):
    """Configure loguru to show debug logs during tests"""
    setup_logger(level="DEBUG")


def smoke_configure():
    """Configure polars for smoke tests"""
    setup_logger(level="DEBUG")
