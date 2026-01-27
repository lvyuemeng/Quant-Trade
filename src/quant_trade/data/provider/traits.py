"""Abstract interface for data providers.

Defines the contract that all data providers must implement using composable traits.
"""

from typing import Protocol

import polars as pl

# Import the new protocol-based traits
from .ashare import AShareSpecificInfo
from .stock import GeneralStockInfo


class MarketData(Protocol):
    """Protocol for market data including daily prices and financial statements."""

    def market_ohlcv(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "hfq"
    ) -> pl.DataFrame:
        """Fetch daily OHLCV data."""
        ...

    def quarterly_income_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly income statement snapshot."""
        ...

    def quarterly_balance_sheet(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot."""
        ...

    def quarterly_cashflow_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot."""
        ...

    def quarterly_fundamentals(self, report_date: str) -> pl.DataFrame:
        """Fetch and merge income/balance/cashflow for the given quarter."""
        ...

    # def fundamental_data(self, symbol: str) -> pl.DataFrame:
    #     """Fetch financial statement data."""
    #     ...

    # def fetch_macro_indicators(self, start_date: str, end_date: str) -> pl.DataFrame:
    #     """Fetch market-wide context features."""
    #     ...


class ShareProvider(GeneralStockInfo, MarketData, Protocol):
    """Share provider interface based on market.

    This interface combines multiple protocols to create a flexible, composable
    data provider that can be implemented by different data sources.
    """

    pass


class AshareProdiver(ShareProvider, AShareSpecificInfo, Protocol):
    """
    Share provider interface based on market equipped with A-share market informations.

    This interface combines multiple protocols to create a flexible, composable
    data provider that can be implemented by different data sources.
    """
