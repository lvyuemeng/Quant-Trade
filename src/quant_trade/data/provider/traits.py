"""Abstract interface for data providers.

Defines the contract that all data providers must implement using composable traits.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Protocol

import polars as pl

# Import the new protocol-based traits
from .ashare_data import AShareSpecificInfo
from .stock_info import GeneralStockInfo


class MarketData(Protocol):
    """Protocol for market data including daily prices and financial statements."""

    def daily_market(
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

    def fundamental_data(self, symbol: str) -> pl.DataFrame:
        """Fetch financial statement data."""
        ...

    def fetch_macro_indicators(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch market-wide context features."""
        ...


class DataProvider(GeneralStockInfo, AShareSpecificInfo, MarketData, Protocol):
    """Composable data provider interface using Protocol composition.

    This interface combines multiple protocols to create a flexible, composable
    data provider that can be implemented by different data sources.
    """

    pass


# Keep the ABC version for backward compatibility during transition
class DataProviderABC(ABC):
    """Abstract base class for data providers (legacy version)."""

    @abstractmethod
    def stock_universe(self, trade_date: str | date | None = None) -> pl.DataFrame:
        """Get tradeable share universe for a given date."""
        pass

    @abstractmethod
    def daily_market(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "hfq"
    ) -> pl.DataFrame:
        """Fetch daily OHLCV data."""
        pass

    @abstractmethod
    def quarterly_income_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly income statement snapshot."""
        pass

    @abstractmethod
    def quarterly_balance_sheet(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot."""
        pass

    @abstractmethod
    def quarterly_cashflow_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot."""
        pass

    @abstractmethod
    def quarterly_fundamentals(self, report_date: str) -> pl.DataFrame:
        """Fetch and merge income/balance/cashflow for the given quarter."""
        pass

    @abstractmethod
    def northbound_flow(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch northbound flow (沪深港通 北向) data."""
        pass

    @abstractmethod
    def industry_classification(
        self,
        trade_date: str | date | None = None,
        *,
        sw3_codes: list[str] | None = None,
        max_sw3: int | None = None,
    ) -> pl.DataFrame:
        """Fetch stock -> Shenwan industry (L1/L2) classification."""
        pass

    # @abstractmethod
    # def fetch_index_valuation_csindex(self, symbol: str = "000300") -> pl.DataFrame:
    #     """Fetch index valuation from the csindex valuation endpoint."""
    #     pass

    @abstractmethod
    def margin_balance(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch total A-share margin balance (SH+SZ) as a daily time series."""
        pass

    @abstractmethod
    def fundamental_data(self, symbol: str) -> pl.DataFrame:
        """Fetch financial statement data."""
        pass

    @abstractmethod
    def fetch_macro_indicators(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch market-wide context features."""
        pass

    # @abstractmethod
    # def fetch_stock_info(self, symbol: str) -> pl.DataFrame:
    #     """Fetch individual stock information."""
    #     pass

    # @abstractmethod
    # def fetch_stock_name_code_map(self) -> pl.DataFrame:
    #     """Fetch A-share stock code and name mapping."""
    #     pass
