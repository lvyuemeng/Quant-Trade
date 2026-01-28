"""Abstract interface for data providers.

Defines the contract that all data providers must implement using composable traits.
"""

from typing import Protocol

import polars as pl

from quant_trade.provider.utils import AdjustCN, DateLike, Period, Quarter


class Stock(Protocol):
    def stock_name_code_map(self) -> pl.DataFrame: ...
    def stock_universe(self, trade_date: DateLike | None = None) -> pl.DataFrame: ...


class StockCls(Protocol):
    def stock_industry_cls(self) -> pl.DataFrame: ...


class CNStock(Stock, Protocol):
    """Protocol for market data including daily prices and financial statements."""

    def market_ohlcv(
        self,
        symbol: str,
        period: Period,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> pl.DataFrame:
        """Fetch daily OHLCV data."""
        ...

    def quarterly_income_statement(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly income statement snapshot."""
        ...

    def quarterly_balance_sheet(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot."""
        ...

    def quarterly_cashflow_statement(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot."""
        ...

    def quarterly_fundamentals(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch and merge income/balance/cashflow for the given quarter."""
        ...


class CNMacro(Protocol):
    def northbound_flow(
        self, start_date: DateLike | None, end_date: DateLike | None
    ) -> pl.DataFrame: ...
    def market_margin_short(
        self, start_date: DateLike | None, end_date: DateLike | None
    ) -> pl.DataFrame: ...
    def shibor(
        self, start_date: DateLike | None, end_date: DateLike | None
    ) -> pl.DataFrame: ...


class CNShareProvider(CNStock, CNMacro, Protocol):
    pass
