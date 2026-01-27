"""General stock information protocols.

Defines protocols for general stock functionality that applies to any stock market,
not just A-shares.
"""

from datetime import date
from typing import Protocol

import polars as pl


class StockUniverse(Protocol):
    """Protocol for fetching tradeable stock universe."""

    def stock_universe(self, trade_date: str | date | None = None) -> pl.DataFrame:
        """Get tradeable share universe for a given date."""
        ...


class StockNameCodeMapping(Protocol):
    """Protocol for stock name and code mapping."""

    def fetch_stock_name_code_map(self) -> pl.DataFrame:
        """Fetch stock code and name mapping."""
        ...


class IndustryClassification(Protocol):
    """Protocol for industry classification."""

    def industry_classification(
        self,
        trade_date: str | date | None = None,
        *,
        sw3_codes: list[str] | None = None,
        max_sw3: int | None = None,
    ) -> pl.DataFrame:
        """Fetch stock -> Shenwan industry (L1/L2) classification."""
        ...


class GeneralStockInfo(
    StockUniverse, StockNameCodeMapping, IndustryClassification, Protocol
):
    """Combined protocol for all general stock information."""

    pass
