"""A-share specific data protocols.

Defines protocols for A-share specific functionality that applies only to
Chinese A-share market.
"""

from typing import Protocol

import polars as pl


class NorthboundFlow(Protocol):
    """Protocol for northbound flow data."""

    def northbound_flow(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch northbound flow (沪深港通 北向) data."""
        ...


class MarginBalance(Protocol):
    """Protocol for margin balance data."""

    def margin_balance(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch total A-share margin balance (SH+SZ) as a daily time series."""
        ...


# class IndexValuation(Protocol):
#     """Protocol for index valuation data."""

#     def fetch_index_valuation_csindex(self, symbol: str = "000300") -> pl.DataFrame:
#         """Fetch index valuation from the csindex valuation endpoint."""
#         ...


class AShareSpecificInfo(NorthboundFlow, MarginBalance, Protocol):
    """Combined protocol for all A-share specific information."""

    pass
