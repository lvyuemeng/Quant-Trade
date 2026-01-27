from typing import Literal

import marimo as mo
import narwhals as nw
from src.data.processing import get_stock_data as _get_stock_data


@mo.cache
def get_stock_data(
    symbol: str,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> nw.DataFrame:
    return _get_stock_data(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
