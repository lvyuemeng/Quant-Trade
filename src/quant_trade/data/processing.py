from __future__ import annotations

from typing import Literal

import akshare as ak
import narwhals as nw
import pandas as pd

Period = Literal["daily", "weekly", "monthly"]
Adjust = Literal["", "qfq", "hfq"]


def transform_akshare_hist(raw: pd.DataFrame) -> nw.DataFrame:
    """Transform AkShare `stock_zh_a_hist` output into a normalized schema.

    Output schema (used by notebooks + tests):
      - date: datetime
      - symbol: str
      - open/high/low/close: float
      - volume: float (万股; i.e. /100 from AkShare's 成交量)
      - price_change_pct: float
      - color: str (red for up, green for down)

    Notes:
      - Narwhals currently has known friction parsing dates from pandas strings;
        we convert using pandas first (see tests/test_reproduce.py).
    """

    if raw.empty:
        msg = "Empty AkShare dataframe"
        raise ValueError(msg)

    df = raw.copy()
    if "日期" not in df.columns:
        msg = "Expected column '日期' in AkShare dataframe"
        raise KeyError(msg)

    df["日期"] = pd.to_datetime(df["日期"])

    return (
        nw.from_native(df, eager_only=True)
        .select(
            date=nw.col("日期"),
            symbol=nw.col("股票代码"),
            open=nw.col("开盘"),
            high=nw.col("最高"),
            low=nw.col("最低"),
            close=nw.col("收盘"),
            volume=nw.col("成交量") / 100,
            price_change_pct=nw.col("涨跌幅"),
            color=nw.when(nw.col("收盘") >= nw.col("开盘"))
            .then(nw.lit("#FF0000"))
            .otherwise(nw.lit("#00B800")),
        )
        .sort("date")
    )


def get_stock_data(
    *,
    symbol: str,
    period: Period = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: Adjust = "qfq",
) -> nw.DataFrame:
    """Fetch A-share price history via AkShare and return normalized Narwhals DF."""

    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    return transform_akshare_hist(raw)
