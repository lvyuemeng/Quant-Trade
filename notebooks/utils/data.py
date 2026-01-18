from typing import Literal

import akshare as ak
import marimo as mo
import narwhals as nw
import pandas as pd


@mo.cache
def get_stock_data(
    symbol: str,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> nw.DataFrame:
    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if raw.empty:
        msg = f"No data returned for symbol {symbol}"
        raise ValueError(msg)

    raw["日期"] = pd.to_datetime(raw["日期"])

    return (
        nw.from_native(raw, eager_only=True)
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
