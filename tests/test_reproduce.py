"""
Minimal error reproduction code.

This script demonstrates the issue where narwhals fails to parse dates
from pandas, and the fix using pd.to_datetime() before narwhals.
"""

import narwhals as nw
import pandas as pd


def demonstrate_error():
    """Show the error when using .str.to_date() directly."""
    data = {
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "股票代码": ["000001", "000001", "000001"],
        "开盘": [11.0, 11.2, 11.1],
        "收盘": [11.2, 11.1, 11.3],
        "最高": [11.3, 11.2, 11.4],
        "最低": [10.9, 11.0, 11.0],
        "成交量": [1000000, 1200000, 900000],
        "涨跌幅": [1.5, -0.9, 1.8],
    }
    df_pandas = pd.DataFrame(data)

    print("=" * 60)
    print("Pandas DataFrame (original Chinese columns):")
    print("=" * 60)
    print(df_pandas)
    print()

    df = nw.from_native(df_pandas, eager_only=True)

    print("=" * 60)
    print("Attempting select with .str.to_date() ...")
    print("=" * 60)
    try:
        df_selected = df.select(
            date=nw.col("日期").str.to_date(),
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
        ).sort("date")
        print("SUCCESS!")
        print(f"Columns: {list(df_selected.columns)}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    print()


def demonstrate_fix():
    """Show the fix: convert date using pandas first."""
    data = {
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "股票代码": ["000001", "000001", "000001"],
        "开盘": [11.0, 11.2, 11.1],
        "收盘": [11.2, 11.1, 11.3],
        "最高": [11.3, 11.2, 11.4],
        "最低": [10.9, 11.0, 11.0],
        "成交量": [1000000, 1200000, 900000],
        "涨跌幅": [1.5, -0.9, 1.8],
    }
    df_pandas = pd.DataFrame(data)

    print("=" * 60)
    print("FIX: Convert date using pd.to_datetime() first")
    print("=" * 60)
    df_pandas["日期"] = pd.to_datetime(df_pandas["日期"])
    print(f"Date dtype: {df_pandas['日期'].dtype}")
    print()

    df = nw.from_native(df_pandas, eager_only=True)

    try:
        df_selected = df.select(
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
        ).sort("date")

        print("SUCCESS! Selected columns:")
        print(f"Columns: {list(df_selected.columns)}")
        print()
        print(df_selected)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PART 1: Demonstrating the error")
    print("=" * 60 + "\n")
    demonstrate_error()

    print("\n" + "=" * 60)
    print("PART 2: Demonstrating the fix")
    print("=" * 60 + "\n")
    demonstrate_fix()
