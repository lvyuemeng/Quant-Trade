import narwhals as nw
import pandas as pd
import pytest


@pytest.fixture
def mock_akshare_data():
    """Mock pandas DataFrame with Chinese column names (akshare format)."""
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
    return pd.DataFrame(data)


@pytest.fixture
def mock_narwhals_df(mock_akshare_data):
    """Convert mock pandas DataFrame to narwhals DataFrame."""
    df = mock_akshare_data.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    return nw.from_native(df, eager_only=True)


@pytest.fixture
def processed_df(mock_narwhals_df):
    """Apply the same transformations as get_stock_data()."""
    return mock_narwhals_df.select(
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
