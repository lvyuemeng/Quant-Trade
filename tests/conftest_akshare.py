"""AkShare-specific test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def mock_stock_codes_sh():
    """Mock Shanghai stock codes data."""
    data = {
        "证券代码": ["600000", "600001", "600002"],
        "证券简称": ["浦发银行", "邯郸钢铁", "万科企业"],
        "上市日期": ["1999-11-10", "1998-01-01", "1991-01-29"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_stock_codes_sz():
    """Mock Shenzhen stock codes data."""
    data = {
        "A股代码": ["000001", "000002", "000003"],
        "A股简称": ["平安银行", "万科A", "PT金田"],
        "A股上市日期": ["1991-04-03", "1991-01-29", "1990-12-01"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_market_ohlcv():
    """Mock market OHLCV data with Chinese column names."""
    data = {
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "股票代码": ["000001", "000001", "000001"],
        "开盘": [11.0, 11.2, 11.1],
        "收盘": [11.2, 11.1, 11.3],
        "最高": [11.3, 11.2, 11.4],
        "最低": [10.9, 11.0, 11.0],
        "成交量": [1000000, 1200000, 900000],
        "成交额": [11200000, 13320000, 10170000],
        "振幅": [3.6, 1.8, 3.6],
        "涨跌幅": [1.5, -0.9, 1.8],
        "涨跌额": [0.16, -0.10, 0.20],
        "换手率": [0.05, 0.06, 0.04],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_financial_data():
    """Mock financial statement data."""
    data = {
        "股票代码": ["000001", "000002", "600000"],
        "股票简称": ["平安银行", "万科A", "浦发银行"],
        "公告日期": ["2024-01-31", "2024-01-31", "2024-01-31"],
        "净利润": [1000000000, 2000000000, 1500000000],
        "营业利润": [1200000000, 2200000000, 1700000000],
        "营业总收入": [5000000000, 6000000000, 4500000000],
        "净利润同比": [10.5, 8.2, 12.1],
        "营业总收入同比": [15.2, 12.8, 18.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_industry_data():
    """Mock Shenwan industry classification data."""
    data = {
        "行业代码": ["850111", "850112", "850113"],
        "行业名称": ["种子生产", "种植业", "林业"],
        "上级行业": ["种植业与林业", "种植业与林业", "种植业与林业"],
        "成份个数": [10, 25, 15],
        "静态市盈率": [25.5, 30.2, 28.7],
        "市净率": [2.1, 2.5, 2.3],
    }
    return pd.DataFrame(data)
