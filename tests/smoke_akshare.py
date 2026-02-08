from datetime import date
from typing import Final

import polars as pl

import quant_trade.provider.akshare as ak
from quant_trade.feature.process import (
    Behavioral,
    Fundamental,
    MarginShort,
    Northbound,
    Shibor,
)
from tests.conftest import smoke_configure

INDEX_CODE: Final[str] = "000001"
PERIOD: Final[str] = "daily"

SHEET_YEAR: Final[int] = 2021
SHEET_QUATER: Final[int] = 1


def test_stock_daily(m: ak.AkShareMicro):
    df = m.market_ohlcv("000001", period=PERIOD)
    print(f"stock daily: {df}")


def test_quarter_income(m: ak.AkShareMicro):
    df = m.quarterly_income(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter balance: {df.columns} \n {len(df)} {df}")


def test_quarter_balance(m: ak.AkShareMicro):
    df = m.quarterly_balance(year=SHEET_YEAR, quarter=SHEET_QUATER)
    filter_df = df.filter(
        pl.col("total_equity").is_not_nan() & pl.col("total_equity").gt(0)
    )
    print(f"quarter balance: {df.columns} \n {len(filter_df)} {filter_df}")


def test_quarter_cashflow(m: ak.AkShareMicro):
    df = m.quarterly_cashflow(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter cashflow: {df.columns} \n {df}")


def test_quarter_fundamental(m: ak.AkShareMicro):
    df = m.quarterly_fundamentals(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter balance: {df.columns} \n {len(df)} {df}")


def test_batch_quarter_fundamental(m: ak.AkShareMicro):
    dfs = m.batch_quarterly_fundamentals(date(2015, 1, 1), date(2015, 12, 1))
    print(f"quarters fetched: {len(dfs)}")
    for df in dfs[:3]:
        if df.is_empty():
            print("DataFrame is empty")
        else:
            print(f"Shape: {df.shape}")
            print(df.head(5))


def test_northbound(m: ak.AkShareMacro):
    df = m.northbound_flow()
    print(f"northbound: {df.columns} \n {df}")


def test_marginshort(m: ak.AkShareMacro):
    df = m.market_margin_short()
    print(f"margin short: {df.columns} \n {df}")


def test_shibor(m: ak.AkShareMacro):
    df = m.shibor()
    print(f"shibor: {df.columns} \n {df}")


def test_index_daily(m: ak.AkShareMacro):
    df = m.csi1000_daily_ohlcv()
    print(f"index daily: {df.columns} \n {df}")


def test_qvix_daily(m: ak.AkShareMacro):
    df = m.csi1000qvix_daily_ohlc()
    print(f"qvix index daily: {df.columns} \n {df}")


def test_industry_cls():
    icls = ak.SWIndustryCls()
    df = icls.stock_l1_industry_cls()
    print(f"industry class: {df.columns} \n {df}")


# ===


def test_fundamentalf(m: ak.AkShareMicro):
    df = m.quarterly_fundamentals(SHEET_YEAR, SHEET_QUATER)
    df = Fundamental().metrics(df)
    print(f"fundamental: {df.columns} \n {df}")


def test_behavoiralf(m: ak.AkShareMicro):
    df = m.market_ohlcv(INDEX_CODE, PERIOD)
    df = Behavioral().metrics(df)
    print(f"behavoiral: {df.columns} \n {df}")


def test_behavoiral_qvix(m: ak.AkShareMacro):
    df = m.csi1000qvix_daily_ohlc()
    df = Behavioral().metrics(df)
    print(f"qvix behavoiral: {df.columns} \n {df}")


def test_northboundf(m: ak.AkShareMacro):
    df = m.northbound_flow()
    df = Northbound().metrics(df)
    print(f"northbound: {df.columns} \n {df}")


def test_marginshortf(m: ak.AkShareMacro):
    df = m.market_margin_short()
    df = MarginShort().metrics(df)
    print(f"northbound: {df.columns} \n {df}")


def test_shiborf(m: ak.AkShareMacro):
    df = m.shibor()
    df = Shibor().metrics(df)
    print(f"northbound: {df.columns} \n {df}")


if __name__ == "__main__":
    smoke_configure()
    pl.Config(tbl_cols=20)

    # === Micro ===
    micro = ak.AkShareMicro()
    # test_stock_daily(micro)
    # test_quarter_income(micro)
    # test_quarter_balance(micro)
    # test_quarter_cashflow(micro)
    # test_quarter_fundamental(micro)
    test_batch_quarter_fundamental(micro)
    # === Macro ===
    macro = ak.AkShareMacro()
    # test_northbound(macro)
    # test_marginshort(macro)
    # test_shibor(macro)
    # test_index_daily(macro)
    # test_qvix_daily(macro)
    # test_industry_cls()
    # === Feature ===
    # test_fundamentalf(micro)
    # test_behavoiralf(micro)
    # test_behavoiral_qvix(macro)
    # test_northboundf(macro)
    # test_marginshortf(macro)
    # test_shiborf(macro)
    # === Database ===
