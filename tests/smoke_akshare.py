import polars as pl

import quant_trade.provider.akshare as ak
from quant_trade.config.arctic import ArcticAdapter, ArcticDB
from quant_trade.feature.store import CNStockMap

INDEX_CODE = "000001"
PERIOD = "daily"

SHEET_YEAR = 2021
SHEET_QUATER = 1


def test_stock_daily(m: ak.AkShareMicro):
    df = m.market_ohlcv(INDEX_CODE, period=PERIOD)
    print(f"stock daily: {df}")


def test_quarter_income(m: ak.AkShareMicro):
    df = m.quarterly_income_statement(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter income: {df.columns} \n {df}")


def test_quarter_balance(m: ak.AkShareMicro):
    df = m.quarterly_balance_sheet(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter balance: {df.columns} \n {df}")


def test_quarter_cashflow(m: ak.AkShareMicro):
    df = m.quarterly_cashflow_statement(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter cashflow: {df.columns} \n {df}")


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


def test_db_stock(db: ArcticDB):
    stock_lib = CNStockMap(db)
    stock_lib.setup()
    df = ArcticAdapter.output(stock_lib.lib.read("stock_code"))
    print(f"stock whole: {df.columns} \n {df}")
    df_2 = ArcticAdapter.output(stock_lib.lib.read("industry_code"))
    print(f":stock industry cls: {df_2.columns} \n {df_2}")


if __name__ == "__main__":
    pl.Config.set_tbl_cols(20)

    micro = ak.AkShareMicro()
    # test_stock_universe(micro)
    # test_stock_map(micro)
    # test_stock_daily(micro)
    # test_quarter_income(micro)
    # test_quarter_balance(micro)
    # test_quarter_cashflow(micro)
    macro = ak.AkShareMacro()
    # test_northbound(macro)
    # test_marginshort(macro)
    # test_shibor(macro)
    # test_index_daily(macro)
    # test_qvix_daily(macro)
    db = ArcticDB()
    test_db_stock(db)
