import datetime

import polars as pl

from quant_trade.config.arctic import ArcticDB
from quant_trade.feature.store import (
    CNFundamental,
    CNMacro,
    CNMarket,
    CNStockPool,
)
from quant_trade.feature.query import MacroBook, QuarterBook, TickerBook
from tests.conftest import smoke_configure

INDEX_CODE = "000001"
INDEX_CODE2 = "000002"
PERIOD = "daily"

SHEET_YEAR = 2021
SHEET_QUATER = 1


def test_db_stock(db: ArcticDB):
    stock_lib = CNStockPool(db)
    stock_codes = stock_lib.read_codes("stock_code")
    print(f"stock code: {stock_codes}")
    indus_codes = stock_lib.read_codes("industry_code")
    print(f"stock code: {indus_codes.sample(30)}")

def test_db_fundamental(db: ArcticDB):
    lib = CNFundamental(db)
    book = QuarterBook(year=SHEET_YEAR, quarter=SHEET_QUATER)
    fund = lib.read(book)
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(f"fund: {fund.columns} \n  {len(fund_filtered)} {fund_filtered}")


def test_db_fundamental_batch(db: ArcticDB):
    lib = CNFundamental(db)
    fund = lib.range_read(
        start=datetime.date(2021, 1, 1), end=datetime.date(2021, 12, 31))
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(f"fund: {fund.columns} \n  {len(fund_filtered)} {fund_filtered}")


def test_db_market(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    df = lib.read(TickerBook(INDEX_CODE), fresh=True)
    print(f"stock index: {df.columns} \n {df}")

def test_db_market_batch(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    indices = [TickerBook(INDEX_CODE),TickerBook(INDEX_CODE2)]
    dfs = lib.batch_read(indices, fresh=True)
    df = pl.concat(dfs,how="vertical_relaxed")
    print(f"stock index: {df.columns} \n {df}")

def test_db_macro(db: ArcticDB):
    lib = CNMacro(db)
    index = lib.read(MacroBook("index"))
    print(f"index: {index.columns} \n {index}")
    qvix = lib.read(MacroBook("qvix"))
    print(f"qvix: {qvix.columns} \n {qvix}")
    shibor = lib.read(MacroBook("shibor"))
    print(f"shibor: {shibor.columns} \n {shibor}")
    marginshort = lib.read(MacroBook("marginshort"))
    print(f"marginshort: {marginshort.columns} \n {marginshort}")


if __name__ == "__main__":
    smoke_configure()
    db = ArcticDB.from_config()
    # test_db_stock(db)
    # test_db_fundamental(db)
    # test_db_fundamental_batch(db)
    # test_db_market(db)
    # test_db_market_batch(db)
    # test_db_macro(db)
