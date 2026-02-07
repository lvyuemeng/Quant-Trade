import datetime

import polars as pl

from quant_trade.config.arctic import ArcticDB
from quant_trade.feature.query import MacroBook, QuarterBook
from quant_trade.feature.store import (
    CNFundamental,
    CNMacro,
    CNMarket,
    CNStockPool,
)
from tests.conftest import smoke_configure

INDEX_CODE = "000001"
INDEX_CODE2 = "000002"
PERIOD = "daily"

SHEET_YEAR = 2021
SHEET_QUATER = 1


def test_db_stock(db: ArcticDB):
    lib = CNStockPool(db)
    # stock_codes = lib.read_codes("stock_code")
    # print(f"stock code: {stock_codes}")
    # indus_codes = lib.read_codes("industry_code")
    # print(f"stock code: {indus_codes.sample(30)}")
    csi1000 = lib.read_pool("csi1000", date=datetime.date.today())
    print(f"csi1000 {csi1000}")


def test_db_fundamental(db: ArcticDB):
    lib = CNFundamental(db)
    book = QuarterBook(year=SHEET_YEAR, quarter=SHEET_QUATER)
    fund = lib.read(book)
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(f"fund: {fund.columns} \n  {len(fund_filtered)} {fund_filtered}")


def test_db_fundamental_batch(db: ArcticDB):
    lib = CNFundamental(db)
    fund = lib.range_read(
        start=datetime.date(2021, 1, 1), end=datetime.date(2021, 12, 31)
    )
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(f"fund: {fund.columns} \n  {len(fund_filtered)} {fund_filtered}")


def test_db_market(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    df = lib.read(INDEX_CODE)
    print(f"market: {df.columns} \n {df}")


def test_db_market_batch(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    indices = [INDEX_CODE, INDEX_CODE2]
    df = lib.stack_read(indices)
    print(f"batch market: {df.columns} \n {df}")
    df = lib.range_read(
        indices, start=datetime.date(2017, 1, 1), end=datetime.date(2020, 1, 1)
    )
    print(f"batch market range: {df.columns} \n {df}")


def test_db_macro(db: ArcticDB):
    lib = CNMacro(db)
    nf = lib.read(MacroBook("northbound"), fresh=True)
    print(f"northbound: {nf}")
    shibor = lib.read(MacroBook("shibor"), fresh=True)
    print(f"shibor: {shibor}")
    marginshort = lib.read(MacroBook("marginshort"), fresh=True)
    print(f"marginshort: {marginshort}")


if __name__ == "__main__":
    smoke_configure()
    db = ArcticDB.from_config()
    test_db_stock(db)
    # test_db_fundamental(db)
    # test_db_fundamental_batch(db)
    # test_db_market(db)
    # test_db_market_batch(db)
    # test_db_macro(db)
