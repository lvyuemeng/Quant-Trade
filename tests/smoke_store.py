import polars as pl

from quant_trade.config.arctic import ArcticDB
from quant_trade.feature.store import (
    CNFundamental,
    CNMacro,
    CNMarket,
    CNStockPool,
)

from .conftest import smoke_configure

INDEX_CODE = "000001"
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
    fund = lib.read(SHEET_YEAR, SHEET_QUATER)
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(f"fund: {fund.columns} \n  {len(fund_filtered)} {fund_filtered}")


def test_db_market(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    df = lib.read(INDEX_CODE, fresh=True)
    print(f"stock index: {df.columns} \n {df}")


def test_db_macro(db: ArcticDB):
    lib = CNMacro(db)
    index = lib.read("index")
    print(f"index: {index.columns} \n {index}")
    shibor = lib.read("shibor")
    print(f"shibor: {shibor.columns} \n {shibor}")
    marginshort = lib.read("marginshort")
    print(f"marginshort: {marginshort.columns} \n {marginshort}")
    qvix = lib.read("qvix")
    print(f"qvix: {qvix.columns} \n {qvix}")


if __name__ == "__main__":
    smoke_configure()
    db = ArcticDB.from_config()
    # test_db_stock(db)
    # test_db_fundamental(db)
    test_db_market(db)
    # test_db_macro(db)
