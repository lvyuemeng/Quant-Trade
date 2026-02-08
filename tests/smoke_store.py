import datetime
from collections.abc import Callable

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


def run_test(name: str, func: Callable, *args, **kwargs):
    """Wrapper to run tests safely and report results."""
    print(f"\n{'=' * 20} RUNNING: {name} {'=' * 20}")
    try:
        func(*args, **kwargs)
        print(f"{name}: PASSED")
    except Exception as e:
        print(f"{name}: FAILED, {e}")


# --- Test Implementations ---


def test_db_stock(db: ArcticDB):
    lib = CNStockPool(db)
    df = lib.read_pool("csi1000", date=datetime.date.today())
    print(
        f"Index: csi1000 | Rows: {len(df)} | Sample: {df.get_column('ts_code').head(3).to_list()}"
    )


def test_db_fundamental(db: ArcticDB):
    lib = CNFundamental(db)
    book = QuarterBook(year=SHEET_YEAR, quarter=SHEET_QUATER)
    fund = lib.read(book)
    # Validating data quality
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(
        f"Report: {book.key} | Columns: {len(fund.columns)} | Valid ROE: {len(fund_filtered)}/{len(fund)}"
    )


def test_db_fundamental_batch(db: ArcticDB):
    lib = CNFundamental(db)
    start, end = datetime.date(2021, 1, 1), datetime.date(2021, 12, 31)
    fund = lib.range_read(start=start, end=end)
    print(f"Batch Fundamentals [{start} to {end}] | Total Rows: {len(fund)}")


def test_db_market(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    df = lib.read(INDEX_CODE)
    print(
        f"Ticker: {INDEX_CODE} | Date Range: {df['date'].min()} to {df['date'].max()} | Rows: {len(df)}"
    )


def test_db_market_batch(db: ArcticDB):
    lib = CNMarket(db, "baostock")
    indices = [INDEX_CODE, INDEX_CODE2]

    # Check stack read
    df_stack = lib.stack_read(indices)
    print(f"Stack Read: {indices} | Combined Rows: {len(df_stack)}")

    # Check range read
    df_range = lib.range_read(
        indices, start=datetime.date(2017, 1, 1), end=datetime.date(2020, 1, 1)
    )
    print(
        f"Range Read: {len(df_range)} records across {df_range['ts_code'].n_unique()} symbols"
    )


def test_db_macro(db: ArcticDB):
    lib = CNMacro(db)
    # Testing multiple keys in one block
    keys = ("northbound", "shibor", "marginshort")
    for key in keys:
        df = lib.read(MacroBook(key), fresh=False)
        print(
            f"Macro Key: {key:12} | Rows: {len(df):6} | Latest Date: {df['date'].max() if 'date' in df.columns else 'N/A'}"
        )


# --- Main Entry Point ---

if __name__ == "__main__":
    smoke_configure()
    db = ArcticDB.from_config()
    test_db_stock(db)
    # test_db_fundamental(db)
    # test_db_fundamental_batch(db)
    # test_db_market(db)
    # test_db_market_batch(db)
    # test_db_macro(db)
