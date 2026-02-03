from datetime import date

import polars as pl

import quant_trade.provider.akshare as ak
from quant_trade.config.arctic import ArcticDB
from quant_trade.feature.process import (
    Behavioral,
    Fundamental,
    MarginShort,
    Northbound,
    Shibor,
)
from quant_trade.feature.store import (
    CNFundamental,
    CNMacro,
    CNMarket,
    CNStockPool,
)

INDEX_CODE = "000001"
PERIOD = "daily"

SHEET_YEAR = 2021
SHEET_QUATER = 1


def test_stock_daily(m: ak.AkShareMicro):
    df = m.market_ohlcv(INDEX_CODE, period=PERIOD)
    print(f"stock daily: {df}")


def test_quarter_income(m: ak.AkShareMicro):
    df = m.quarterly_income(year=SHEET_YEAR, quarter=SHEET_QUATER)
    filter_df = df.filter(
        pl.col("net_profit").is_not_nan() & pl.col("net_profit").gt(0)
    )
    print(f"quarter balance: {df.columns} \n {len(filter_df)} {filter_df}")


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
    total_stocks = df.height
    print(f"1. raw: {total_stocks}")
    print(f"   shape: {df.shape}")
    print(f"   cols: {df.columns}")

    # 2. 分别检查两个关键字段的缺失值和负值情况
    print("\n2. 'net_profit':")
    profit_stats = df.select(
        [
            pl.col("net_profit").is_null().sum().alias("null_count"),
            pl.col("net_profit").is_nan().sum().alias("nan_count"),
            pl.col("net_profit").is_not_null().sum().alias("not_null_count"),
            (pl.col("net_profit") <= 0).sum().alias("non_positive_count"),
            (pl.col("net_profit") > 0).sum().alias("positive_count"),
        ]
    )
    print(profit_stats)

    print("\n3. 'total_equity' 字段分析:")
    equity_stats = df.select(
        [
            pl.col("total_equity").is_null().sum().alias("null_count"),
            pl.col("total_equity").is_nan().sum().alias("nan_count"),
            pl.col("total_equity").is_not_null().sum().alias("not_null_count"),
            (pl.col("total_equity") <= 0).sum().alias("non_positive_count"),
            (pl.col("total_equity") > 0).sum().alias("positive_count"),
        ]
    )
    print(equity_stats)

    # 3. 检查筛选条件交集
    print("\n4. intersection:")

    # 条件1：net_profit 有效且为正
    condition1_df = df.filter(
        pl.col("net_profit").is_not_null() & pl.col("net_profit").gt(0)
    )

    # 条件2：total_equity 有效且为正
    condition2_df = df.filter(
        pl.col("total_equity").is_not_null() & pl.col("total_equity").gt(0)
    )

    # 两个条件的并集
    condition1_or_2 = df.filter(
        (pl.col("net_profit").is_not_null() & pl.col("net_profit").gt(0))
        | (pl.col("total_equity").is_not_null() & pl.col("total_equity").gt(0))
    )

    # 两个条件的交集（你的原始筛选条件）
    condition1_and_2 = df.filter(
        pl.col("net_profit").is_not_null()
        & pl.col("net_profit").gt(0)
        & pl.col("total_equity").is_not_null()
        & pl.col("total_equity").gt(0)
    )

    print(f"   仅满足'净利润>0'条件: {condition1_df.height}")
    print(f"   仅满足'股东权益>0'条件: {condition2_df.height}")
    print(f"   满足任一条件: {condition1_or_2.height}")
    print(f"   同时满足两个条件（你的筛选）: {condition1_and_2.height}")

    # 4. 随机采样查看具体数据
    print("\n5. 随机采样示例 (前5行):")
    if not condition1_and_2.is_empty():
        sample = condition1_and_2.head(5)
        print(sample.select(["net_profit", "total_equity"]))
    else:
        print("   没有符合条件的数据")
    filter_df = df.filter(
        pl.col("net_profit").is_not_nan()
        & pl.col("net_profit").gt(0)
        & pl.col("total_equity").is_not_nan()
        & pl.col("total_equity").gt(0)
    )
    print(f"quarter balance: {df.columns} \n {len(filter_df)} {filter_df}")


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


# ===


def test_db_stock(db: ArcticDB):
    stock_lib = CNStockPool(db)
    stock_codes = stock_lib.read("stock_code")
    print(f"stock code: {stock_codes}")
    indus_codes = stock_lib.read("industry_code")
    print(f"stock code: {indus_codes.sample(30)}")


def test_db_fundamental(db: ArcticDB):
    lib = CNFundamental(db)
    fund = lib.read(SHEET_YEAR, SHEET_QUATER)
    fund_filtered = fund.filter(pl.col("roe").is_not_nan())
    print(f"fund: {fund.columns} \n  {len(fund_filtered)} {fund_filtered}")


def test_db_market(db: ArcticDB):
    lib = CNMarket(db)
    df = lib.read(INDEX_CODE)
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


def test_db_features(db: ArcticDB):
    lib = CNFeatures(db)
    feat = lib.load_range(date(2024, 1, 1), date(2025, 1, 1), ["301391"])
    print(f"feature: {feat.columns} \n {feat}")


if __name__ == "__main__":
    pl.Config(tbl_cols=20, tbl_rows=20)

    # === Micro ===
    micro = ak.AkShareMicro()
    # test_stock_daily(micro)
    # test_quarter_income(micro)
    # test_quarter_balance(micro)
    # test_quarter_cashflow(micro)
    # test_quarter_fundamental(micro)
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
    db = ArcticDB.from_config()
    # test_db_stock(db)
    # test_db_fundamental(db)
    # test_db_market(db)
    # test_db_macro(db)
    test_db_features(db)
