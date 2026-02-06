from datetime import date

import polars as pl

from quant_trade.config.arctic import ArcticDB
from quant_trade.feature.query import MacroBook
from quant_trade.feature.store import (
    CNFundamental,
    CNIndustrySectorGroup,
    CNMacro,
    CNMarket,
    CNStockPool,
)
from quant_trade.model.lgb import (
    MetricConfig,
    Processor,
    Trainer,
    TuneConfig,
)
from quant_trade.model.process import (
    DiscreteLabelBuilder,
    PurgedKFold,
)
from quant_trade.model.store import FSStorage, ModelStore
from tests.conftest import smoke_configure

LABEL = [
    "ret_1m",
    "ret_1q",
    "ret_1y",
]

SELECTED = [
    # Behavoiral
    # MA deviation
    "ma12_dev",
    "ma24_dev",
    # trend signal
    "above_ma12",
    "ma_cross_up",
    # rate of change
    "roc_4",
    "roc_12",
    # volatility
    "vola_20",
    "vola_regime_raio",
    "vola_parkinson",
    "atr_pct",
    # volume
    "vol_dev20",
    "vol_dev60",
    "vol_spikeac_5",
    # price structure
    "near_high",
    "near_low",
    # Fundamental
    "profit_growth_premium",
    # Macro
    # northbound
    "nb_flow_z_short_northbound"
    "nb_flow_conviction_northbound"
    "nb_flow_trend_northbound"
    "nb_flow_accel_up_northbound",
    # marginshort
    # "margin_dev60_marginshort",
    # "short_long_ratio_marginshort",
    # "short_stress_60d_marginshort"
    # shibor
    # "shibor_spread_3m_on_shibor",
    # "shibor_liquidity_tighten_shibor",
]

SELECTED_NEUTRAL = [
    "roe",
    "gross_margin",
    "net_profit_growth_yoy",
    "asset_turnover",
    "accrual_ratio",
    "current_ratio",
    "quick_raio",
]


def stack_all(
    market: pl.DataFrame, fund: pl.DataFrame, macro: pl.DataFrame
) -> pl.DataFrame:
    market = (
        market.sort(by="date").with_columns(pl.col("date").cast(pl.Date)).sort("date")
    )
    fund = (
        fund.sort(by=["notice_date", "ts_code"])
        .with_columns(pl.col("notice_date").cast(pl.Date))
        .sort("notice_date")
    )
    macro = (
        macro.sort(by=["date"]).with_columns(pl.col("date").cast(pl.Date)).sort("date")
    )

    merged = market.join_asof(
        fund,
        left_on="date",
        right_on="notice_date",
        by=["ts_code"],
        strategy="backward",
    )
    merged = merged.join(macro, on="date", how="inner")
    merged = merged.sort("ts_code", "date")

    merged = merged.filter(
        (pl.col("date").diff().over("ts_code") > pl.duration(days=10))
        | (pl.col("date").diff().over("ts_code").is_null())
    )

    return merged


if __name__ == "__main__":
    smoke_configure()
    db = ArcticDB.from_config()
    pool = CNStockPool(db)
    market = CNMarket(db, source="baostock")
    fund = CNFundamental(db)
    macro = CNMacro(db)

    start_date = date(2017, 2, 1)
    end_date = date(2020, 12, 31)

    cons = pool.read_pool("csi500", date=end_date)
    market_df = market.range_read(
        books=cons["ts_code"].to_list(), start=start_date, end=end_date
    )
    fund_df = fund.range_read(
        start=start_date,
        end=end_date,
    )
    sector = CNIndustrySectorGroup(db, factors=SELECTED_NEUTRAL, std_suffix="_z")
    neu_fund_df = sector(fund_df)
    north_df = macro.read(book=MacroBook("northbound"))

    merged = stack_all(market_df, neu_fund_df, north_df)
    print(f"{len(merged)}")
    factor_col = "ret_1m"
    feature_cols = SELECTED + sector.zfactors()

    # Use DiscreteLabelBuilder for LambdaRank (discrete relevance labels)
    discrete_label = DiscreteLabelBuilder("ret_1m", rank_by="date", num_bins=4)
    metric_config = MetricConfig.ranking(discrete_label)
    processor = Processor(features=feature_cols, config=metric_config)
    trainer = Trainer(processor=processor)

    data_batch = PurgedKFold(5, horizon_days=0, embargo_days=0).split(
        merged, date_col="date"
    )
    result = trainer.batch_train(data_batch, TuneConfig())
    print(f"results: {result}")
    card = result.pack(name="test")
    store = ModelStore(FSStorage(base_dir="./model"))
    store.register(card=card)
