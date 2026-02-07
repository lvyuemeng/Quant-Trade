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
    Predictor,
    Processor,
    Trainer,
    TuneConfig,
)
from quant_trade.model.process import (
    DiscreteLabelBuilder,
    WalkForwardValidation,
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
    # "above_ma12",
    # "ma_cross_up",
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
    "nb_flow_z_shortnb_flow_conviction",
    "nb_flow_trendnb_flow_accel_up",
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
        (pl.col("date").diff().over("ts_code") > pl.duration(days=5))
        | (pl.col("date").diff().over("ts_code").is_null())
    )

    return merged


def merged_data(
    zfeatures: list[str], start: date, end: date
) -> tuple[pl.DataFrame, list[str]]:
    db = ArcticDB.from_config()
    pool = CNStockPool(db)
    market = CNMarket(db, source="akshare")
    fund = CNFundamental(db)
    macro = CNMacro(db)
    cons = pool.read_pool("csi1000", date=end)
    market_df = market.range_read(
        books=cons["ts_code"].to_list(), start=start, end=end
    )
    fund_df = fund.range_read(
        start=start,
        end=end,
    )
    sector = CNIndustrySectorGroup(db, factors=zfeatures, std_suffix="z")
    neu_fund_df = sector(fund_df)
    north_df = macro.read(book=MacroBook("northbound"))

    merged = stack_all(market_df, neu_fund_df, north_df)
    return (merged, sector.zfactors())


def train(data: pl.DataFrame, name: str, label: str, features: list[str]):
    # Use DiscreteLabelBuilder for LambdaRank (discrete relevance labels)
    discrete_label = DiscreteLabelBuilder(label, rank_by="date", num_bins=6)
    metric_config = MetricConfig.ranking(discrete_label)
    processor = Processor(features=features, config=metric_config)
    trainer = Trainer(processor=processor)

    data_batch = WalkForwardValidation(5, horizon_days=0, embargo_days=0).split(
        data, date_col="date"
    )
    result = trainer.batch_train(data_batch, TuneConfig())
    print(f"results: {result}")
    card = result.pack(name=name)
    store = ModelStore(FSStorage(base_dir="./model"))
    store.register(card=card, overwrite=True)


def predict(data: pl.DataFrame, name: str, label: str) -> pl.DataFrame:
    store = ModelStore(FSStorage(base_dir="./model"))
    discrete_label = DiscreteLabelBuilder(label, rank_by="date", num_bins=6)
    config = MetricConfig.ranking(discrete_label)
    predictor = Predictor.from_store(config, store, name)
    return predictor.predict(data)


if __name__ == "__main__":
    smoke_configure()
    start = date(2017, 1, 1)
    end = date(2025, 1, 1)
    data, zfactors = merged_data(SELECTED_NEUTRAL, start, end)
    train(data, name="17_25_ret_1y", label="ret_1y", features=SELECTED_NEUTRAL + SELECTED)
