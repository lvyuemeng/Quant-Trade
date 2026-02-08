from datetime import date

import polars as pl

from quant_trade.config.arctic import ArcticDB
from quant_trade.config.logger import debug_null_profile
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
    "vola_regime_ratio",
    "vola_parkinson",
    "atr_pct",
    # volume
    "vol_dev20",
    "vol_dev60",
    "vol_spike",
    "ac_5",
    # price structure
    "near_high",
    "near_low",
    # Fundamental
    "profit_growth_premium",
    # Macro
    # northbound
    "nb_flow_z_shortnb_flow_conviction",
    "nb_flow_trendnb_flow_accel_up",
    "nb_flow_shock_nb_flow_z_longnb_cum_z_trend",
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
    "quick_ratio",
]


def reorder(data: pl.DataFrame, label: str) -> pl.DataFrame:
    return data.select(
        [
            pl.col("date"),
            pl.col("ts_code"),
            pl.col(label),
            pl.all().exclude("date", "ts_code", label),
        ]
    )


def stack_all(
    market: pl.DataFrame, fund: pl.DataFrame, macro: pl.DataFrame
) -> pl.DataFrame:
    market = market.sort(by="date").with_columns(pl.col("date").cast(pl.Date))
    fund = fund.sort(by=["ts_code", "notice_date"]).with_columns(
        pl.col("notice_date").cast(pl.Date)
    )
    macro = macro.sort(by=["date"]).with_columns(pl.col("date").cast(pl.Date))

    merged = market.join_asof(
        fund,
        left_on="date",
        right_on="notice_date",
        by=["ts_code"],
        strategy="backward",
    )
    merged = merged.join(macro, on="date", how="inner")
    merged = merged.sort("ts_code", "date")
    print(f"debug null profile {debug_null_profile(merged)}")

    # merged = merged.filter(
    # (pl.col("date").diff().over("ts_code") > pl.duration(days=5))
    # | (pl.col("date").diff().over("ts_code").is_null())
    # )

    # print(f"merged: {merged.filter(pl.col("date") > date(2025,12,1))}")
    return merged


def merged_data(
    zfeatures: list[str],symbol:str, label: str, start: date, end: date
) -> tuple[pl.DataFrame, list[str]]:
    db = ArcticDB.from_config()
    pool = CNStockPool(db)
    market = CNMarket(db, source="akshare")
    fund = CNFundamental(db)
    macro = CNMacro(db)
    cons = pool.read_pool(symbol, date=end) # pyright: ignore[reportArgumentType]
    ts_codes = cons["ts_code"].to_list()
    market_df = (
            market.range_read(books=ts_codes, start=start, end=end)
            .filter(pl.col("date").dt.weekday() == 5)
        )
    fund_df = fund.range_read(start=start, end=end)
    sector = CNIndustrySectorGroup(db, factors=zfeatures, zsuffix="z")
    neu_fund_df = sector(fund_df)
    north_df = macro.read(book=MacroBook("northbound"))

    merged = stack_all(market_df, neu_fund_df, north_df)
    merged = reorder(merged, label=label).filter(pl.col("date").dt.weekday() == 5)
    # sample_code = "000012"
    # check = merged.filter(pl.col("ts_code") == sample_code).select(["date", "notice_date", "roe"])
    # print(check.tail(20))
    # print(merged.filter(pl.col("ts_code") == "000012").select(pl.col("notice_date").min()))
    # full_info_sample = merged.filter(
    #     pl.col("roe").is_not_null() & 
    #     pl.col("gross_margin").is_not_null()
    # ).head(10)

    # print("Rows where fundamentals are fully present:")
    # print(full_info_sample.select(["date", "ts_code", "roe", "gross_margin"]))
    return (merged, sector.zfactors())


def train(data: pl.DataFrame, name: str, label: str, features: list[str]):
    # Use DiscreteLabelBuilder for LambdaRank (discrete relevance labels)
    discrete_label = DiscreteLabelBuilder(label, rank_by="date", num_bins=6)
    metric_config = MetricConfig.ranking(discrete_label)
    processor = Processor(idents=["ts_code"], features=features, config=metric_config)
    trainer = Trainer(processor=processor)

    data_batch = WalkForwardValidation(5, horizon_days=0, embargo_days=0).split(
        data, date_col="date"
    )
    result = trainer.batch_train(data_batch, TuneConfig())
    print(f"{result}")
    card = result.pack(name=name)
    store = ModelStore(FSStorage(base_dir="./model"))
    store.register(card=card, overwrite=True)


def predict(data: pl.DataFrame, name: str, label: str) -> pl.DataFrame:
    store = ModelStore(FSStorage(base_dir="./model"))
    discrete_label = DiscreteLabelBuilder(label, rank_by="date", num_bins=6)
    config = MetricConfig.ranking(discrete_label)
    predictor = Predictor.from_store(["ts_code"], config, store, name)
    return predictor.predict(data, score_name="predict_score").sort(
        by="predict_score", descending=True
    )


if __name__ == "__main__":
    # smoke_configure()
    label = "ret_1m"
    # pl.Config(tbl_cols=20, tbl_rows=100)
    # start = date(2016, 9, 1)
    # end = date(2025, 1, 1)
    # data, zfactors = merged_data(SELECTED_NEUTRAL, "csi1000",label, start, end)
    # train(
    #     data, name=f"17_25_{label}", label=label, features=zfactors + SELECTED
    # )
    symbol = "ssmi"
    start = date(2025, 2, 1)
    end = date(2026, 2, 7)
    data, zfactors = merged_data(SELECTED_NEUTRAL,symbol, label, start, end)
    data = reorder(data, label)
    res = predict(data=data, name=f"17_25_{label}", label=label)
    res = res.filter(pl.col("date").is_between(date(2025, 12, 1), date(2026, 2, 7)))
    print(f"{res.columns} \n {res}")
    res.to_pandas().to_excel(f"./predict/predict_{label}_{symbol}.xlsx",)
