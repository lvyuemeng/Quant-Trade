from datetime import date
from pathlib import Path

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
    """Consistently orders columns: IDs first, then Label (if exists), then Features."""
    base_cols = ["date", "ts_code"]
    cols = [c for c in [*base_cols, label] if c in data.columns]
    return data.select([*cols, pl.all().exclude([*base_cols, label])])


def stack_all(
    market: pl.DataFrame, fund: pl.DataFrame, macro: pl.DataFrame
) -> pl.DataFrame:
    """Joins datasets using Point-in-Time (PIT) logic."""
    market = market.sort("date").with_columns(pl.col("date").cast(pl.Date))
    fund = fund.sort("notice_date").with_columns(pl.col("notice_date").cast(pl.Date))
    macro = macro.sort("date").with_columns(pl.col("date").cast(pl.Date))
    merged = market.join_asof(
        fund,
        left_on="date",
        right_on="notice_date",
        by="ts_code",
        strategy="backward",
    )
    merged = merged.join(macro, on="date", how="left")
    return merged.sort("ts_code", "date")


def merged_data(
    zfeatures: list[str], symbol: str, label: str, start: date, end: date
) -> tuple[pl.DataFrame, list[str]]:
    """Loads and merges daily data, then downsamples for modeling."""
    db = ArcticDB.from_config()
    pool = CNStockPool(db)
    market = CNMarket(db, source="baostock")
    fund = CNFundamental(db)
    macro = CNMacro(db)

    cons = pool.read_pool(symbol, date=end)  # pyright: ignore[reportArgumentType]
    ts_codes = cons["ts_code"].to_list()

    market_df = market.range_read(books=ts_codes, start=start, end=end)
    fund_df = fund.range_read(start=start, end=end)

    sector = CNIndustrySectorGroup(db, factors=zfeatures, zsuffix="z")
    neu_fund_df = sector(fund_df)
    north_df = macro.read(book=MacroBook("northbound"))

    merged = stack_all(market_df, neu_fund_df, north_df)
    merged = merged.filter(pl.col("date").dt.weekday() == 5)
    merged = reorder(merged, label)
    return (merged, sector.zfactors())


def train(data: pl.DataFrame, model_name: str, label: str, features: list[str]):
    """Standardizes the training flow."""
    print(f"Starting Training: {model_name} on {label}")

    discrete_label = DiscreteLabelBuilder(label, rank_by="date", num_bins=6)
    processor = Processor.ranking(discrete_label)
    trainer = Trainer(processor=processor, features=features)
    data_batch = WalkForwardValidation(5, horizon_days=0, embargo_days=0).split(
        data, date_col="date"
    )

    result = trainer.batch_train(data_batch, TuneConfig())
    card = result.pack(name=model_name)
    store = ModelStore(FSStorage(base_dir="./model"))
    store.register(card=card, overwrite=True)
    print(f"Model {model_name} successfully stored.")


def predict(data: pl.DataFrame, model_name: str) -> pl.DataFrame:
    """Standardizes the inference flow."""
    print(f"Starting Prediction using: {model_name}")

    store = ModelStore(FSStorage(base_dir="./model"))
    predictor = Predictor.from_store(model_name, store=store)
    scored_df = predictor.predict(data, score_name="predict_score")

    return scored_df.sort(by=["date", "predict_score"], descending=[False, True])


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    smoke_configure()
    current_label = "ret_1q"
    model_id = f"17_25_{current_label}"

    # train_start, train_end = date(2017, 1, 1), date(2024, 12, 31)
    # train_raw, z_feats = merged_data(SELECTED_NEUTRAL, "csi1000", current_label, train_start, train_end)
    # print(f"{train_raw.columns}")
    # full_features = z_feats + SELECTED
    # train(train_raw, model_id, current_label, full_features)

    pred_symbol = "csi1000"
    pred_start, pred_end = date(2025, 2, 1), date(2026, 2, 7)
    pred_raw, _ = merged_data(
        SELECTED_NEUTRAL, pred_symbol, current_label, pred_start, pred_end
    )
    results = predict(pred_raw, model_id)

    final_view = results.filter(
        pl.col("date").is_between(date(2025, 12, 1), date(2026, 2, 7))
    )
    out_dir = Path("./predict")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"output_{current_label}_{pred_symbol}.xlsx"
    final_view.to_pandas().to_excel(file_path, index=False)
    print(f"Final predictions saved to {file_path}")
