from datetime import date
from typing import Generator
import polars as pl

from quant_trade.model.base import GaussianLabelBuilder, LGBDataset, LGBRankConfig, LGBTrainer, PurgedKFold
from quant_trade.config.arctic import ArcticDB
from quant_trade.config.logger import log
from quant_trade.feature.store import CNFeatures


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
	"ch_pos",
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
	"vol_spike"
	"ac_5",
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
	"nb_flow_accel_up_northbound"
	# marginshort
	"margin_dev60_marginshort",
	"short_long_ratio_marginshort",
	"short_stress_60d_marginshort"
	# shibor
	"shibor_spread_3m_on_shibor",
	"shibor_liquidity_tighten_shibor"
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

def batch_loader(lib: CNFeatures, start_date:date, end_date:date, n_splits:int, batch_size_days=261) -> Generator[tuple[pl.DataFrame,pl.DataFrame]]:
    """Generator that yields data in batches."""
    from datetime import timedelta
    
    current_start = start_date
    while current_start <= end_date:
        current_end = min(
            current_start + timedelta(days=batch_size_days),
            end_date
        )

        batch_df = lib.load_range(start=current_start, end=current_end, ts_codes=["301391"])
        log.info(f"batch df: {batch_df.columns} \n {batch_df}")
        kfold = PurgedKFold(n_splits=2,horizon_days=0,embargo_days=0)
        for train,test in kfold.split(batch_df,date_col="date"):
            yield train, test
        
        current_start = current_end + timedelta(days=1)

db = ArcticDB.from_config()
lib = CNFeatures(db)
start_date = date(2017,2,1)
end_date = date(2020,12,31)

factor = "ret_1y"
feature_columns = SELECTED + [f"{item}_z" for item in SELECTED_NEUTRAL]

label_builder = GaussianLabelBuilder("ret_1m",rank_over="date")
dataset_builder = LGBDataset(features=feature_columns,label_cls=label_builder)
config = LGBRankConfig()
trainer = LGBTrainer(dataset=dataset_builder, config=config)
result  = trainer.train_batchwise(batch_loader(lib,start_date,end_date,n_splits=3),optimize=True)
result.model.save_model("./model/aka.txt")
print(f"results: {result}")