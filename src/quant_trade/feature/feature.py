"""Feature engineering module for quantitative trading.

Implements the feature calculation pipeline as specified in docs/table.md,
including quality metrics, leverage ratios, growth metrics, valuation ratios,
behavioral indicators, and macro context features.
"""

from typing import Literal

import numpy as np
import polars as pl

from quant_trade.utils.logger import log

type Freq = Literal["daily", "weekly", "quarterly", "yearly"]


def _resample(df: pl.DataFrame, rules: dict[str, pl.Expr], freq: Freq) -> pl.DataFrame:
    if freq == "daily":
        return df
    aggs = [rules[c].alias(c) for c in rules.keys() if c in df.columns]
    freq_map = {"W": "1w", "M": "1mo", "Q": "3mo", "Y": "1y"}
    if (date_range := freq_map.get(freq)) is None:
        raise ValueError(freq)

    return df.group_by_dynamic("date", every=date_range).agg(aggs).sort("date")


class Fundamental:
    """Feature engineering class for fundamental metrics (profitability, growth, cost structure).

    This class requires the input `DataFrame` to contain specific columns that correspond to
    standardized financial statement items. The columns are grouped into logical categories
    for clarity.

    **REQUIRED COLUMNS** in the input `DataFrame`:

    ```
    Core Identifiers:                (CN)
        ts_code (str):               股票代码
        name (str):                  股票简称
        announcement_date (date):    公告日期

    Profit & Revenue (Core Performance):
        net_profit (float):          净利润
        operating_profit (float):    营业利润
        total_profit (float):        利润总额
        total_revenue (float):       营业总收入

    Growth Rates (Year-over-Year):
        net_profit_yoy (float):      净利润同比
        total_revenue_yoy (float):   营业总收入同比

    Cost Structure (Operating Expenses Breakdown):
        operating_cost (float):      营业总支出-营业支出
        selling_cost (float):        营业总支出-销售费用
        admin_cost (float):          营业总支出-管理费用
        finance_cost (float):        营业总支出-财务费用
        total_cost (float):          营业总支出-营业总支出

    Assets:
        cash (float):                资产-货币资金
        accounts_receivable (float): 资产-应收账款
        inventory (float):           资产-存货
        total_assets (float):        资产-总资产
        total_assets_yoy (float):    资产-总资产同比 (%)

    Liabilities:
        accounts_payable (float):    负债-应付账款
        advance_receipts (float):    负债-预收账款
        total_debts (float):         负债-总负债
        total_debts_yoy (float):     负债-总负债同比 (%)

    Ratios & Equity:
        debt_to_assets (float):      资产负债率 (%)
        total_equity (float):        股东权益合计

    Cash Flow Components:
        net_cashflow (float):        净现金流-净现金流
        net_cashflow_yoy (float):    净现金流-同比增长
        cfo (float):                 经营性现金流-现金流量净额
        cfo_share (float):           经营性现金流-净现金流占比
        cfi (float):                 投资性现金流-现金流量净额
        cfi_share (float):           投资性现金流-净现金流占比 (%)
        cff (float):                 融资性现金流-现金流量净额
        cff_share (float):           融资性现金流-净现金流占比 (%)

    All monetary values are expected to be in unit of its currency unless otherwise specified.
    Percentage values are in standard percentage units (e.g., 10.5 for 10.5%).
    ```
    """

    IDENT_COLS = ["ts_code", "name", "announcement_date"]

    @staticmethod
    def quality_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate quality metrics."""
        log.info("Calculating quality metrics")

        metric = df.select(
            Fundamental.IDENT_COLS,
            [
                (pl.col("net_profit") / pl.col("total_equity")).alias("roe"),
                (pl.col("net_profit") / pl.col("total_assets")).alias("roa"),
                (pl.col("gross_profit") / pl.col("total_revenue")).alias(
                    "gross_margin"
                ),
                (pl.col("operatig_profit") / pl.col("total_revenue")).alias(
                    "operating_margin"
                ),
                (pl.col("net_profit") / pl.col("total_revenue")).alias("net_margin"),
                (pl.col("net_profit") / pl.col("total_revenue")).alias("net_margin"),
            ],
        )
        return metric

    @staticmethod
    def leverage_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate leverage and safety metrics."""
        log.info("Calculating leverage metrics")

        metric = df.select(
            Fundamental.IDENT_COLS,
            "total_debts",
            [
                (
                    (pl.col("total_debts") / pl.col("total_equity")).alias(
                        "debt_to_equity"
                    )
                ),
                (
                    (
                        (
                            pl.col("cash")
                            + pl.col("accounts_receivable")
                            + pl.col("inventory")
                        )
                        / (pl.col("accounts_payable") + pl.col("advance_receipts"))
                    ).alias("current_ratio")
                ),
                (
                    (pl.col("cash") + pl.col("accounts_receivable"))
                    / (pl.col("accounts_payable") + pl.col("advance_receipts")).alias(
                        "quick_ratio"
                    )
                ),
                (
                    pl.col("operating_profit")
                    / pl.col("finance_cost").alias("interest_coverage")
                ),
            ],
        )
        return metric

    @staticmethod
    def growth_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate growth metrics."""
        log.info("Calculating growth metrics")
        metric = df.select(
            [
                Fundamental.IDENT_COLS,
                pl.col("total_assets_yoy"),
                pl.col("total_debts_yoy"),
                (pl.col("net_profit_yoy") - pl.col("total_revenue_yoy")).alias(
                    "profit_growth_premium"
                ),
            ]
        )
        return metric

    @staticmethod
    def value_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate valuation metrics."""
        log.info("Calculating valuation metrics")

        metric = df.select(
            [
                Fundamental.IDENT_COLS,
                (pl.col("cfo") / pl.col("net_profit")).alias("cfo_to_net_profit"),
                (pl.col("cfo") + pl.col("cfi")).alias("cf_free"),
                (pl.col("cfo") / pl.col("total_assets")).alias("cf_adequacy"),
                (pl.col("cfo_share")),
                (pl.col("cfi").abs() / pl.col("total_assets")).alias(
                    "invest_intensity"
                ),
            ]
        )
        return metric


class Behavioral:
    """
    Feature engineering class for behavioral (market microstructure and price action) metrics.

    This class processes market data to generate technical indicators, price patterns,
    and market microstructure features for quantitative analysis.

    REQUIRED COLUMNS in the input DataFrame for behavioral analysis:

    Core Market Data (OHLCV):     (CN)
        date (date):              交易日日期
        ts_code (str):            股票/指数代码
        open (float):             开盘价
        high (float):             最高价
        low (float):              最低价
        close (float):            收盘价
        volume (float/int):       成交量 (股/手)
        amount (float):           成交额 (元)

    Derived Market Metrics:
        amplitude (float):        振幅 [(high-low)/prev_close]
        pct_chg (float):          涨跌幅 (%)
        change (float):           涨跌额 (close - prev_close)
        turnover_rate (float):    换手率 (%)

    For Index Data:
        date (date):              日期
        open (float):             开盘价
        high (float):             最高价
        low (float):              最低价
        close (float):            收盘价
        volume (float/int):       成交量 (可选)

    For QVIX Volatility Data:
        date (date):              日期
        open (float):             开盘价 (波动率指数)
        high (float):             最高价 (波动率指数)
        low (float):              最低价 (波动率指数)
        close (float):            收盘价 (波动率指数)

    Note: All price data should be in consistent currency units (typically yuan).
    Volume and amount data should be in appropriate units as specified.
    """

    IDENT_COLS = [
        "ts_code",
        "date",
    ]

    @staticmethod
    def momentum(df: pl.DataFrame) -> pl.DataFrame:
        """
        Captures Trend Strength, Reversion, and Efficiency.
        """
        return df.select(
            Behavioral.IDENT_COLS,
            [
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1"),
                (pl.col("close") / pl.col("close").shift(4) - 1).alias("ret_4"),
                (pl.col("close") / pl.col("close").shift(12) - 1).alias("ret_12"),
                pl.col("close").rolling_mean(4).alias("ma_4"),
                pl.col("close").rolling_mean(12).alias("ma_12"),
                pl.col("close").rolling_mean(24).alias("ma_24"),
                (pl.col("close") / pl.col("ma_12") - 1).alias("ma12_dev"),
                (pl.col("close") / pl.col("ma_24") - 1).alias("ma24_dev"),
                pl.col("close").pct_change(4).alias("roc_4"),
                pl.col("close").pct_change(12).alias("roc_12"),
                pl.col("high").rolling_max(20).alias("ch_high"),
                pl.col("low").rolling_min(20).alias("ch_low"),
                (
                    (pl.col("close") - pl.col("ch_low"))
                    / (pl.col("ch_high") - pl.col("ch_low"))
                ).alias("ch_pos"),
                (pl.col("close") > pl.col("ma_12")).cast(pl.Int8).alias("above_ma12"),
                (pl.col("ma_4") > pl.col("ma_12")).cast(pl.Int8).alias("ma_cross_up"),
            ],
        )

    @staticmethod
    def volatiity(df: pl.DataFrame) -> pl.DataFrame:
        """
        Captures Variance, Range, and Regime.
        """
        df = df.with_columns(pl.col("close").log().diff().alias("log_ret_1"))
        # Parkinson Volatility (High-Low based, more efficient than Close-Close)
        parkinson = (
            (pl.col("high") / pl.col("low")).log() ** 2 / (4 * np.log(2))
        ).sqrt()

        # Garman-Klass Volatility (Includes Open-Close information)
        gk_comp1 = 0.5 * ((pl.col("high") / pl.col("low")).log() ** 2)
        gk_comp2 = (2 * np.log(2) - 1) * ((pl.col("close") / pl.col("open")).log() ** 2)
        gk = (gk_comp1 - gk_comp2).sqrt()

        # True Range
        tr = pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            ]
        )
        return df.select(
            Behavioral.IDENT_COLS,
            [
                parkinson.alias("vola_parkinson"),
                gk.alias("vola_gk"),
                # Realized Volatility (Standard Deviation of returns)
                pl.col("log_ret_1").rolling_std(20).alias("vola_20"),
                pl.col("log_ret_1").rolling_std(60).alias("vola_60"),
                # Volatility Regime (Current Vol / Long-term Vol) - Mean Reversion Signal
                (
                    pl.col("log_ret_1").rolling_std(20)
                    / (pl.col("log_ret_1").rolling_std(60) + 1e-9)
                ).alias("vola_regime_ratio"),
                # Normalized ATR
                (tr.rolling_mean(14) / pl.col("close")).alias("atr_pct"),
            ],
        )

    @staticmethod
    def volume(df: pl.DataFrame) -> pl.DataFrame:
        """
        Captures Liquidity shocks and VWAP deviations.
        """
        if "volume" not in df.columns:
            raise ValueError(
                f"volume is not in the given data frame. columns: \n {df.columns}"
            )

        vol_ma20 = pl.col("volume").rolling_mean(20)

        exprs = [
            # 1. Volume Trend
            (pl.col("volume") / vol_ma20 - 1).alias("vol_dev20"),
            (pl.col("volume") / pl.col("volume").rolling_mean(60) - 1).alias(
                "vol_dev60"
            ),
            # 2. Volume Shock
            (pl.col("volume") > vol_ma20 * 2).cast(pl.Int8).alias("vol_spike"),
        ]

        if "amount" in df.columns:
            avg_price = pl.col("amount") / (pl.col("volume") + 1e-9)
            exprs.append((pl.col("close") / avg_price - 1).alias("vwap_dev_1d"))
            exprs.append((pl.col("amount") / pl.col("volume")).alias("avg_trade_price"))

        # Return only the new calculated columns
        return df.select(Behavioral.IDENT_COLS, exprs)

    @staticmethod
    def structure(df: pl.DataFrame) -> pl.DataFrame:
        """
        Captures Time-Series properties: Autocorrelation and Variance Ratio.
        CRITICAL FIX: Removed full-series look-ahead bias using rolling windows.
        """
        df = df.with_columns(pl.col("close").log().diff().alias("log_ret_1"))
        return df.select(
            Behavioral.IDENT_COLS,
            [
                (
                    pl.col("close").log().diff(2).rolling_var(60)
                    / (pl.col("log_ret_1").rolling_var(60) * 2 + 1e-9)
                ).alias("vr_2"),
                pl.rolling_corr(
                    pl.col("log_ret_1"), pl.col("log_ret_1").shift(1), window_size=60
                ).alias("ac_1"),
                pl.corr(pl.col("log_ret_1"), pl.col("log_ret_1").shift(5)).alias(
                    "ac_5"
                ),
                (pl.col("close") / pl.col("high").rolling_max(50) - 1).alias(
                    "near_high"
                ),
                (pl.col("close") / pl.col("low").rolling_min(50) - 1).alias("near_low"),
                (
                    (pl.col("open") - pl.col("close").shift(1))
                    / pl.col("close").shift(1)
                ).alias("gap"),
            ],
        )


class Northbound:
    """
    Feature extractor for northbound capital flow (沪深港通 北向资金).

    Accepted input columns:
    - date (date)
    - net_buy (float): 当日成交净买额
    - fund_inflow (float): 当日资金流入
    - cum_net_by (float): 历史累计净买额

    Returns:
    - flow momentum
    - flow volatility
    - flow regime indicators
    """

    IDENT_COLS = ["date"]

    @staticmethod
    def flow(df: pl.DataFrame) -> pl.DataFrame:
        z = (pl.col("net_buy") - pl.col("net_buy").rolling_mean(20)) / (
            pl.col("net_buy").rolling_std(60)
        )

        return df.select(
            Northbound.IDENT_COLS,
            [
                z.alias("nb_flow_z"),
                z.abs().alias("nb_flow_abs_z"),
                (z > 0).cast(pl.Int8).alias("nb_flow_pos"),
                (z.abs() > 2).cast(pl.Int8).alias("nb_flow_shock"),
            ],
        )

    @staticmethod
    def persistence(df: pl.DataFrame) -> pl.DataFrame:
        signed_strength = pl.col("net_buy").sign() * pl.col(
            "net_buy"
        ).abs().rolling_mean(5)

        return df.select(
            Northbound.IDENT_COLS,
            [
                signed_strength.alias("nb_flow_conviction"),
                signed_strength.rolling_mean(20).alias("nb_flow_trend"),
            ],
        )

    @staticmethod
    def accel(df: pl.DataFrame) -> pl.DataFrame:
        accel = pl.col("net_buy").diff().diff()

        return df.select(
            Northbound.IDENT_COLS,
            [
                accel.alias("nb_flow_accel"),
                (accel > 0).cast(pl.Int8).alias("nb_flow_accel_up"),
            ],
        )


class MarginShort:
    """
    Feature extractor for total A-share margin & short balance (SH + SZ).

    Accepted input columns:
    - date
    - margin_balance
    - margin_buy_amount
    - short_sell_volume
    - short_balance
    - total_margin_balance

    Returns:
    - leverage level
    - leverage impulse
    - leverage regime indicators
    """

    IDENT_COLS = ["date"]

    @staticmethod
    def level(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            MarginShort.IDENT_COLS,
            [
                pl.col("total_margin_balance").alias("margin_total"),
                pl.col("total_margin_balance").rolling_mean(20).alias("margin_ma20"),
                (
                    pl.col("total_margin_balance")
                    / pl.col("total_margin_balance").rolling_mean(60)
                    - 1
                ).alias("margin_dev60"),
            ],
        )

    @staticmethod
    def impulse(df: pl.DataFrame) -> pl.DataFrame:
        delta = pl.col("total_margin_balance").diff()

        return df.select(
            MarginShort.IDENT_COLS,
            [
                delta.alias("margin_delta"),
                delta.diff().alias("margin_accel"),
                delta.rolling_mean(5).alias("margin_impulse5"),
                delta.rolling_mean(20).alias("margin_impulse20"),
            ],
        )

    @staticmethod
    def stress(df: pl.DataFrame) -> pl.DataFrame:
        short_long_ratio = (
            pl.col("short_sell_volume") / pl.col("margin_buy_amount")
        ).alias("short_long_ratio")

        short_leverage_ratio = (
            pl.col("short_balance") / pl.col("total_margin_balance")
        ).alias("short_leverage_ratio")

        return df.select(
            MarginShort.IDENT_COLS,
            [
                short_long_ratio,
                short_leverage_ratio,
                (
                    pl.col("short_sell_volume")
                    > pl.col("short_sell_volume").rolling_mean(20) * 1.5
                )
                .cast(pl.Int8)
                .alias("short_pressure_20d"),
                (
                    pl.col("short_sell_volume")
                    > pl.col("short_sell_volume").rolling_mean(60) * 1.5
                )
                .cast(pl.Int8)
                .alias("short_pressure_60d"),
                (short_leverage_ratio > short_leverage_ratio.rolling_mean(60))
                .cast(pl.Int8)
                .alias("short_stress_60d"),
            ],
        )


class Shibor:
    """
    Feature extractor for SHIBOR funding rates.

    Accepted columns:
    - date
    - ON_rate, 1W_rate, 1M_rate, 3M_rate, 1Y_rate

    Focus:
    - funding shock
    - term structure compression
    - liquidity stress
    """

    IDENT_COLS = ["date"]

    @staticmethod
    def funding_shock(df: pl.DataFrame) -> pl.DataFrame:
        # df = Shibor._resample(df, freq)

        on_change = pl.col("ON_rate").diff()
        shock = on_change - on_change.rolling_mean(20)

        return df.select(
            Shibor.IDENT_COLS,
            [
                shock.alias("shibor_on_shock"),
                (shock > 0).cast(pl.Int8).alias("shibor_liquidity_tighten"),
            ],
        )

    @staticmethod
    def curve_structure(df: pl.DataFrame) -> pl.DataFrame:
        curve_std = pl.concat_list(
            [
                pl.col("ON_rate"),
                pl.col("1W_rate"),
                pl.col("1M_rate"),
                pl.col("3M_rate"),
                pl.col("1Y_rate"),
            ]
        ).list.std()

        return df.select(
            Shibor.IDENT_COLS,
            [
                (pl.col("3M_rate") - pl.col("ON_rate")).alias("shibor_term_spread"),
                curve_std.alias("shibor_curve_dispersion"),
            ],
        )


class CrossSection:
    """
    Cross-Sectional Operations (Peer/Market context).
    Strictly uses 'group_by' (context).
    """

    @staticmethod
    def winsorize(
        col: pl.Expr | str,
        by: list[str] = ["date"],
        limits: tuple[float, float] = (0.01, 0.99),
    ) -> pl.Expr:
        """
        Clips outliers based on peer distribution at that specific moment.
        """
        c = pl.col(col) if isinstance(col, str) else col

        llimit, ulimit = limits
        lower = c.quantile(llimit).over(by)
        upper = c.quantile(ulimit).over(by)

        return c.clip(lower, upper).alias(c.meta.output_name())

    @staticmethod
    def standardize(col: pl.Expr | str, by: list[str]) -> pl.Expr:
        """
        Z-Score Normalization relative to peers/groups.
        (Value - GroupMean) / GroupStd
        """
        c = pl.col(col) if isinstance(col, str) else col

        mu = c.mean().over(by)
        sigma = c.std().over(by)

        return ((c - mu) / (sigma + 1e-9)).alias(c.meta.output_name())


class SectorGroup:
    """
    Orchestrates the normalization of Industry-Neutral Fundamental Factors.
    """

    @staticmethod
    def normalize(
        df: pl.DataFrame,
        sector: pl.DataFrame,
        factors: list[str],
        by: list[str],
        limits: tuple[float, float] = (0.02, 0.98),
    ) -> pl.DataFrame:
        merged = df.join(sector, on="ts_code", how="semi")
        ops: list[pl.Expr] = []
        for col in factors:
            if col not in merged.columns:
                continue
            ops.append(CrossSection.winsorize(col, by=by, limits=limits))
            ops.append(CrossSection.standardize(col, by=by).alias(f"{col}_z"))

        return merged.with_columns(ops)