from collections.abc import Callable, Sequence
from datetime import date
from functools import partial
from typing import Any, Protocol

import baostock as bs
import polars as pl

import quant_trade.provider.concurrent as concur
from quant_trade.config.logger import log
from quant_trade.transform import (
    AdjustCN,
    DateLike,
    Period,
    Quarter,
    normalize_date_column,
    normalize_ts_code,
    normalize_ts_code_str,
    to_ymd_str,
)


def normalize_bao_ts_code(code: str) -> str:
    return normalize_ts_code_str(code, add_exchange=True, position="prefix", sep=".")


def to_bao_ymd_str(d: DateLike) -> str:
    return to_ymd_str(d, "-")


def to_bao_period(period: Period) -> str:
    match period:
        case "daily":
            return "d"
        case "weekly":
            return "w"
        case "monthly":
            return "m"


def to_bao_adjust(adjust: AdjustCN | None) -> str:
    match adjust:
        case "hfq":
            return "1"
        case "qfq":
            return "2"
        case _:
            return "3"


class BaoFetched(Protocol):
    error_code: str
    error_msg: str

    def next(self) -> bool: ...
    def get_row_data(self) -> list[Any]: ...


QueryFunc = Callable[..., BaoFetched | None]


class BaoSession:
    _active: bool = False

    def __enter__(self):
        if BaoSession._active:
            raise RuntimeError("BaoStockSession already active")

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

        BaoSession._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        bs.logout()
        BaoSession._active = False
        return False

    @staticmethod
    def active() -> bool:
        return BaoSession._active


class BaoFetcher:
    def __init__(self, query_func: QueryFunc):
        self._query_func = query_func

    def fetch(self, **kwargs) -> pl.DataFrame:
        if not BaoSession._active:
            raise RuntimeError(
                "BaoStockFetcher.fetch() called outside of BaoStockSession"
            )

        rs = self._query_func(**kwargs)

        if rs is None:
            return pl.DataFrame()
        if rs.error_code != "0":
            raise RuntimeError(f"Query failed: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        fields = getattr(rs, "fields", None)
        if fields is None:
            log.warning("The fields is empty")
        return pl.DataFrame(rows, schema=fields)


def _fetch_quarterly(
    code: str,
    fetch: QueryFunc,
    rename: dict[str, str],
    year: int | None = None,
    quarter: Quarter | None = None,
) -> pl.DataFrame:
    code = normalize_bao_ts_code(code)
    ident_rename = {
        "code": "ts_code",
        "pubDate": "notice_date",
        "statDate": "date",
    }
    df = (
        BaoFetcher(fetch)
        .fetch(code=code, year=year, quarter=quarter)
        .rename(ident_rename)
        .rename(rename)
    )
    df = normalize_date_column(df, date_col="date")
    df = normalize_date_column(df, date_col="notice_date")
    df = df.with_columns(normalize_ts_code("ts_code", exchange=None))
    return df


def quarterly_profit(
    code: str, year: int | None = None, quarter: Quarter | None = None
) -> pl.DataFrame:
    rename = {
        "roeAvg": "roe",
        "npMargin": "net_margin",
        "gpMargin": "gross_margin",
        "netProfit": "net_profit",
        "epsTTM": "eps_ttm",
        "MBRevenue": "total_revenue",
        "totalShare": "total_shares",
        "liqaShare": "float_shares",
    }
    return _fetch_quarterly(
        code=code, fetch=bs.query_profit_data, rename=rename, year=year, quarter=quarter
    )


def quarterly_operation(
    code: str, year: int | None = None, quarter: Quarter | None = None
) -> pl.DataFrame:
    rename = {
        "NRTurnRatio": "receivable_turnover",  # 应收账款周转率
        "NRTurnDays": "receivable_turnover_days",  # 应收账款周转天数
        "INVTurnRatio": "inventory_turnover",  # 存货周转率
        "INVTurnDays": "inventory_turnover_days",  # 存货周转天数
        "CATurnRatio": "current_assets_turnover",  # 流动资产周转率
        "AssetTurnRatio": "total_asset_turnover",  # 总资产周转率
    }
    return _fetch_quarterly(
        code=code,
        fetch=bs.query_operation_data,
        rename=rename,
        year=year,
        quarter=quarter,
    )


def quarterly_growth(
    code: str, year: int | None = None, quarter: Quarter | None = None
) -> pl.DataFrame:
    rename = {
        "YOYEquity": "equity_yoy",  # 净资产同比增长率
        "YOYAsset": "total_asset_yoy",  # 总资产同比增长率
        "YOYNI": "net_profit_yoy",  # 净利润同比增长率
        "YOYEPSBasic": "eps_basic_yoy",  # 基本每股收益同比增长率
        "YOYPNI": "net_profit_parent_yoy",  # 归属母公司股东净利润同比增长率
    }
    return _fetch_quarterly(
        code=code, fetch=bs.query_growth_data, rename=rename, year=year, quarter=quarter
    )


def quarterly_balance(
    code: str, year: int | None = None, quarter: Quarter | None = None
) -> pl.DataFrame:
    rename = {
        "currentRatio": "current_ratio",
        "quickRatio": "quick_ratio",
        "cashRatio": "cash_ratio",
        "YOYLiability": "total_debt_yoy",
        "liabilityToAsset": "debts_to_assets",
        "assetToEquity": "assets_to_equity",
    }
    return _fetch_quarterly(
        code=code,
        fetch=bs.query_balance_data,
        rename=rename,
        year=year,
        quarter=quarter,
    )


def quarterly_cashflow(
    code: str, year: int | None = None, quarter: Quarter | None = None
) -> pl.DataFrame:
    rename = {
        "CAToAsset": "current_assets_to_total_asset",  # 流动资产占总资产比例
        "NCAToAsset": "non_current_assets_to_total_asset",  # 非流动资产占总资产比例
        "tangibleAssetToAsset": "tangible_assets_to_total_asset",  # 有形资产占总资产比例
        "ebitToInterest": "ebit_to_interest",  # 已获利息倍数
        "CFOToOR": "cfo_to_revenue",  # 经营活动现金流占营业收入比例
        "CFOToNP": "cfo_to_net_profit",  # 经营性现金净流量占净利润比例
        "CFOToGr": "cfo_to_total_revenue",  # 经营性现金净流量占营业总收入比例
    }
    return _fetch_quarterly(
        code=code,
        fetch=bs.query_cash_flow_data,
        rename=rename,
        year=year,
        quarter=quarter,
    )


def quarterly_dupont(
    code: str, year: int | None = None, quarter: Quarter | None = None
) -> pl.DataFrame:
    rename = {
        # 杜邦分析指标（DuPont Analysis Metrics）
        "dupontROE": "roe",  # 净资产收益率（杜邦）
        "dupontAssetStoEquity": "assets_to_equity",  # 权益乘数
        "dupontAssetTurn": "total_asset_turnover",  # 总资产周转率（杜邦）
        "dupontPnitoni": "parent_profit_ratio",  # 归属母公司股东净利润占比
        "dupontNitogr": "net_margin",  # 净利率
        "dupontTaxBurden": "tax_burden",  # 税收负担率
        "dupontIntburden": "interest_burden",  # 利息负担率
        "dupontEbittogr": "ebit_margin",  # 息税前利润率
    }
    return _fetch_quarterly(
        code=code, fetch=bs.query_dupont_data, rename=rename, year=year, quarter=quarter
    )


def market_ohlcv(
    code: str,
    period: Period,
    start: DateLike | None = None,
    end: DateLike | None = None,
    adjust: AdjustCN | None = "hfq",
) -> pl.DataFrame:
    code = normalize_bao_ts_code(code)
    log.info(f"Fetching {code} {period} OHLCV {start} → {end} (adj={adjust})")

    fields = "date,code,open,high,low,close,volume,turn,tradestatus,isST"
    start = to_bao_ymd_str(start) if start else None
    end = to_bao_ymd_str(end) if end else None
    period_str = to_bao_period(period)
    adjust_str = to_bao_adjust(adjust)
    rename = {
        "code": "ts_code",
        "turn": "turnover",
        "tradestatus": "trade_status",
        "isST": "is_st",
    }
    df = (
        BaoFetcher(bs.query_history_k_data_plus)
        .fetch(
            code=code,
            fields=fields,
            start_date=start,
            end_date=end,
            frequency=period_str,
            adjustflag=adjust_str,
        )
        .rename(rename)
    )
    has_st = df.select((pl.col("is_st") == "1").any()).item()
    if has_st:
        log.info(f"{code} filtered out (ST or suspended)")
        return pl.DataFrame()

    ohlcv = ["open", "high", "low", "close", "volume", "turnover"]
    df = normalize_date_column(df, date_col="date").drop(["trade_status", "is_st"])
    df = df.with_columns(normalize_ts_code("ts_code", exchange=None)).with_columns(
        pl.col(ohlcv).replace("", None).cast(pl.Float64)
    )
    return df


def csi500_cons(date: date | None = None) -> pl.DataFrame:
    log.info(f"Fetching csi500 cons at {date}")
    date_str = to_bao_ymd_str(date) if date else None
    df = (
        BaoFetcher(bs.query_zz500_stocks)
        .fetch(date=date_str)
        .rename(
            {
                "updateDate": "date",
                "code": "ts_code",
                "code_name": "name",
            }
        )
    )
    df = normalize_date_column(df, date_col="date")
    df = df.with_columns(normalize_ts_code("ts_code", exchange=None))
    df = df.filter(~pl.col("name").str.contains(r"(?i)ST|\*ST"))
    return df


class BaoUniverse:
    """financial statement macro-APIs (BaoStock façade)."""

    @staticmethod
    def csi500_cons(date: date | None = None) -> pl.DataFrame:
        with BaoSession():
            df = csi500_cons(date)
        return df


class BaoMicro:
    """financial statement micro-APIs (BaoStock façade)."""

    @staticmethod
    def market_ohlcv(
        symbol: str,
        period: Period,
        start_date: date | None = None,
        end_date: date | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> pl.DataFrame:
        with BaoSession():
            df = market_ohlcv(symbol, period, start_date, end_date, adjust)
        return df

    @staticmethod
    def batch_market_ohlcv(
        symbols: Sequence[str],
        period: Period,
        start_date: date | None = None,
        end_date: date | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> list[pl.DataFrame]:
        """Parallel fetch OHLCV for many symbols.
        Returns list of pl.DataFrame **in the same order** as input `codes`.
        Empty DataFrame = no data / filtered / failed.
        """
        # tenacity @retry can't be used due to pickable of process
        worker = partial(
            concur.Try()(BaoMicro.market_ohlcv),
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        config = concur.BatchConfig.process()
        return concur.batch_fetch(
            config=config,
            worker=worker,
            items=symbols,
        )
