"""AkShare data provider implementation."""

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import akshare as ak
import polars as pl

from quant_trade.config.logger import log
from quant_trade.provider.utils import (
    AdjustCN,
    DateLike,
    Period,
    Quarter,
    current_quarter_end,
    normalize_date_column,
    normalize_ts_code,
    quarter_end,
    to_date,
    to_ymd_str,
)


def _fetch_and_clean_whole(
    fetch_func: Callable,
    rename_map: dict,
    date_col: str = "publish_date",
    code_col: str = "ts_code",
    name_col: str = "name",
) -> pl.DataFrame:
    """Common pattern: fetch → pandas → polars → rename → select → normalize date"""
    try:
        raw = fetch_func()
        if raw.empty:
            log.warning(f"Empty result from {fetch_func.__name__}")
            return pl.DataFrame(
                schema={code_col: pl.Utf8, name_col: pl.Utf8, date_col: pl.Date}
            )

        df = pl.from_pandas(raw)
        df = df.rename(rename_map)
        df = df.select([code_col, name_col, date_col])
        df = normalize_date_column(df, date_col)
        df = df.with_columns(
            pl.col(code_col).cast(pl.Utf8),
            pl.col(name_col).cast(pl.Utf8),
        )
        return df
    except Exception as e:
        log.error(f"Failed to fetch whole list: {e}")
        return pl.DataFrame(
            schema={code_col: pl.Utf8, name_col: pl.Utf8, date_col: pl.Date}
        )


def stock_code_sh_whole() -> pl.DataFrame:
    return _fetch_and_clean_whole(
        lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "上市日期": "publish_date",
        },
    )


def stock_code_sz_whole() -> pl.DataFrame:
    return _fetch_and_clean_whole(
        lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
        {
            "A股代码": "ts_code",
            "A股简称": "name",
            "A股上市日期": "publish_date",
        },
    )


def stock_code_bj_whole() -> pl.DataFrame:
    return _fetch_and_clean_whole(
        ak.stock_info_bj_name_code,
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "上市日期": "publish_date",
        },
    )


def stock_code_delist_sh() -> pl.DataFrame:
    df = _fetch_and_clean_whole(
        lambda: ak.stock_info_sh_delist(symbol="全部"),
        {
            "公司代码": "ts_code",
            "公司简称": "name",
            "暂停上市日期": "off_date",
        },
        date_col="off_date",
    )
    return df.filter(pl.col("off_date").is_not_null())


def stock_code_delist_sz() -> pl.DataFrame:
    df = _fetch_and_clean_whole(
        lambda: ak.stock_info_sz_delist(symbol="终止上市公司"),
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "终止上市日期": "off_date",
        },
        date_col="off_date",
    )
    return df.filter(pl.col("off_date").is_not_null())


def stock_code_delist() -> pl.DataFrame:
    sh = stock_code_delist_sh()
    sz = stock_code_delist_sz()
    if sh.is_empty() and sz.is_empty():
        return pl.DataFrame(
            schema={"ts_code": pl.Utf8, "name": pl.Utf8, "off_date": pl.Date}
        )
    return pl.concat([sh, sz], how="vertical_relaxed").unique("ts_code", keep="last")


def stock_code_whole() -> pl.DataFrame:
    sh = stock_code_sh_whole()
    sz = stock_code_sz_whole()
    bj = stock_code_bj_whole()
    if all(d.is_empty() for d in [sh, sz, bj]):
        log.warning("No listed stocks fetched from any exchange")
        return pl.DataFrame(
            schema={"ts_code": pl.Utf8, "name": pl.Utf8, "publish_date": pl.Date}
        )
    return pl.concat([sh, sz, bj], how="vertical_relaxed").unique(
        "ts_code", keep="first"
    )


def stock_code_sus(date: DateLike) -> pl.DataFrame:
    ymd = to_ymd_str(date)
    try:
        raw = ak.stock_tfp_em(date=ymd)
        if raw.empty:
            return pl.DataFrame(schema={"ts_code": pl.Utf8, "name": pl.Utf8})

        df = pl.from_pandas(raw).select(
            pl.col("代码").alias("ts_code"),
            pl.col("名称").alias("name"),
        )
        df = df.with_columns(
            pl.col("ts_code").cast(pl.Utf8),
            pl.col("name").cast(pl.Utf8),
        )
        return df
    except Exception as e:
        log.error(f"Failed to fetch stock_tfp_em for {ymd}: {e}")
        return pl.DataFrame(schema={"ts_code": pl.Utf8, "name": pl.Utf8})


# ────────────────────────────────────────────────
#   AkShareMicro class
# ────────────────────────────────────────────────


class AkShareMicro:
    """AkShare implementation of the micro data interface."""

    def __init__(self):
        log.info("Initialized AkShareMicro.")

    def stock_whole(self) -> pl.DataFrame:
        """
        Fetch A-Share stock universe in general.
        """
        df = stock_code_whole()
        return df

    def stock_slice(
        self,
        trade_date: DateLike | None = None,
        min_listed_days: int = 365,
        remove_st: bool = True,
    ) -> pl.DataFrame:
        trade_date = to_date(trade_date) if trade_date else date.today()
        log.info(f"Building tradable universe for {trade_date}")

        universe = self.stock_whole()
        if universe.is_empty():
            return pl.DataFrame(schema={"ts_code": pl.Utf8, "name": pl.Utf8})

        delist = stock_code_delist()

        q = (
            universe.lazy()
            .filter(pl.col("publish_date") <= pl.lit(trade_date))
            .filter(
                pl.col("publish_date")
                <= pl.lit(trade_date - timedelta(days=min_listed_days))
            )
        )

        if remove_st:
            q = q.filter(~pl.col("name").str.contains(r"(?i)ST|\*ST"))

        if not delist.is_empty():
            delisted = (
                delist.filter(pl.col("off_date") <= pl.lit(trade_date))
                .select("ts_code")
                .lazy()  # Convert to LazyFrame
            )
            q = q.join(delisted, on="ts_code", how="anti")

        return q.sort("ts_code").select(["ts_code", "name"]).collect()

    def market_ohlcv(
        self,
        symbol: str,
        period: Period,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> pl.DataFrame:
        start_str = to_ymd_str(start_date) if start_date else "19910403"
        end_str = (
            to_ymd_str(end_date) if end_date else datetime.now().strftime("%Y%m%d")
        )

        log.info(
            f"Fetching {symbol} {period} OHLCV {start_str} → {end_str} (adj={adjust})"
        )

        try:
            df_raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_str,
                end_date=end_str,
                adjust=adjust or "",
                timeout=60,
            )
            if df_raw.empty:
                return pl.DataFrame()

            df = pl.from_pandas(df_raw).rename(
                {
                    "日期": "date",
                    "股票代码": "ts_code",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_chg",
                    "涨跌额": "change",
                    "换手率": "turnover_rate",
                }
            )

            df = normalize_date_column(df, "date")
            return df
        except Exception as e:
            log.error(f"market_ohlcv failed for {symbol}: {e}")
            return pl.DataFrame()

    def _fetch_quarterly_em(
        self,
        func: Callable,
        date_str: str,
        rename_map: dict,
        log_name: str,
    ) -> pl.DataFrame:
        log.info(f"Fetching {log_name} for {date_str}")
        try:
            raw = func(date=date_str)
            if raw.empty:
                return pl.DataFrame()
            df = pl.from_pandas(raw).rename(rename_map).drop("序号", strict=False)
            df = normalize_date_column(df, "announcement_date")
            return df
        except Exception as e:
            log.error(f"{log_name} fetch failed for {date_str}: {e}")
            return pl.DataFrame()

    def quarterly_income_statement(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        date_str = quarter_end(year, quarter) if year else current_quarter_end()
        rename_map = {
            "股票代码": "ts_code",
            "股票简称": "name",
            "公告日期": "announcement_date",
            "净利润": "net_profit",
            "营业利润": "operating_profit",
            "利润总额": "total_profit",
            "营业总收入": "total_revenue",
            "净利润同比": "net_profit_yoy",
            "营业总收入同比": "total_revenue_yoy",
            "营业总支出-营业支出": "operating_cost",
            "营业总支出-销售费用": "selling_cost",
            "营业总支出-管理费用": "admin_cost",
            "营业总支出-财务费用": "finance_cost",
            "营业总支出-营业总支出": "total_cost",
        }
        return self._fetch_quarterly_em(
            ak.stock_lrb_em, date_str, rename_map, "income statement"
        )

    def quarterly_balance_sheet(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_zcfz_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        date_str = quarter_end(year, quarter) if year else current_quarter_end()
        rename_map = {
            # identifiers
            "股票代码": "ts_code",
            "股票简称": "name",
            # dates
            "公告日期": "announcement_date",
            # assets (unit：yuan)
            "资产-货币资金": "cash",
            "资产-应收账款": "accounts_receivable",
            "资产-存货": "inventory",
            "资产-总资产": "total_assets",
            "资产-总资产同比": "total_assets_yoy",  # unit：%
            # liabilities (unit：yuan)
            "负债-应付账款": "accounts_payable",
            "负债-预收账款": "advance_receipts",
            "负债-总负债": "total_debts",
            "负债-总负债同比": "total_debts_yoy",  # unit：%
            # ratios / equity
            "资产负债率": "debt_to_assets",  # unit：%
            "股东权益合计": "total_equity",  # unit：yuan
        }
        return self._fetch_quarterly_em(
            ak.stock_zcfz_em, date_str, rename_map, "balance sheet"
        )

    def quarterly_cashflow_statement(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_xjll_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        date_str = quarter_end(year, quarter) if year else current_quarter_end()
        rename_map = {
            # identifiers
            "股票代码": "ts_code",
            "股票简称": "name",
            # date
            "公告日期": "announcement_date",
            # total net cashflow
            "净现金流-净现金流": "net_cashflow",  # (unit: yuan)
            "净现金流-同比增长": "net_cashflow_yoy",
            # operating cashflow
            "经营性现金流-现金流量净额": "cfo",  # (unit: yuan)
            "经营性现金流-净现金流占比": "cfo_share",
            # investing cashflow
            "投资性现金流-现金流量净额": "cfi",  # (unit: yuan)
            "投资性现金流-净现金流占比": "cfi_share",  # %
            # financing cashflow
            "融资性现金流-现金流量净额": "cff",  # (unit: yuan)
            "融资性现金流-净现金流占比": "cff_share",  # %
        }

        return self._fetch_quarterly_em(
            ak.stock_xjll_em, date_str, rename_map, "cashflow statement"
        )

    def quarterly_fundamentals(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        inc = self.quarterly_income_statement(year, quarter)
        bal = self.quarterly_balance_sheet(year, quarter)
        cf = self.quarterly_cashflow_statement(year, quarter)

        if all(df.is_empty() for df in (inc, bal, cf)):
            return pl.DataFrame()

        keys = ["ts_code", "name", "announcement_date"]
        df = inc.join(bal, on=keys, how="full", coalesce=True).join(
            cf, on=keys, how="full", coalesce=True
        )
        return df


class AkShareMacro:
    """AkShare implementation of the macro data interface."""

    def __init__(self):
        log.info("Initialized AkShareMacro.")

    def _fetch_and_filter_date_range(
        self,
        df: pl.DataFrame,
        start_date: date,
        end_date: date,
        date_col: str = "date",
    ) -> pl.DataFrame:
        """Common final step: normalize date + filter range"""
        if df.is_empty():
            return pl.DataFrame(schema={date_col: pl.Date})

        df = normalize_date_column(df, date_col)
        return df.filter(
            (pl.col(date_col) >= pl.lit(start_date))
            & (pl.col(date_col) <= pl.lit(end_date))
        ).sort(date_col)

    def northbound_flow(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        Fetch combined northbound (沪港通 + 深港通) net buy flow.
        Columns: date, net_buy, fund_inflow, cum_net_buy
        """
        start_dt = to_date(start_date) if start_date else date(2014, 11, 17)
        end_dt = to_date(end_date) if end_date else date.today()

        log.info(f"Fetching northbound flow {start_dt} → {end_dt}")

        try:
            # SH
            sh_raw = ak.stock_hsgt_hist_em(symbol="港股通沪")
            sh = (
                pl.from_pandas(sh_raw)
                .rename(
                    {
                        "日期": "date",
                        "当日成交净买额": "net_buy",
                        "当日资金流入": "fund_inflow",
                        "历史累计净买额": "cum_net_buy",
                    }
                )
                .select(["date", "net_buy", "fund_inflow", "cum_net_buy"])
            )

            # SZ
            sz_raw = ak.stock_hsgt_hist_em(symbol="港股通深")
            sz = (
                pl.from_pandas(sz_raw)
                .rename(
                    {
                        "日期": "date",
                        "当日成交净买额": "net_buy",
                        "当日资金流入": "fund_inflow",
                        "历史累计净买额": "cum_net_buy",
                    }
                )
                .select(["date", "net_buy", "fund_inflow", "cum_net_buy"])
            )

            if sh.is_empty() and sz.is_empty():
                log.warning("No northbound data fetched from either exchange")
                return pl.DataFrame()

            combined = (
                pl.concat([sh, sz], how="vertical").group_by("date").sum().sort("date")
            )

            return self._fetch_and_filter_date_range(combined, start_dt, end_dt)

        except Exception as e:
            log.error(f"northbound_flow failed: {e}")
            return pl.DataFrame()

    def market_margin_short(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        Combined SH+SZ margin & short-selling balance (daily).
        Main column of interest: total_margin_balance
        """
        start_dt = to_date(start_date) if start_date else date(2010, 3, 31)
        end_dt = to_date(end_date) if end_date else date.today()

        log.info(f"Fetching market margin/short data {start_dt} → {end_dt}")

        try:
            sz_raw = ak.macro_china_market_margin_sz()
            sz = pl.from_pandas(sz_raw)

            sh_raw = ak.macro_china_market_margin_sh()
            sh = pl.from_pandas(sh_raw)

            def _process(df: pl.DataFrame) -> pl.DataFrame:
                return (
                    df.rename(
                        {
                            "日期": "date",
                            "融资买入额": "margin_buy_amount",
                            "融资余额": "margin_balance",
                            "融券卖出量": "short_sell_volume",
                            "融券余额": "short_balance",
                            "融资融券余额": "total_margin_balance",
                        }
                    )
                    .drop("融券余量", strict=False)
                    .with_columns(pl.exclude("date").cast(pl.Float64))
                )

            sz_proc = _process(sz)
            sh_proc = _process(sh)

            if sz_proc.is_empty() and sh_proc.is_empty():
                return pl.DataFrame()

            combined = (
                pl.concat([sz_proc, sh_proc], how="vertical")
                .group_by("date")
                .sum()
                .sort("date")
            )

            return self._fetch_and_filter_date_range(combined, start_dt, end_dt)

        except Exception as e:
            log.error(f"market_margin_short failed: {e}")
            return pl.DataFrame()

    def shibor(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        Full SHIBOR rates history (all tenors).
        """
        start_dt = to_date(start_date) if start_date else date(2017, 3, 17)
        end_dt = to_date(end_date) if end_date else date.today()

        log.info(f"Fetching SHIBOR rates {start_dt} → {end_dt}")

        try:
            raw = ak.macro_china_shibor_all()
            if raw.empty:
                return pl.DataFrame()

            df = pl.from_pandas(raw).rename(
                {
                    "日期": "date",
                    "O/N-定价": "ON_rate",
                    "O/N-涨跌幅": "ON_change",
                    "1W-定价": "1W_rate",
                    "1W-涨跌幅": "1W_change",
                    "2W-定价": "2W_rate",
                    "2W-涨跌幅": "2W_change",
                    "1M-定价": "1M_rate",
                    "1M-涨跌幅": "1M_change",
                    "3M-定价": "3M_rate",
                    "3M-涨跌幅": "3M_change",
                    "6M-定价": "6M_rate",
                    "6M-涨跌幅": "6M_change",
                    "9M-定价": "9M_rate",
                    "9M-涨跌幅": "9M_change",
                    "1Y-定价": "1Y_rate",
                    "1Y-涨跌幅": "1Y_change",
                }
            )

            return self._fetch_and_filter_date_range(df, start_dt, end_dt)

        except Exception as e:
            log.error(f"shibor fetch failed: {e}")
            return pl.DataFrame()

    def csi1000_daily_ohlcv(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        CSI 1000 Index daily OHLCV (akshare uses em interface).
        """
        start_dt = to_date(start_date) if start_date else date(2005, 1, 5)
        end_dt = to_date(end_date) if end_date else date.today()

        log.info(f"Fetching CSI1000 daily OHLCV {start_dt} → {end_dt}")

        try:
            raw = ak.stock_zh_index_daily_em()
            if raw.empty:
                return pl.DataFrame()

            df = pl.from_pandas(raw).drop("amount", strict=False)
            df = normalize_date_column(df, "date")  # usually already has 'date'

            return self._fetch_and_filter_date_range(df, start_dt, end_dt)

        except Exception as e:
            log.error(f"csi1000_daily_ohlcv failed: {e}")
            return pl.DataFrame()

    def csi1000qvix_daily_ohlc(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        CSI 1000 Volatility Index (QVIX) daily data.
        """
        start_dt = to_date(start_date) if start_date else date(2005, 2, 9)
        end_dt = to_date(end_date) if end_date else date.today()

        log.info(f"Fetching CSI1000 QVIX {start_dt} → {end_dt}")

        try:
            raw = ak.index_option_1000index_qvix()
            if raw.empty:
                return pl.DataFrame()

            df = pl.from_pandas(raw).drop_nulls()
            df = normalize_date_column(df, "date")

            return self._fetch_and_filter_date_range(df, start_dt, end_dt)

        except Exception as e:
            log.error(f"csi1000qvix_daily_ohlc failed: {e}")
            return pl.DataFrame()


class SWIndustryCls:
    """
    Shenwan (申万) industry classification manager.
    Supports L1/L2/L3 levels, point-in-time views, and caching.

    Cache file: ~/.cache/quant/sw_industry_cls.parquet (zstd compressed)
    """

    CACHE_DIR = Path("~/.cache/quant").expanduser()
    CACHE_FILE = CACHE_DIR / "sw_industry_cls.parquet"

    def __init__(self, use_cache: bool = True, rebuild_cache: bool = False):
        self.use_cache = use_cache
        self.rebuild_cache = rebuild_cache
        self._full_cls: pl.DataFrame | None = None
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load_or_build(self, force_rebuild: bool = False) -> pl.DataFrame:
        """
        Load from cache if available and valid, otherwise build and save.
        """
        if (
            not force_rebuild
            and not self.rebuild_cache
            and self.use_cache
            and self.CACHE_FILE.exists()
        ):
            try:
                df = pl.read_parquet(self.CACHE_FILE)
                if "ts_code" in df and "sw_l1_code" in df and "join_date" in df:
                    log.info(
                        f"Loaded Shenwan industry classification from cache ({len(df)} rows)"
                    )
                    self._full_cls = df
                    return df
            except Exception as e:
                log.warning(f"Cache load failed: {e}. Will rebuild.")

        log.info(
            "Building Shenwan industry classification (this may take several minutes)..."
        )
        df = self._build_full_classification()
        if not df.is_empty():
            df.write_parquet(self.CACHE_FILE, compression="zstd")
            log.info(
                f"Saved industry classification cache: {self.CACHE_FILE} ({len(df)} rows)"
            )
        self._full_cls = df
        return df

    def _build_full_classification(self) -> pl.DataFrame:
        """Core slow build logic — loop over all L3 industries"""
        try:
            info = self.sw_l3_info()
            if info.is_empty():
                raise ValueError("No L3 industry metadata fetched")

            all_cons = []
            total = len(info)

            for idx, row in enumerate(info.iter_rows(named=True), 1):
                code = row["sw_l3_code"]
                name = row.get("sw_l3_name", code)

                if idx % 30 == 0 or idx == total:
                    log.info(f"[{idx}/{total}] Processing {code} ({name})")

                try:
                    cons = self.sw_l3_constituents(code)
                    if cons.is_empty():
                        continue

                    l1, l2, _ = self.derive_sw_levels(code)

                    cons = cons.with_columns(
                        pl.lit(l1).alias("sw_l1_code"),
                        pl.lit(l2).alias("sw_l2_code"),
                        pl.lit(code).alias("sw_l3_code"),
                        pl.lit(name).alias("sw_l3_name"),
                    )

                    all_cons.append(cons)
                except Exception as e:
                    log.warning(f"Failed to process {code} ({name}): {e}")
                    continue

            if not all_cons:
                raise ValueError("No valid constituents fetched from any industry")

            combined = (
                pl.concat(all_cons, how="vertical_relaxed")
                .with_columns(
                    normalize_ts_code("ts_code_suffix", add_suffix=False).alias(
                        "ts_code"
                    )
                )
                .drop("ts_code_suffix")
                .unique(
                    subset=["ts_code"], keep="last"
                )  # most recent classification wins
                .sort(["ts_code", "join_date"])
            )

            combined = normalize_date_column(combined, "join_date")

            return combined

        except Exception as e:
            log.error(f"Failed to build full industry classification: {e}")
            return pl.DataFrame()

    def sw_l3_info(self) -> pl.DataFrame:
        """Fetch Shenwan L3 industry metadata"""
        log.info("Fetching Shenwan L3 industry metadata...")
        try:
            raw = ak.sw_index_third_info()
            if raw.empty:
                log.warning("Empty L3 info response")
                return pl.DataFrame()

            df = pl.from_pandas(raw).rename(
                {
                    "行业代码": "sw_l3_code",
                    "行业名称": "sw_l3_name",
                    "上级行业": "sw_l2_name",
                    "成份个数": "constituent_count",
                    "静态市盈率": "pe_static",
                    "TTM(滚动)市盈率": "pe_ttm",
                    "市净率": "pb",
                    "静态股息率": "dividend_yield",
                }
            )
            log.info(f"Fetched {len(df)} L3 industries")
            return df
        except Exception as e:
            log.error(f"L3 info fetch failed: {e}")
            return pl.DataFrame()

    def sw_l3_constituents(self, sw_l3_code: str) -> pl.DataFrame:
        """Fetch constituents of one L3 industry"""
        try:
            raw = ak.sw_index_third_cons(symbol=sw_l3_code)
            if raw.empty:
                log.debug(f"No constituents for {sw_l3_code}")
                return pl.DataFrame()

            df = (
                pl.from_pandas(raw)
                .rename(
                    {
                        "股票代码": "ts_code_suffix",
                        "股票简称": "name",
                        "纳入时间": "join_date",
                        "申万1级": "sw_l1_name",
                        "申万2级": "sw_l2_name",
                        "申万3级": "sw_l3_name",
                    }
                )
                .select(
                    [
                        "ts_code_suffix",
                        "name",
                        "join_date",
                        "sw_l1_name",
                        "sw_l2_name",
                        "sw_l3_name",
                    ]
                )
            )

            df = df.with_columns(pl.lit(sw_l3_code).alias("sw_l3_code"))
            df = normalize_date_column(df, "join_date")

            return df
        except Exception as e:
            log.warning(f"Constituents fetch failed for {sw_l3_code}: {e}")
            return pl.DataFrame()

    @staticmethod
    def derive_sw_levels(l3_code: str) -> tuple[str, str, str]:
        """Derive L1 and L2 codes from L3 code (Shenwan format)"""
        base = l3_code.replace(".SI", "").strip()
        if len(base) != 6 or not base.isdigit():
            raise ValueError(f"Invalid Shenwan L3 code format: {l3_code}")

        l1 = f"{base[:2]}0000.SI"
        l2 = f"{base[:4]}00.SI"
        return l1, l2, l3_code

    def get_full_classification(self) -> pl.DataFrame:
        """Get or build the master table (ts_code → industry codes + join_date)"""
        if self._full_cls is None:
            self._full_cls = self.load_or_build()
        return self._full_cls

    def stock_l1_industry_cls(
        self,
        as_of_date: DateLike | None = None,
        include_names: bool = False,
    ) -> pl.DataFrame:
        """
        Simplified L1 mapping: ts_code → sw_l1_code (optionally with names)
        Can be point-in-time if as_of_date is given.
        """
        df = self.get_full_classification()
        if df.is_empty():
            return pl.DataFrame(schema={"ts_code": pl.Utf8, "sw_l1_code": pl.Utf8})

        cols = ["ts_code", "sw_l1_code"]
        if include_names:
            cols += ["sw_l1_name"]

        if as_of_date:
            as_of = to_date(as_of_date)
            df = self.get_classification_as_of(as_of)

        return df.select(cols).unique("ts_code", keep="last")

    def get_classification_as_of(self, as_of: date) -> pl.DataFrame:
        """
        Point-in-time industry classification valid on or before `as_of` date.
        Returns the latest known classification per stock where join_date <= as_of.
        """
        df = self.get_full_classification()
        if df.is_empty():
            return pl.DataFrame()

        # Keep only rows where the stock was already classified by as_of
        return (
            df.filter(pl.col("join_date") <= pl.lit(as_of))
            .group_by("ts_code")
            .agg(pl.all().sort_by("join_date").last())
            .sort("ts_code")
        )

    def suffix_ts_code(self, code: str) -> str:
        """Add .SH/.SZ/.BJ suffix if missing (your original logic preserved)"""
        if "." in code and len(code.split(".")[0]) == 6:
            return code.upper()

        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"Invalid code format: {code}")

        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"

        log.warning(f"Unknown exchange for code: {code}")
        return code  # fallback - no suffix

    def strip_ts_code(self, code: str) -> str:
        """Remove .EX suffix, keep only 6-digit code (your original preserved)"""
        if len(code) == 6 and code.isdigit():
            return code

        parts = code.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid ts_code format: {code}")

        prefix, suffix = parts[0], parts[1].upper()
        if len(prefix) != 6 or not prefix.isdigit():
            raise ValueError(f"Invalid prefix: {code}")

        if suffix not in {"SH", "SZ", "BJ"}:
            log.warning(f"Unrecognized suffix in {code}, returning prefix anyway")

        return prefix
