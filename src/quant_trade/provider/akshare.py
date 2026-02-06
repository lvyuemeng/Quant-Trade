"""AkShare data provider implementation."""

from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path

import akshare as ak
import polars as pl

import quant_trade.provider.concurrent as concur
from quant_trade.client.eastmoney import EastMoney
from quant_trade.config.logger import log
from quant_trade.transform import (
    AdjustCN,
    DateLike,
    Period,
    Quarter,
    normalize_date_column,
    normalize_ts_code,
    normalize_ts_code_str,
    to_date,
    to_ymd_str,
)


def quarter_to_report_date(year: int, quarter: Quarter, as_ymd: bool = True) -> str:
    """Explicit year + quarter → YYYYMMDD or YYYY-MM-DD"""
    ends = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    s = f"{year}{ends[quarter]}"
    if not as_ymd:
        return f"{year}-{ends[quarter][:2]}-{ends[quarter][2:]}"
    return s


def cur_quarter_end() -> tuple[int, Quarter]:
    """Most recent completed quarter end (explicit — no None handling)"""
    today = datetime.now().date()
    year = today.year
    month = today.month

    if month <= 3:
        return (year - 1, 4)
    elif month <= 6:
        return (year, 1)
    elif month <= 9:
        return (year, 2)
    else:
        return (year, 3)


def quarter_range(
    start_year: int, start_quarter: Quarter, end_year: int, end_quarter: Quarter
) -> list[tuple[int, Quarter]]:
    """Generate list of (year, quarter) tuples covering the range [start, end]."""
    result = []
    current_year = start_year
    current_quarter = start_quarter

    while (current_year, current_quarter) <= (end_year, end_quarter):
        result.append((current_year, current_quarter))
        if current_quarter == 4:
            current_year += 1
            current_quarter = 1
        else:
            current_quarter += 1

    return result


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


def _fetch_ohlcv(
    fetcher: Callable,
    symbol: str,
    period: Period,
    start_str: str,
    end_str: str,
    adjust: str,
    column_map: dict,
    daily_only: bool = False,
    prefix: bool = False,
) -> pl.DataFrame | None:
    """Unified OHLCV fetcher for multiple sources."""
    try:
        # Check if daily-only data requested
        if daily_only and period != "daily":
            log.warning(
                f"{fetcher.__name__} only supports daily data, requested {period}"
            )
            return None

        if prefix:
            symbol = normalize_ts_code_str(
                symbol, add_exchange=True, position="prefix", sep="", case="lower"
            )

        df_raw = fetcher(symbol, start_str, end_str, adjust or "")
        if df_raw.empty:
            return None
        df = pl.from_pandas(df_raw).rename(column_map)

        required_cols = [
            "date",
            "ts_code",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "change",
            "turnover_rate",
        ]
        for col in required_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))

        # Add ts_code if missing
        if "ts_code" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("ts_code"))

        df = df.select(required_cols)
        return normalize_date_column(df, "date")

    except Exception as e:
        log.warning(f"{fetcher.__name__} fetch failed: {e}")
        return None


def fetch_ohlcv_eastmoney(
    symbol: str, period: Period, start_str: str, end_str: str, adjust: str = ""
) -> pl.DataFrame | None:
    EASTMONEY_COLS = {
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
    return _fetch_ohlcv(
        fetcher=lambda s, start, end, adj: ak.stock_zh_a_hist(
            symbol=s,
            period=period,
            start_date=start,
            end_date=end,
            adjust=adj,
            timeout=60,
        ),
        symbol=symbol,
        period=period,
        start_str=start_str,
        end_str=end_str,
        adjust=adjust,
        column_map=EASTMONEY_COLS,
    )


def fetch_ohlcv_sina(
    symbol: str, period: Period, start_str: str, end_str: str, adjust: str = ""
) -> pl.DataFrame | None:
    SINA_COLS = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
        "turnover": "turnover_rate",
    }
    return _fetch_ohlcv(
        fetcher=lambda s, start, end, adj: ak.stock_zh_a_daily(
            symbol=s, start_date=start, end_date=end, adjust=adj
        ),
        symbol=symbol,
        period=period,
        start_str=start_str,
        end_str=end_str,
        adjust=adjust,
        column_map=SINA_COLS,
        daily_only=True,
        prefix=True,
    )


def fetch_ohlcv_tencent(
    symbol: str, period: Period, start_str: str, end_str: str, adjust: str = ""
) -> pl.DataFrame | None:
    TENCENT_COLS = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "amount": "volume",  # special handling
    }
    df = _fetch_ohlcv(
        fetcher=lambda s, start, end, adj: ak.stock_zh_a_hist_tx(
            symbol=s, start_date=start, end_date=end, adjust=adj
        ),
        symbol=symbol,
        period=period,
        start_str=start_str,
        end_str=end_str,
        adjust=adjust,
        column_map=TENCENT_COLS,
        daily_only=True,
        prefix=True,
    )
    if df is not None:
        df.with_columns((pl.col("volume") * 100).alias("volume"))
        return df
    else:
        return None


@cache
def stock_code_sh_whole() -> pl.DataFrame:
    return _fetch_and_clean_whole(
        lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "上市日期": "publish_date",
        },
    )


@cache
def stock_code_sz_whole() -> pl.DataFrame:
    return _fetch_and_clean_whole(
        lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
        {
            "A股代码": "ts_code",
            "A股简称": "name",
            "A股上市日期": "publish_date",
        },
    )


@cache
def stock_code_bj_whole() -> pl.DataFrame:
    return _fetch_and_clean_whole(
        ak.stock_info_bj_name_code,
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "上市日期": "publish_date",
        },
    )


@cache
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


@cache
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


@cache
def stock_code_delist() -> pl.DataFrame:
    sh = stock_code_delist_sh()
    sz = stock_code_delist_sz()
    if sh.is_empty() and sz.is_empty():
        return pl.DataFrame(
            schema={"ts_code": pl.Utf8, "name": pl.Utf8, "off_date": pl.Date}
        )
    return pl.concat([sh, sz], how="vertical_relaxed").unique("ts_code", keep="last")


@cache
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


@cache
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


class SWIndustryCls:
    """
    Shenwan (申万) industry classification manager.
    Supports L1/L2/L3 levels, point-in-time views, and caching.

    Cache file: ~/.cache/quant/sw_industry_cls.parquet (zstd compressed)
    """

    CACHE_DIR = Path("~/.cache/quant").expanduser()
    CACHE_FILE = CACHE_DIR / "sw_industry_cls.parquet"
    CACHE_MAP_FILE = CACHE_DIR / "sw_level_maps.parquet"

    def __init__(self, use_cache: bool = True, rebuild_cache: bool = False):
        self.use_cache = use_cache
        self.rebuild_cache = rebuild_cache
        self._full_cls: pl.DataFrame | None = None
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load_or_build(self, fresh: bool = False) -> pl.DataFrame:
        """
        Load from cache if available and valid, otherwise build and save.
        """
        if (
            not fresh
            and not self.rebuild_cache
            and self.use_cache
            and self.CACHE_FILE.exists()
        ):
            try:
                df = pl.read_parquet(self.CACHE_FILE)
                required = {"ts_code", "sw_l1_code", "join_date"}
                if required.issubset(set(df.columns)):
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

    def _load_or_build_level_maps(self, fresh: bool = False) -> pl.DataFrame:
        """
        Build or load a flat L3 → L2 → L1 mapping table.

        Returned schema (one row per L3 industry):
            sw_l3_code | sw_l2_code | sw_l2_name | sw_l1_code | sw_l1_name

        Build logic:
            L1 info  →  {l1_name → l1_code}
            L2 info  →  {l2_name → l2_code},  {l2_code → l1_code} (via l2.sw_l1_name)
            L3 info  →  {l3_code → l2_code}   (via l3.sw_l2_name)
            Final table: join the three maps on l3_code
        """
        if not fresh and self.use_cache and self.CACHE_MAP_FILE.exists():
            try:
                df = pl.read_parquet(self.CACHE_MAP_FILE)
                required = {"sw_l3_code", "sw_l2_code", "sw_l1_code"}
                if required.issubset(set(df.columns)):
                    log.info(f"Loaded level maps from cache ({df.height} rows)")
                    return df
            except Exception as e:
                log.warning(f"Level maps cache load failed: {e}. Will rebuild.")

        log.info("Building Shenwan level maps...")

        l1 = self.sw_l1_info()
        l2 = self.sw_l2_info()
        l3 = self.sw_l3_info()

        if l1.is_empty() or l2.is_empty() or l3.is_empty():
            raise ValueError("One or more level info fetches returned empty data")

        l1_name_to_code: dict[str, str] = dict(
            zip(l1["sw_l1_name"].to_list(), l1["sw_l1_code"].to_list())
        )

        l2_enriched = l2.with_columns(
            pl.col("sw_l1_name").replace(l1_name_to_code).alias("sw_l1_code")
        )

        missing_l1 = l2_enriched.filter(pl.col("sw_l1_code").is_null())
        if missing_l1.height > 0:
            log.warning(
                f"Could not resolve L1 code for {missing_l1.height} L2 industries: "
                f"{missing_l1['sw_l2_name'].to_list()}"
            )

        l2_name_to_code: dict[str, str] = dict(
            zip(
                l2_enriched["sw_l2_name"].to_list(), l2_enriched["sw_l2_code"].to_list()
            )
        )

        l3_enriched = l3.with_columns(
            pl.col("sw_l2_name").replace(l2_name_to_code).alias("sw_l2_code")
        )

        missing_l2 = l3_enriched.filter(pl.col("sw_l2_code").is_null())
        if missing_l2.height > 0:
            log.warning(
                f"Could not resolve L2 code for {missing_l2.height} L3 industries: "
                f"{missing_l2['sw_l3_name'].to_list()}"
            )

        l2_code_to_l1_code: dict[str, str] = dict(
            zip(
                l2_enriched["sw_l2_code"].to_list(),
                l2_enriched["sw_l1_code"].to_list(),
            )
        )
        l2_code_to_l2_name: dict[str, str] = dict(
            zip(
                l2_enriched["sw_l2_code"].to_list(),
                l2_enriched["sw_l2_name"].to_list(),
            )
        )

        maps_df = (
            l3_enriched.select(["sw_l3_code", "sw_l2_code"])
            .with_columns(
                pl.col("sw_l2_code").replace(l2_code_to_l2_name).alias("sw_l2_name"),
                pl.col("sw_l2_code").replace(l2_code_to_l1_code).alias("sw_l1_code"),
            )
            .with_columns(
                pl.col("sw_l1_code")
                .replace(
                    dict(
                        zip(
                            l1["sw_l1_code"].to_list(),
                            l1["sw_l1_name"].to_list(),
                        )
                    )
                )
                .alias("sw_l1_name")
            )
            .select(
                [
                    "sw_l3_code",
                    "sw_l2_code",
                    "sw_l2_name",
                    "sw_l1_code",
                    "sw_l1_name",
                ]
            )
        )

        log.info(
            f"Level maps built: {maps_df.height} L3 entries, "
            f"{maps_df['sw_l2_code'].n_unique()} L2, "
            f"{maps_df['sw_l1_code'].n_unique()} L1"
        )

        maps_df.write_parquet(self.CACHE_MAP_FILE, compression="zstd")
        log.info(f"Saved level maps cache: {self.CACHE_MAP_FILE}")

        return maps_df

    def _build_full_classification(self) -> pl.DataFrame:
        try:
            maps_df = self._load_or_build_level_maps()
            self._level_maps = maps_df

            l3_info = self.sw_l3_info()
            if l3_info.is_empty():
                raise ValueError("No L3 industry metadata fetched")

            all_cons: list[pl.DataFrame] = []
            total = l3_info.height

            for idx, row in enumerate(l3_info.iter_rows(named=True), 1):
                l3_code: str = row["sw_l3_code"]
                l3_name: str = row.get("sw_l3_name", l3_code)

                if idx % 30 == 0 or idx == total:
                    log.info(f"[{idx}/{total}] Processing {l3_code} ({l3_name})")

                try:
                    cons = self.sw_l3_constituents(l3_code)
                    if cons.is_empty():
                        continue

                    map_row = maps_df.filter(pl.col("sw_l3_code") == l3_code)
                    if map_row.is_empty():
                        log.warning(f"No level-map entry for {l3_code}, skipping")
                        continue

                    map_row = map_row.row(0, named=True)

                    cons = cons.with_columns(
                        pl.lit(map_row["sw_l1_code"]).alias("sw_l1_code"),
                        pl.lit(map_row["sw_l1_name"]).alias("sw_l1_name"),
                        pl.lit(map_row["sw_l2_code"]).alias("sw_l2_code"),
                        pl.lit(map_row["sw_l2_name"]).alias("sw_l2_name"),
                        pl.lit(l3_code).alias("sw_l3_code"),
                        pl.lit(l3_name).alias("sw_l3_name"),
                    )

                    all_cons.append(cons)

                except Exception as e:
                    log.warning(f"Failed to process {l3_code} ({l3_name}): {e}")
                    continue

            if not all_cons:
                raise ValueError("No valid constituents fetched from any industry")

            combined = (
                pl.concat(all_cons, how="vertical_relaxed")
                .with_columns(
                    normalize_ts_code("ts_code_suffix", add_exchange=False).alias(
                        "ts_code"
                    )
                )
                .drop("ts_code_suffix")
                .unique(subset=["ts_code"], keep="last")
                .sort(["sw_l1_code", "sw_l2_code", "sw_l3_code", "ts_code"])
            )

            log.info(
                f"Classification built: {combined.height} stocks, "
                f"{combined['sw_l1_code'].n_unique()} L1, "
                f"{combined['sw_l2_code'].n_unique()} L2, "
                f"{combined['sw_l3_code'].n_unique()} L3"
            )

            return combined

        except Exception as e:
            log.error(f"Failed to build full industry classification: {e}")
            return pl.DataFrame()

    def sw_l1_info(self) -> pl.DataFrame:
        """Fetch Shenwan L1 industry metadata."""
        log.info("Fetching Shenwan L1 industry metadata...")
        try:
            raw = ak.sw_index_first_info()
            if raw.empty:
                log.warning("Empty L1 info response")
                return pl.DataFrame()
            df = pl.from_pandas(raw).rename(
                {
                    "行业代码": "sw_l1_code",
                    "行业名称": "sw_l1_name",
                    # Add other relevant columns if available
                }
            )
            log.info(f"Fetched {df.height} L1 industries")
            return df
        except Exception as e:
            log.error(f"L1 info fetch failed: {e}")
            return pl.DataFrame()

    def sw_l2_info(self) -> pl.DataFrame:
        """Fetch Shenwan L2 industry metadata."""
        log.info("Fetching Shenwan L2 industry metadata...")
        try:
            raw = ak.sw_index_second_info()
            if raw.empty:
                log.warning("Empty L2 info response")
                return pl.DataFrame()
            df = pl.from_pandas(raw).rename(
                {
                    "行业代码": "sw_l2_code",
                    "行业名称": "sw_l2_name",
                    "上级行业": "sw_l1_name",
                    # Add other relevant columns if available
                }
            )
            log.info(f"Fetched {df.height} L2 industries")
            return df
        except Exception as e:
            log.error(f"L2 info fetch failed: {e}")
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

    # @staticmethod
    # def derive_sw_levels(l3_code: str) -> tuple[str, str, str]:
    #     """Derive L1 and L2 codes from L3 code (Shenwan format)"""
    #     base = l3_code.replace(".SI", "").strip()
    #     if len(base) != 6 or not base.isdigit():
    #         raise ValueError(f"Invalid Shenwan L3 code format: {l3_code}")

    #     l1 = f"{base[:2]}0000.SI"
    #     l2 = f"{base[:4]}00.SI"
    #     return l1, l2, l3_code

    def get_full_classification(self, fresh: bool = False) -> pl.DataFrame:
        """Get or build the master table (ts_code → industry codes + join_date)"""
        if self._full_cls is None:
            log.info(f"Get full classification in force: {fresh}")
            self._full_cls = self.load_or_build(fresh=fresh)
        return self._full_cls

    def stock_l1_industry_cls(
        self,
        as_of_date: DateLike | None = None,
        include_names: bool = False,
        fresh: bool = False,
    ) -> pl.DataFrame:
        """
        Simplified L1 mapping: ts_code → sw_l1_code (optionally with names)
        Can be point-in-time if as_of_date is given.
        """
        df = self.get_full_classification(fresh=fresh)
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


class AkShareUniverse:
    @staticmethod
    def stock_whole() -> pl.DataFrame:
        """
        Fetch A-Share stock universe in general.
        """
        df = stock_code_whole()
        return df

    @staticmethod
    def stock_slice(
        trade_date: DateLike | None = None,
        min_listed_days: int = 365,
        remove_st: bool = True,
    ) -> pl.DataFrame:
        trade_date = to_date(trade_date) if trade_date else date.today()
        log.info(f"Building tradable universe for {trade_date}")

        universe = AkShareUniverse().stock_whole()
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


class AkShareMicro:
    @staticmethod
    def market_ohlcv(
        symbol: str,
        period: Period,
        start_date: date | None = None,
        end_date: date | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> pl.DataFrame:
        with EastMoney() as client:
            return client.stock_hist(
                symbol, period, start_date=start_date, end_date=end_date, adjust=adjust
            )
        # start_str = to_ymd_str(start_date) if start_date else "19910403"
        # end_str = (
        #     to_ymd_str(end_date) if end_date else datetime.now().strftime("%Y%m%d")
        # )

        # log.info(
        #     f"Fetching {symbol} {period} OHLCV {start_str} → {end_str} (adj={adjust})"
        # )

        # # 尝试多个数据源，直到成功获取数据
        # data_sources = [
        #     ("eastmoney", fetch_ohlcv_eastmoney),
        #     ("sina", fetch_ohlcv_sina),
        #     ("tencent", fetch_ohlcv_tencent),
        # ]

        # for source_name, fetch_func in data_sources:
        #     try:
        #         log.info(f"Trying {source_name} source for {symbol}")
        #         df = fetch_func(
        #             symbol,
        #             period,
        #             start_str,
        #             end_str,
        #             adjust=adjust if adjust is not None else "",
        #         )
        #         if df is not None and not df.is_empty():
        #             log.info(
        #                 f"Successfully fetched data from {source_name} for {symbol}"
        #             )
        #             return df
        #         elif df is not None:
        #             log.warning(f"Got empty DataFrame from {source_name} for {symbol}")
        #     except Exception as e:
        #         log.warning(f"Failed to fetch from {source_name} for {symbol}: {e}")

        # log.error(f"All data sources failed for {symbol}")
        # return pl.DataFrame()

    @staticmethod
    def batch_market_ohlcv(
        symbols: Sequence[str],
        period: Period,
        start_date: date | None = None,
        end_date: date | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> list[pl.DataFrame]:
        """
        Parallel fetch OHLCV for many symbols.
        Returns list of pl.DataFrame **in the same order** as input `codes`.
        Empty DataFrame = no data / filtered / failed.
        """
        with EastMoney() as client:

            def worker(symbol: str) -> pl.DataFrame:
                return client.stock_hist(
                    symbol,
                    period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )

            config = concur.BatchConfig.thread()
            return concur.batch_fetch(config=config, worker=worker, items=symbols)

    # @staticmethod
    # def _fetch_quarterly_em(
    #     func: Callable,
    #     date_str: str,
    #     rename_map: dict,
    #     log_name: str,
    # ) -> pl.DataFrame:
    #     log.info(f"Fetching {log_name} for {date_str}")
    #     try:
    #         raw = func(date=date_str)
    #         if raw.empty:
    #             return pl.DataFrame()
    #         df = pl.from_pandas(raw).rename(rename_map).drop("序号", strict=False)
    #         df = normalize_date_column(df, "notice_date")
    #         return df
    #     except Exception as e:
    #         log.error(f"{log_name} fetch failed for {date_str}: {e}")
    #         return pl.DataFrame()

    @staticmethod
    @cache
    def quarterly_income(
        year: int | None = None, quarter: Quarter | None = None
    ) -> pl.DataFrame:
        with EastMoney() as client:
            if year is None or quarter is None:
                year, quarter = cur_quarter_end()
            return client.quarterly_income(year, quarter)

    @staticmethod
    @cache
    def quarterly_balance(
        year: int | None = None, quarter: Quarter | None = None
    ) -> pl.DataFrame:
        with EastMoney() as client:
            if year is None or quarter is None:
                year, quarter = cur_quarter_end()
            return client.quarterly_balance(year, quarter)

    @staticmethod
    @cache
    def quarterly_cashflow(
        year: int | None = None, quarter: Quarter | None = None
    ) -> pl.DataFrame:
        with EastMoney() as client:
            if year is None or quarter is None:
                year, quarter = cur_quarter_end()
            return client.quarterly_cashflow(year, quarter)

    @staticmethod
    def _fundamental_worker(
        client: EastMoney, year: int, quarter: Quarter
    ) -> pl.DataFrame:
        inc = client.quarterly_income(year, quarter)
        bal = client.quarterly_balance(year, quarter)
        cf = client.quarterly_cashflow(year, quarter)

        if all(df.is_empty() for df in (inc, bal, cf)):
            return pl.DataFrame()

        keys = ["ts_code", "name", "notice_date"]
        return inc.join(bal, on=keys).join(cf, on=keys)

    @staticmethod
    def quarterly_fundamentals(
        year: int | None = None, quarter: Quarter | None = None
    ) -> pl.DataFrame:
        with EastMoney() as client:
            if year is None or quarter is None:
                year, quarter = cur_quarter_end()
            return AkShareMicro._fundamental_worker(client, year, quarter)

    @staticmethod
    def batch_quarterly_fundamentals(
        yqs: Sequence[tuple[int, Quarter]],
    ) -> list[pl.DataFrame]:
        def worker(yq: tuple[int, Quarter]) -> pl.DataFrame:
            year, quarter = yq
            return concur.Try()(AkShareMicro.quarterly_fundamentals)(year, quarter)

        config = concur.BatchConfig.thread()
        return concur.batch_fetch(config=config, worker=worker, items=yqs)
        # with EastMoney() as client:

        #     def worker(yq: tuple[int, Quarter]) -> pl.DataFrame:
        #         year, quarter = yq
        #         return concur.Try()(AkShareMicro._fundamental_worker)(
        #             client, year, quarter
        #         )

        #     config = concur.BatchConfig.thread()
        #     return concur.batch_fetch(
        #         config=config,
        #         worker=worker,
        #         items=yqs,
        #     )


class AkShareMacro:
    """AkShare implementation of the macro data interface."""

    # @staticmethod
    # def _fetch_and_filter_date_range(
    #     df: pl.DataFrame,
    #     start_date: date,
    #     end_date: date,
    #     date_col: str = "date",
    # ) -> pl.DataFrame:
    #     """Common final step: normalize date + filter range"""
    #     if df.is_empty():
    #         return pl.DataFrame(schema={date_col: pl.Date})

    #     df = normalize_date_column(df, date_col)
    #     return df.filter(
    #         (pl.col(date_col) >= pl.lit(start_date))
    #         & (pl.col(date_col) <= pl.lit(end_date))
    #     ).sort(date_col)

    @staticmethod
    def northbound_flow() -> pl.DataFrame:
        """
        Fetch combined northbound (沪港通 + 深港通) net buy flow.
        Columns: date, net_buy, fund_inflow, cum_net_buy
        """
        # start_dt = to_date(start_date) if start_date else date(2014, 11, 17)
        # end_dt = to_date(end_date) if end_date else date.today()

        log.info("Fetching northbound flow")

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

            return normalize_date_column(combined, date_col="date")

        except Exception as e:
            log.error(f"northbound_flow failed: {e}")
            return pl.DataFrame()

    @staticmethod
    def market_margin_short(
        # start_date: DateLike | None = None,
        # end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        Combined SH+SZ margin & short-selling balance (daily).
        Main column of interest: total_margin_balance
        """
        # start_dt = to_date(start_date) if start_date else date(2010, 3, 31)
        # end_dt = to_date(end_date) if end_date else date.today()

        log.info("Fetching market margin/short data")

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

            return normalize_date_column(combined, date_col="date")

        except Exception as e:
            log.error(f"market_margin_short failed: {e}")
            return pl.DataFrame()

    @staticmethod
    def shibor() -> pl.DataFrame:
        """
        Full SHIBOR rates history (all tenors).
        """
        # start_dt = to_date(start_date) if start_date else date(2017, 3, 17)
        # end_dt = to_date(end_date) if end_date else date.today()

        log.info("Fetching SHIBOR rates")

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

            return normalize_date_column(df, date_col="date")

        except Exception as e:
            log.error(f"shibor fetch failed: {e}")
            return pl.DataFrame()

    @staticmethod
    def csi1000_daily_ohlcv() -> pl.DataFrame:
        """
        CSI 1000 Index daily OHLCV (akshare uses em interface).
        """
        # start_dt = to_date(start_date) if start_date else date(2005, 1, 5)
        # end_dt = to_date(end_date) if end_date else date.today()

        log.info("Fetching CSI1000 daily OHLCV")

        try:
            raw = ak.stock_zh_index_daily_em()
            if raw.empty:
                return pl.DataFrame()

            df = pl.from_pandas(raw).drop("amount", strict=False)
            return normalize_date_column(df, date_col="date")

        except Exception as e:
            log.error(f"csi1000_daily_ohlcv failed: {e}")
            return pl.DataFrame()

    @staticmethod
    def csi1000qvix_daily_ohlc(
        # start_date: DateLike | None = None,
        # end_date: DateLike | None = None,
    ) -> pl.DataFrame:
        """
        CSI 1000 Volatility Index (QVIX) daily data.
        """
        # start_dt = to_date(start_date) if start_date else date(2005, 2, 9)
        # end_dt = to_date(end_date) if end_date else date.today()

        log.info("Fetching CSI1000 QVIX")

        try:
            raw = ak.index_option_1000index_qvix()
            if raw.empty:
                return pl.DataFrame()

            df = pl.from_pandas(raw).drop_nulls()
            return normalize_date_column(df, date_col="date")

        except Exception as e:
            log.error(f"csi1000qvix_daily_ohlc failed: {e}")
            return pl.DataFrame()
