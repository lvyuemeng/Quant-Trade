"""AkShare data provider implementation."""

from datetime import date, datetime

import akshare as ak
import pandas as pd
import polars as pl
import yaml

from quant_trade.utils.logger import log

from .traits import DataProvider


def _first_existing_column(
    columns: list[str], *, candidates: list[str], label: str
) -> str:
    for c in candidates:
        if c in columns:
            return c
    msg = f"Unable to find {label} column. Tried: {candidates}. Available columns: {columns}"
    raise KeyError(msg)


def _strip_exchange_suffix(code: str) -> str:
    """Convert AkShare-style stock code like '600313.SH' to '600313'."""
    return code.split(".")[0]

def _to_trade_date(trade_date: str | date | None) -> date:
    if trade_date is None:
        return datetime.now().date()
    if isinstance(trade_date, str):
        return datetime.strptime(trade_date, "%Y%m%d").date()
    return trade_date


class AkShareProvider(DataProvider):
    """AkShare implementation of the DataProvider interface."""

    def __init__(self, config_path: str = "config/data.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        log.info("Initialized AkShareProvider")

    def stock_universe(
        self, trade_date: str | date | None = None
    ) -> pl.DataFrame:
        """
        Get tradeable A-share universe for a given date.

        Filters:
        - Remove ST/*ST stocks
        - Remove suspended stocks
        - Remove stocks < 60 days from IPO
        """
        trade_date = _to_trade_date(trade_date)

        log.info(f"Fetching stock universe for {trade_date}")

        df_spot = ak.stock_zh_a_spot_em()
        df = pl.from_pandas(df_spot)

        df = df.rename(
            {
                "代码": "ts_code",
                "名称": "name",
                "成交量": "volume",
                "换手率": "turnover_rate",
            }
        )

        exclude_st = self.config.get("market", {}).get("exclude_st", True)
        if exclude_st:
            df = df.filter(~pl.col("name").str.contains("ST"))

        exclude_suspended = self.config.get("market", {}).get("exclude_suspended", True)
        if exclude_suspended:
            df = df.filter(pl.col("volume") > 0)

        log.warning(
            "IPO date filtering is not yet fully implemented for historical dates in AkShare without local DB."
        )

        min_days = self.config.get("market", {}).get("min_days_since_ipo", 60)

        return df.select(["ts_code", "name"])

    def daily_market(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "hfq"
    ) -> pl.DataFrame:
        """
        Daily OHLCV + adjustment factors.
        """
        log.info(
            f"Fetching market data for {symbol} from {start_date} to {end_date} (adjust={adjust})"
        )
        df_hist = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if df_hist.empty:
            return pl.DataFrame()

        df = pl.from_pandas(df_hist)

        # Rename to a consistent schema (matching docs/table.md naming where possible)
        df = df.rename(
            {
                "日期": "date",
                "股票代码": "ts_code",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_chg",
                "涨跌额": "change",
                "换手率": "turnover_rate",
            }
        )

        # Parse date to Date when possible (AkShare sometimes returns strings)
        if df.schema.get("date") == pl.Utf8:
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

        # Derived field used in docs/table.md
        if "change" in df.columns:
            df = df.with_columns(
                (pl.col("close") - pl.col("change")).alias("pre_close")
            )

        # Add adjustment factor if available for proper backtesting
        if adjust != "":
            factor = adjust + "-factor"
            df_adj = ak.stock_zh_a_daily(symbol=symbol,start_date=start_date,end_date=end_date,adjust=factor)
            df_adj = pl.from_pandas(df_adj)
            df = df.join(df_adj.select(["date",factor]),on="date",how="left")

        return df

    def quarterly_income_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly income statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_lrb_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """

        log.info(f"Fetching income statement for report_date={report_date}")
        raw = ak.stock_lrb_em(date=report_date)
        df = pl.from_pandas(raw)

        df = df.rename(
            {
                "股票代码": "ts_code",
                "公告日期": "announcement_date",
                "净利润": "net_profit",
                "营业总收入": "total_revenue",
                "营业利润": "operating_profit",
                "利润总额": "profit_total",
            }
        )

        if df.schema.get("announcement_date") == pl.Utf8:
            df = df.with_columns(
                pl.col("announcement_date").str.strptime(
                    pl.Date, "%Y-%m-%d", strict=False
                )
            )

        keep = [
            c
            for c in [
                "ts_code",
                "announcement_date",
                "net_profit",
                "total_revenue",
                "operating_profit",
                "profit_total",
            ]
            if c in df.columns
        ]
        return df.select(keep)

    def quarterly_balance_sheet(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_zcfz_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """

        log.info(f"Fetching balance sheet for report_date={report_date}")
        raw = ak.stock_zcfz_em(date=report_date)
        df = pl.from_pandas(raw)

        df = df.rename(
            {
                "股票代码": "ts_code",
                "公告日期": "announcement_date",
                "资产-总资产": "total_assets",
                "负债-总负债": "total_liabilities",
                "股东权益合计": "total_equity",
                "资产负债率": "debt_to_asset_pct",
            }
        )

        if df.schema.get("announcement_date") == pl.Utf8:
            df = df.with_columns(
                pl.col("announcement_date").str.strptime(
                    pl.Date, "%Y-%m-%d", strict=False
                )
            )

        keep = [
            c
            for c in [
                "ts_code",
                "announcement_date",
                "total_assets",
                "total_liabilities",
                "total_equity",
                "debt_to_asset_pct",
            ]
            if c in df.columns
        ]
        return df.select(keep)

    def quarterly_cashflow_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_xjll_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """

        log.info(f"Fetching cashflow statement for report_date={report_date}")
        raw = ak.stock_xjll_em(date=report_date)
        df = pl.from_pandas(raw)

        df = df.rename(
            {
                "股票代码": "ts_code",
                "公告日期": "announcement_date",
                "经营性现金流-现金流量净额": "operating_cf",
                "投资性现金流-现金流量净额": "investing_cf",
                "融资性现金流-现金流量净额": "financing_cf",
            }
        )

        if df.schema.get("announcement_date") == pl.Utf8:
            df = df.with_columns(
                pl.col("announcement_date").str.strptime(
                    pl.Date, "%Y-%m-%d", strict=False
                )
            )

        keep = [
            c
            for c in [
                "ts_code",
                "announcement_date",
                "operating_cf",
                "investing_cf",
                "financing_cf",
            ]
            if c in df.columns
        ]
        return df.select(keep)

    def quarterly_fundamentals(self, report_date: str) -> pl.DataFrame:
        """Fetch and merge income/balance/cashflow for the given quarter.

        Result is keyed by (ts_code, announcement_date).
        """

        income = self.quarterly_income_statement(report_date)
        balance = self.quarterly_balance_sheet(report_date)
        cashflow = self.quarterly_cashflow_statement(report_date)

        out = income.join(balance, on=["ts_code", "announcement_date"], how="outer")
        out = out.join(cashflow, on=["ts_code", "announcement_date"], how="outer")
        return out

    def northbound_flow(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch northbound flow (沪深港通 北向) using Eastmoney summary.

        Uses:
          - ak.stock_hsgt_fund_flow_summary_em()

        Returns per-day:
          - date
          - northbound_flow (all northbound)
          - northbound_flow_sh (沪股通)
          - northbound_flow_sz (深股通)

        Note: The AkShare endpoint is described as a one-shot fetch; it may contain
        only recent days depending on upstream behavior. We still filter by the requested
        range when possible.
        """

        log.info(f"Fetching northbound flow from {start_date} to {end_date}")
        raw = ak.stock_hsgt_fund_flow_summary_em()
        df = pl.from_pandas(raw)

        df = df.rename(
            {
                "交易日": "date",
                "板块": "board",
                "资金方向": "direction",
                "成交净买额": "net_buy",
            }
        )

        if df.schema.get("date") == pl.Utf8:
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

        # Filter date range
        start = datetime.strptime(start_date, "%Y%m%d").date()
        end = datetime.strptime(end_date, "%Y%m%d").date()
        df = df.filter(
            (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
        )

        # Keep only northbound
        df = df.filter(pl.col("direction") == "北向")

        # Ensure numeric
        if df.schema.get("net_buy") == pl.Utf8:
            df = df.with_columns(pl.col("net_buy").cast(pl.Float64))

        # Aggregate (unit in docs is 亿元)
        total = df.group_by("date").agg(
            pl.col("net_buy").sum().alias("northbound_flow")
        )
        sh = (
            df.filter(pl.col("board") == "沪股通")
            .group_by("date")
            .agg(pl.col("net_buy").sum().alias("northbound_flow_sh"))
        )
        sz = (
            df.filter(pl.col("board") == "深股通")
            .group_by("date")
            .agg(pl.col("net_buy").sum().alias("northbound_flow_sz"))
        )

        out = total.join(sh, on="date", how="left").join(sz, on="date", how="left")
        return out.sort("date")

    def industry_classification(
        self,
        trade_date: str | date | None = None,
        *,
        sw3_codes: list[str] | None = None,
        max_sw3: int | None = None,
    ) -> pl.DataFrame:
        """Fetch stock -> Shenwan industry (L1/L2) classification.

        Sources (documented in repo):
          - ak.sw_index_first_info()
          - ak.sw_index_second_info()
          - ak.sw_index_third_cons(symbol="...")

        Notes:
        - `sw_index_third_cons` returns per-stock rows and includes `申万1级/2级/3级` names.
        - To get a full-universe mapping, we need a list of SW3 industry codes.
          AkShare provides `ak.sw_index_third_info()` (listed in akentry and referenced by the docstring),
          but the repo may or may not include a markdown doc for it.

        Parameters:
        - trade_date: used only as the returned `date` stamp (monthly cadence per docs/table.md).
        - sw3_codes: optional override list of SW3 industry codes (e.g. ["851921.SI", ...])
        - max_sw3: optional safety cap when iterating all SW3 industries (keeps runtime bounded)
        """

        as_of = _to_trade_date(trade_date)
        log.info(f"Fetching Shenwan industry classification as of {as_of}")

        # 1) Build name->code mapping for L1/L2 using the overview endpoints.
        first = pl.from_pandas(ak.sw_index_first_info()).select(
            sw_l1_code=pl.col("行业代码"),
            sw_l1_name=pl.col("行业名称"),
        )

        second = pl.from_pandas(ak.sw_index_second_info()).select(
            sw_l2_code=pl.col("行业代码"),
            sw_l2_name=pl.col("行业名称"),
            sw_l1_name=pl.col("上级行业"),
        )

        # 2) Determine which SW3 industries to fetch constituents for.
        if sw3_codes is None:
            if not hasattr(ak, "sw_index_third_info"):
                raise AttributeError(
                    "AkShare missing sw_index_third_info(); pass sw3_codes explicitly to fetch_industry_classification"
                )

            third_info = pl.from_pandas(ak.sw_index_third_info())
            code_col = _first_existing_column(
                third_info.columns,
                candidates=["行业代码", "指数代码", "代码"],
                label="SW3 industry code",
            )
            sw3_codes = third_info.select(pl.col(code_col)).to_series().to_list()

        if max_sw3 is not None:
            sw3_codes = sw3_codes[:max_sw3]

        # 3) Fetch constituents per SW3 industry and extract per-stock SW1/SW2.
        frames: list[pd.DataFrame] = []
        for code in sw3_codes:
            df = ak.sw_index_third_cons(symbol=code)
            if df is None or df.empty:
                continue
            frames.append(df)

        if not frames:
            return pl.DataFrame(schema={"date": pl.Date, "ts_code": pl.Utf8})

        raw = pd.concat(frames, ignore_index=True)

        # Normalize columns
        df = pl.from_pandas(raw)
        df = df.rename(
            {
                "股票代码": "stock_code",
                "申万1级": "sw_l1_name",
                "申万2级": "sw_l2_name",
                "申万3级": "sw_l3_name",
            }
        )

        # Strip exchange suffix (e.g. 600313.SH -> 600313)
        if df.schema.get("stock_code") == pl.Utf8:
            df = df.with_columns(
                pl.col("stock_code")
                .map_elements(_strip_exchange_suffix)
                .alias("ts_code")
            )
        else:
            df = df.with_columns(
                pl.col("stock_code")
                .cast(pl.Utf8)
                .map_elements(_strip_exchange_suffix)
                .alias("ts_code")
            )

        df = df.with_columns(pl.lit(as_of).alias("date"))

        # Join to resolve codes (name->code)
        df = df.join(first, on="sw_l1_name", how="left")
        df = df.join(second, on=["sw_l2_name", "sw_l1_name"], how="left")

        # De-dup (one stock might appear in multiple SW3 lists in edge cases)
        df = df.unique(subset=["date", "ts_code"], keep="first")

        return df.select(
            [
                "date",
                "ts_code",
                "sw_l1_code",
                "sw_l1_name",
                "sw_l2_code",
                "sw_l2_name",
                "sw_l3_name",
            ]
        ).sort(["date", "ts_code"])

    def fetch_index_valuation_csindex(self, symbol: str = "000300") -> pl.DataFrame:
        """Fetch index valuation from the csindex valuation endpoint.

        Uses (listed in akentry):
          - ak.stock_zh_index_value_csindex(symbol=...)

        Intended to support CSI 300 (`symbol="000300"`) for `docs/table.md` fields like `csi300_pe`.
        We normalize by best-effort column detection and return a time series.
        """

        if not hasattr(ak, "stock_zh_index_value_csindex"):
            raise AttributeError("AkShare missing stock_zh_index_value_csindex")

        raw = ak.stock_zh_index_value_csindex(symbol=symbol)
        df = pl.from_pandas(raw)

        date_col = _first_existing_column(
            df.columns, candidates=["日期", "date"], label="index valuation date"
        )
        df = df.rename({date_col: "date"})

        if df.schema.get("date") == pl.Utf8:
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

        # Column detection by substring (csindex schemas vary over time)
        cols = df.columns
        pe_col = next((c for c in cols if "市盈率" in c and "TTM" in c), None) or next(
            (c for c in cols if "市盈率" in c), None
        )
        pb_col = next((c for c in cols if "市净率" in c), None)
        dy_col = next((c for c in cols if "股息率" in c), None)

        if pe_col is None:
            raise KeyError(
                f"Unable to find PE column in csindex valuation data. Columns={cols}"
            )

        out = df.select(
            [
                pl.col("date"),
                pl.col(pe_col).alias("pe"),
                (
                    pl.col(pb_col).alias("pb")
                    if pb_col
                    else pl.lit(None).cast(pl.Float64).alias("pb")
                ),
                (
                    pl.col(dy_col).alias("dividend_yield")
                    if dy_col
                    else pl.lit(None).cast(pl.Float64).alias("dividend_yield")
                ),
            ]
        ).sort("date")

        return out

    def margin_balance(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch total A-share margin balance (SH+SZ) as a daily time series.

        Uses:
          - ak.macro_china_market_margin_sh()
          - ak.macro_china_market_margin_sz()

        Per the repo docs, both return a `融资融券余额` column (unit: 元).

        Returns:
          - date
          - margin_balance (SH+SZ)
          - margin_balance_sh
          - margin_balance_sz
        """

        log.info(f"Fetching margin balance from {start_date} to {end_date}")
        start = datetime.strptime(start_date, "%Y%m%d").date()
        end = datetime.strptime(end_date, "%Y%m%d").date()

        def _load_one(market: str) -> pl.DataFrame:
            fn_name = f"macro_china_market_margin_{market}"
            if not hasattr(ak, fn_name):
                raise AttributeError(f"AkShare missing {fn_name}")
            raw = getattr(ak, fn_name)()
            df = pl.from_pandas(raw)

            date_col = _first_existing_column(
                df.columns, candidates=["日期", "date"], label=f"{market} margin date"
            )
            bal_col = _first_existing_column(
                df.columns,
                candidates=["融资融券余额"],
                label=f"{market} margin balance",
            )

            df = df.rename({date_col: "date", bal_col: f"margin_balance_{market}"})

            if df.schema.get("date") == pl.Utf8:
                df = df.with_columns(
                    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                )
            df = df.filter(
                (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
            )

            # Ensure integer/float type
            if df.schema.get(f"margin_balance_{market}") == pl.Utf8:
                df = df.with_columns(
                    pl.col(f"margin_balance_{market}").cast(pl.Float64)
                )

            return df.select(["date", f"margin_balance_{market}"]).sort("date")

        sh = _load_one("sh")
        sz = _load_one("sz")

        out = sh.join(sz, on="date", how="outer")
        out = out.with_columns(
            (
                pl.col("margin_balance_sh").fill_null(0)
                + pl.col("margin_balance_sz").fill_null(0)
            ).alias("margin_balance")
        )
        return out.sort("date")

    def fetch_macro_indicators(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        Market-wide context features.
        """
        log.info(f"Fetching macro indicators from {start_date} to {end_date}")

        # Shibor (daily)
        df_shibor_pd = ak.macro_china_shibor_all()
        df_shibor = pl.from_pandas(df_shibor_pd)

        date_col = _first_existing_column(
            df_shibor.columns,
            candidates=["日期", "date"],
            label="shibor date",
        )
        df_shibor = df_shibor.rename({date_col: "date"})

        if df_shibor.schema.get("date") == pl.Utf8:
            df_shibor = df_shibor.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

        start = datetime.strptime(start_date, "%Y%m%d").date()
        end = datetime.strptime(end_date, "%Y%m%d").date()
        df_shibor = df_shibor.filter(
            (pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end))
        )

        # Map key tenors expected by docs/table.md
        shibor_1w_col = _first_existing_column(
            df_shibor.columns,
            candidates=["1W-定价"],
            label="shibor 1W pricing",
        )
        shibor_3m_col = _first_existing_column(
            df_shibor.columns,
            candidates=["3M-定价"],
            label="shibor 3M pricing",
        )

        df_shibor = df_shibor.with_columns(
            pl.col(shibor_1w_col).alias("shibor_1w"),
            pl.col(shibor_3m_col).alias("shibor_3m"),
        )

        df = df_shibor

        # Northbound flow
        try:
            northbound = self.northbound_flow(start_date, end_date)
            df = df.join(northbound, on="date", how="left")
        except (AttributeError, KeyError) as e:
            log.warning(f"Northbound flow not available: {e}")

        # Margin balance
        try:
            margin = self.margin_balance(start_date, end_date)
            df = df.join(margin, on="date", how="left")
        except (AttributeError, KeyError) as e:
            log.warning(f"Margin balance not available: {e}")

        # CSI 300 valuation (optional)
        try:
            idx = self.fetch_index_valuation_csindex(symbol="000300")
            idx = idx.rename(
                {
                    "pe": "csi300_pe",
                    "pb": "csi300_pb",
                    "dividend_yield": "csi300_dividend_yield",
                }
            )
            df = df.join(idx, on="date", how="left")
        except (AttributeError, KeyError) as e:
            log.warning(f"CSI300 valuation not available: {e}")

        # Keep key macro columns plus any original shibor columns
        return df.sort("date")

    def fundamental_data(self, symbol: str) -> pl.DataFrame:
        """
        Fetch financial statement data.
        Note: AkShare provides multiple interfaces.
        """
        log.info(f"Fetching fundamental data for {symbol}")
        # Simplified for now: just get the latest indicators
        df_indicator = ak.stock_financial_analysis_indicator(symbol=symbol)
        return pl.from_pandas(df_indicator)

    def fetch_stock_info(self, symbol: str) -> pl.DataFrame:
        """
        Fetch individual stock information from Xueqiu (Snowball Finance).
        """
        log.info(f"Fetching stock info for {symbol}")
        # Convert to format expected by ak.stock_individual_basic_info_xq (e.g., SH601127)
        if not symbol.startswith(("SH", "SZ", "BJ")):
            # Default to SH for Shanghai if no exchange specified
            exchange = "SH"  # Default to Shanghai
            # For a more sophisticated approach, we could detect exchange based on stock code
            if symbol.startswith("6"):
                exchange = "SH"
            elif symbol.startswith(("0", "2", "3")):  # Covers 00x, 02x, 20x, 30x, 300x
                exchange = "SZ"
            elif symbol.startswith("8"):
                exchange = "BJ"
            formatted_symbol = f"{exchange}{symbol}"
        else:
            formatted_symbol = symbol

        df_raw = ak.stock_individual_basic_info_xq(symbol=formatted_symbol)
        df = pl.from_pandas(df_raw)

        df = df.rename({"item": "field", "value": "value"})

        df = df.with_columns(pl.lit(symbol).alias("ts_code"))

        return df

    def fetch_stock_name_code_map(self) -> pl.DataFrame:
        """
        Fetch A-share stock code and name mapping.
        """
        log.info("Fetching stock code and name mapping")
        df_raw = ak.stock_info_a_code_name()
        df = pl.from_pandas(df_raw)

        df = df.rename({"code": "ts_code", "name": "name"})

        return df
