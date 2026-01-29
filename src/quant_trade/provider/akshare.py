"""AkShare data provider implementation."""

from datetime import date, datetime, timedelta

import akshare as ak
import polars as pl

from quant_trade.provider.utils import (
    AdjustCN,
    DateLike,
    Period,
    Quarter,
    date_f_datelike,
    normal_df_time,
    str_f_datelike,
    str_f_quater,
)
from quant_trade.config.logger import log


def stock_code_sh_whole() -> pl.DataFrame:
    df = ak.stock_info_sh_name_code(symbol="主板A股")
    df = pl.from_pandas(df)

    df = df.rename(
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "上市日期": "publish_date",
        }
    ).select(["ts_code", "name", "publish_date"])
    df = normal_df_time(df, date_col="publish_date")
    return df


def stock_code_sz_whole() -> pl.DataFrame:
    df = ak.stock_info_sz_name_code(symbol="A股列表")
    df = pl.from_pandas(df)

    df = df.rename(
        {
            "A股代码": "ts_code",
            "A股简称": "name",
            "A股上市日期": "publish_date",
        }
    ).select(["ts_code", "name", "publish_date"])
    df = normal_df_time(df, date_col="publish_date")
    return df


def stock_code_bj_whole() -> pl.DataFrame:
    df = ak.stock_info_bj_name_code()
    df = pl.from_pandas(df)

    df = df.rename(
        {
            "证券代码": "ts_code",
            "证券简称": "name",
            "上市日期": "publish_date",
        }
    ).select(["ts_code", "name", "publish_date"])
    df = normal_df_time(df, date_col="publish_date")
    return df


def stock_code_sus(date: str | datetime) -> pl.DataFrame:
    df = ak.stock_tfp_em(date=str_f_datelike(date))
    df = pl.from_pandas(df).select(
        [pl.col("代码").alias("ts_code"), pl.col("名称").alias("name")]
    )
    df = df.with_columns(
        [pl.col("ts_code").cast(pl.String), pl.col("name").cast(pl.String)]
    )

    return df


def stock_code_sh_delist() -> pl.DataFrame:
    df = ak.stock_info_sh_delist(symbol="全部")
    df = pl.from_pandas(df).select(
        [
            pl.col("公司代码").alias("ts_code"),
            pl.col("公司简称").alias("name"),
            pl.col("暂停上市日期").alias("off_date"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("ts_code").cast(pl.String),
            pl.col("name").cast(pl.String),
            pl.col("off_date").cast(pl.String).str.to_date(strict=False),
        ]
    )

    return df


def stock_code_sz_delist() -> pl.DataFrame:
    df = ak.stock_info_sz_delist(symbol="终止上市公司")

    df = pl.from_pandas(df).select(
        [
            pl.col("证券代码").alias("ts_code"),
            pl.col("证券简称").alias("name"),
            pl.col("终止上市日期").alias("off_date"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("ts_code").cast(pl.String),
            pl.col("name").cast(pl.String),
            pl.col("off_date").cast(pl.String).str.to_date(strict=False),
        ]
    )

    return df


def stock_code_delist() -> pl.DataFrame:
    sh_df = stock_code_sh_delist()
    sz_df = stock_code_sz_delist()
    return pl.concat([sh_df, sz_df], how="vertical_relaxed")


def stock_code_whole() -> pl.DataFrame:
    sh_df = stock_code_sh_whole()
    sz_df = stock_code_sz_whole()
    bj_df = stock_code_bj_whole()

    return pl.concat([sh_df, sz_df, bj_df], how="vertical_relaxed")


class AkShareMicro:
    """AkShare implementation of the micro data interface."""

    def __init__(self):
        # with open(config_path) as f:
        #     self.config = yaml.safe_load(f)
        log.info("Initialized AkShareMicro.")

    def stock_whole(self) -> pl.DataFrame:
        """
        Fetch A-Share stock universe in general.
        """
        df = stock_code_whole()
        return df

    def stock_slice(self, trade_date: DateLike | None = None) -> pl.DataFrame:
        """
        Get tradeable A-share universe for a given date from the given complete stock list.

        Filters:
            - Remove ST/*ST(delist) stocks
            - Remove newly listed stocks >= 360 days

        Args:
            trade_date: date in format 'YYYYMMDD'.

        Returns:
            DataFrame with 'ts_code' and 'name' columns
        """
        trade_date = (
            date_f_datelike(trade_date) if trade_date else datetime.now().date()
        )
        df = self.stock_whole()
        log.info(f"Fetching stock universe for {trade_date}")

        # Filter out stocks that were listed after the target date
        df = df.filter(pl.col("publish_date") <= pl.lit(trade_date))

        # Filter out stocks that are too new (less than min_days since IPO)
        df = df.filter(
            pl.col("publish_date") <= pl.lit(trade_date - timedelta(days=365))
        )

        # Filter out ST/*ST stocks
        df = df.filter(~pl.col("name").str.contains("ST", literal=True))
        delist = stock_code_delist()
        delisted_stocks = delist.filter(pl.col("off_date") <= pl.lit(trade_date))
        df = df.join(delisted_stocks.select("ts_code"), on="ts_code", how="anti")

        df = df.sort("ts_code").select(["ts_code", "name"])

        return df

    def market_ohlcv(
        self,
        symbol: str,
        period: Period,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        adjust: AdjustCN | None = "hfq",
    ) -> pl.DataFrame:
        """
        Daily OHLCV + adjustment factors.
        """
        start_date = str_f_datelike(start_date) if start_date else "19910403"
        end_date = (
            str_f_datelike(end_date)
            if end_date
            else datetime.now().date().strftime("%Y%m%d")
        )
        log.info(
            f"Fetching market data for {symbol} from {start_date} to {end_date} (adjust={adjust})"
        )

        params = {
            "symbol": symbol,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust if adjust else "",
            "timeout": None,
        }
        df_hist = ak.stock_zh_a_hist(**params)
        if df_hist.empty:
            return pl.DataFrame()
        df = pl.from_pandas(df_hist)

        df = df.rename(
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
            },
        )
        df = normal_df_time(df, date_col="date")

        return df

    def quarterly_income_statement(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly income statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_lrb_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        report_date = str_f_quater(year, quarter)
        log.info(f"Fetching income statement for report_date={report_date}")
        raw = ak.stock_lrb_em(date=report_date)
        df = pl.from_pandas(raw)

        df = df.rename(
            {
                "股票代码": "ts_code",
                "股票简称": "name",
                # dates
                "公告日期": "announcement_date",
                # core profit & revenue
                "净利润": "net_profit",
                "营业利润": "operating_profit",
                "利润总额": "total_profit",
                "营业总收入": "total_revenue",
                # growth (YoY)
                "净利润同比": "net_profit_yoy",
                "营业总收入同比": "total_revenue_yoy",
                # expenses (cost structure)
                "营业总支出-营业支出": "operating_cost",
                "营业总支出-销售费用": "selling_cost",
                "营业总支出-管理费用": "admin_cost",
                "营业总支出-财务费用": "finance_cost",
                "营业总支出-营业总支出": "total_cost",
            }
        ).drop("序号")
        df = normal_df_time(df, date_col="announcement_date")

        return df

    def quarterly_balance_sheet(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_zcfz_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        report_date = str_f_quater(year, quarter)
        log.info(f"Fetching balance sheet for report_date={report_date}")
        raw = ak.stock_zcfz_em(date=report_date)
        df = pl.from_pandas(raw)

        df = df.rename(
            {
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
        ).drop("序号")
        df = normal_df_time(df, date_col="announcement_date")

        return df

    def quarterly_cashflow_statement(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_xjll_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        report_date = str_f_quater(year, quarter)
        log.info(f"Fetching cashflow statement for report_date={report_date}")
        raw = ak.stock_xjll_em(date=report_date)
        df = pl.from_pandas(raw)

        df = df.rename(
            {
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
        ).drop("序号")
        df = normal_df_time(df, date_col="announcement_date")

        return df

    def quarterly_fundamentals(
        self, year: int | None, quarter: Quarter
    ) -> pl.DataFrame:
        """Fetch and merge income/balance/cashflow for the given quarter.

        Result is keyed by (ts_code, announcement_date).
        """
        income = self.quarterly_income_statement(year, quarter)
        balance = self.quarterly_balance_sheet(year, quarter)
        cashflow = self.quarterly_cashflow_statement(year, quarter)

        df = income.join(balance, on=["ts_code", "announcement_date"], how="semi")
        df = df.join(cashflow, on=["ts_code", "announcement_date"], how="semi")

        return df


class AkShareMacro:
    """AkShare implementation of the macro data interface."""

    def __init__(self):
        # with open(config_path) as f:
        #     self.config = yaml.safe_load(f)

        log.info("Initialized AkShareMacro.")

    def northbound_flow(
        self, start_date: DateLike | None = None, end_date: DateLike | None = None
    ) -> pl.DataFrame:
        """Fetch northbound flow (沪深港通 北向) using Eastmoney historical data.

        Uses:
        - ak.stock_hsgt_hist_em()

        Returns per-day:
        - date
        - northbound_flow

        Note: This endpoint provides full historical data starting from 2014-11-17.
        """
        start_date = date_f_datelike(start_date) if start_date else date(2014, 11, 17)
        end_date = date_f_datelike(end_date) if end_date else datetime.now().date()
        log.info(f"Fetching northbound flow from {start_date} to {end_date}")

        sh_df = ak.stock_hsgt_hist_em(symbol="港股通沪")
        sh_df = pl.from_pandas(sh_df)
        sz_df = ak.stock_hsgt_hist_em(symbol="港股通深")
        sz_df = pl.from_pandas(sz_df)

        def _prepare(df: pl.DataFrame) -> pl.DataFrame:
            return df.rename(
                {
                    "日期": "date",
                    "当日成交净买额": "net_buy",
                    "当日资金流入": "fund_inflow",
                    "历史累计净买额": "cum_net_buy",
                }
            ).select(["date", "net_buy", "fund_inflow", "cum_net_buy"])

        sh_df = _prepare(sh_df)
        sz_df = _prepare(sz_df)

        combined = pl.concat([sz_df, sh_df]).group_by("date").sum().sort("date")
        combined = normal_df_time(combined, date_col="date").filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        return combined

    def market_margin_short(
        self, start_date: DateLike | None = None, end_date: DateLike | None = None
    ) -> pl.DataFrame:
        """Fetch total A-share margin + short data (SH+SZ) as a daily time series.

        Uses:
          - ak.macro_china_market_margin_sh()
          - ak.macro_china_market_margin_sz()

        Per the repo docs, both return a `融资融券余额` column (unit: 元).

        Returns:
          - date
          - margin_balance (SH+SZ)
        """
        start_date = date_f_datelike(start_date) if start_date else date(2010, 3, 31)
        end_date = date_f_datelike(end_date) if end_date else datetime.now().date()
        log.info(f"Fetching margin balance from {start_date} to {end_date}")

        sz_df = ak.macro_china_market_margin_sz()
        sz_df = pl.from_pandas(sz_df)
        sh_df = ak.macro_china_market_margin_sh()
        sh_df = pl.from_pandas(sh_df)

        def process_margin_data(df: pl.DataFrame) -> pl.DataFrame:
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
                .drop("融券余量")
                .with_columns(pl.all().exclude("date").cast(pl.Float64))
            )

        sz_df = process_margin_data(sz_df)
        sh_df = process_margin_data(sh_df)
        combined = pl.concat([sz_df, sh_df]).group_by("date").sum().sort("date")
        combined = normal_df_time(combined, date_col="date").filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        return combined

    def shibor(
        self, start_date: DateLike | None = None, end_date: DateLike | None = None
    ) -> pl.DataFrame:
        """Fetch the shibor data as a daily time series


        :param start_date: Description
        :type start_date: DateLike | None
        :param end_date: Description
        :type end_date: DateLike | None
        :return: Description
        :rtype: DataFrame
        """
        start_date = date_f_datelike(start_date) if start_date else date(2017, 3, 17)
        end_date = date_f_datelike(end_date) if end_date else datetime.now().date()
        log.info(f"Fetching shibor from {start_date} to {end_date}")

        df = ak.macro_china_shibor_all()
        df = pl.from_pandas(df)
        df = df.rename(
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
        df = normal_df_time(df, date_col="date").filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        return df

    def csi1000_daily_ohlcv(
        self, start_date: DateLike | None = None, end_date: DateLike | None = None
    ) -> pl.DataFrame:
        start_date = date_f_datelike(start_date) if start_date else date(2005, 1, 5)
        end_date = date_f_datelike(end_date) if end_date else datetime.now().date()
        log.info(f"Fetching csi1000 daily from {start_date} to {end_date}")

        df = ak.stock_zh_index_daily_em()
        df = pl.from_pandas(df).drop("amount")
        df = normal_df_time(df)
        df = normal_df_time(df, date_col="date").filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        return df

    def csi1000qvix_daily_ohlc(
        self, start_date: DateLike | None = None, end_date: DateLike | None = None
    ) -> pl.DataFrame:
        start_date = date_f_datelike(start_date) if start_date else date(2005, 2, 9)
        end_date = date_f_datelike(end_date) if end_date else datetime.now().date()
        log.info(f"Fetching csi1000qvix daily from {start_date} to {end_date}")

        df = ak.index_option_1000index_qvix()
        df = pl.from_pandas(df).drop_nulls()
        df = normal_df_time(df)
        df = normal_df_time(df, date_col="date").filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        return df


class SWIndustryCls:
    """
    Build and maintain Shenwan industry classification data

    Key features:
    1. Fetch all L3 industry constituents
    2. Derive L1/L2 codes from L3 codes
    3. Handle temporal changes (stocks changing industries)
    4. Provide mapping for normalization
    """

    def __init__(self):
        self.cache = {}

    def sw_l3_info(self) -> pl.DataFrame:
        """
        Fetch metadata for all L3 industries

        Returns:
        - Industry codes, names, constituent counts
        """
        log.info("Fetching Shenwan L3 industry information...")

        df = ak.sw_index_third_info()
        df = pl.from_pandas(df)
        df = df.rename(
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

    def sw_l3_constituents(self, sw_l3_code: str) -> pl.DataFrame:
        """
        Fetch all stocks in a specific L3 industry

        Returns:
        - Stock code, name, join date, and fundamentals
        """
        df = ak.sw_index_third_cons(symbol=sw_l3_code)
        if df.empty:
            log.warning(f"No constituents found for {sw_l3_code}")
            return pl.DataFrame()

        df = pl.from_pandas(df)
        df = df.rename(
            {
                "股票代码": "ts_code_suffix",
                "股票简称": "name",
                "纳入时间": "join_date",
                "申万1级": "sw_l1_name",
                "申万2级": "sw_l2_name",
                "申万3级": "sw_l3_name",
            }
        ).select(
            ["ts_code_suffix", "join_date", "sw_l1_name", "sw_l2_name", "sw_l3_name"]
        )
        df = df.with_columns([pl.lit(sw_l3_code).alias("sw_l3_code")])

        return df

    def stock_industry_cls(self) -> pl.DataFrame:
        """
        Build complete industry classification for all stocks

        Process:
        1. Fetch all L3 industries
        2. For each L3, fetch constituents
        3. Derive L1/L2 codes
        4. Combine into master table
        """
        log.info("Building complete industry classification...")

        industry_info = self.sw_l3_info()
        all_cons: list[pl.DataFrame] = []
        total_rows = len(industry_info)

        for i, row in enumerate(industry_info.iter_rows(named=True), 1):
            sw_l3_code = row["sw_l3_code"]

            if i % 50 == 0:
                log.info(f"Processing industry {i}/{total_rows}: {sw_l3_code}")

            cons = self.sw_l3_constituents(sw_l3_code)
            if cons.is_empty():
                continue

            sw_codes = self.swl_f_l3(sw_l3_code)
            cons = cons.with_columns(
                [
                    pl.lit(sw_codes["l1"]).alias("sw_l1_code"),
                    pl.lit(sw_codes["l2"]).alias("sw_l2_code"),
                    pl.lit(sw_codes["l3"]).alias("sw_l3_code"),
                ]
            )

            all_cons.append(cons)

        if not all_cons:
            raise ValueError("No industry data fetched!")

        combined = pl.concat(all_cons, how="vertical_relaxed")
        combined = combined.with_columns(
            [
                pl.col("ts_code_suffix")
                .map_elements(self.strip_ts_code, return_dtype=pl.String)
                .alias("ts_code")
            ]
        ).unique(subset=["ts_code"], keep="first")
        combined = normal_df_time(combined, "join_date")

        return combined

    def stock_l1_industry_cls(self, start_date: DateLike | None = None) -> pl.DataFrame:
        df = self._l1_industry_cls(self.stock_industry_cls())
        return self._time_industry_cls(df, start_date) if start_date else df

    def _time_industry_cls(
        self, cls_df: pl.DataFrame, start_date: DateLike
    ) -> pl.DataFrame:
        """
        Expand industry classification to daily time series

        Why: We need industry classification for every trading day

        Approach:
        - For each stock, replicate classification from join_date onwards
        - If join_date is before start_date, use start_date

        Result: (date, ts_code, sw_l1_code, ...) for every trading day
        """
        start_date = date_f_datelike(start_date)
        cls_df.filter(pl.col("join_date") >= start_date)

        return cls_df

    def _l1_industry_cls(self, cls_df: pl.DataFrame) -> pl.DataFrame:
        """
        Create quick lookup: ts_code -> sw_l1_code

        For use in normalization
        """
        return cls_df.select(["ts_code", "join_date", "sw_l1_code"])

    def swl_f_l3(self, sw_l3_code: str) -> dict[str, str]:
        """
        Derive L1 and L2 codes from L3 code

        Shenwan code structure:
        - L3: 850111.SI (6 digits before .SI)
        - L2: 8501xx.SI (first 4 digits, pad with 00)
        - L1: 85xxxx.SI (first 2 digits, pad with 0000)

        Example:
        - Input: "850111.SI"
        - L1: "850000.SI" (农林牧渔)
        - L2: "850100.SI" (种植业与林业)
        - L3: "850111.SI" (种子)
        """

        code_base = sw_l3_code.replace(".SI", "")
        if len(code_base) != 6:
            log.warning(f"Invalid Shenwan code format: {sw_l3_code}")
            raise ValueError(
                f"Invalid Shenwan code format: {sw_l3_code}. Must be 6 characters after removing '.SI'"
            )

        l1_code = code_base[:2] + "0000" + ".SI"
        l2_code = code_base[:4] + "00" + ".SI"

        return {"l1": l1_code, "l2": l2_code, "l3": sw_l3_code}

    def suffix_ts_code(self, code: str) -> str:
        """
        Standardize stock code format

        AkShare format: 600313.SH, 000713.SZ
        Tushare format: 600313.SH, 000713.SZ (same)

        Ensure consistent format
        """
        if "." in code and len(code.split(".")[0]) == 6:
            return code

        if len(code) == 6:
            if code.startswith("6"):
                return f"{code}.SH"
            elif code.startswith(("0", "3", "4")):
                return f"{code}.SZ"
            elif code.startswith(("8", "4")):
                return f"{code}.BJ"

        log.warning(f"Could not standardize stock code: {code}")
        raise ValueError(f"Could not standardize stock code: {code}")

    def strip_ts_code(self, code: str) -> str:
        """
        Extract base stock code by removing exchange suffix

        Args:
            code: Stock code in format XXXXXX.EX (e.g., 600313.SH)

        Returns:
            str: Base 6-digit code (e.g., "600313")

        Raises:
            ValueError: If code format is invalid
        """
        # Check if it's already a 6-digit base code
        if "." not in code and len(code) == 6 and code.isdigit():
            return code

        # Split and validate
        parts = code.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid code format: {code}")

        prefix, suffix = parts
        suffix = suffix.upper()

        if not (len(prefix) == 6 and prefix.isdigit()):
            raise ValueError(f"Invalid prefix in code: {code}")

        valid_suffixes = {"SH", "SZ", "BJ"}
        if suffix not in valid_suffixes:
            # Still return prefix but warn
            log.warning(f"Unusual exchange suffix in code: {code}")

        return prefix
