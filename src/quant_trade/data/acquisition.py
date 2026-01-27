from datetime import date, datetime

import akshare as ak
import polars as pl

from quant_trade.utils.logger import log

from .provider.akshare import AkShareProvider
from .provider.traits import DataProvider


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


class DataAcquisition:
    """
    Data acquisition module that uses a data provider interface.
    Focuses on China A-share market data.
    """

    def __init__(
        self,
        provider: DataProvider | None = None,
        config_path: str = "config/data.yaml",
    ):
        if provider is None:
            self.provider = AkShareProvider(config_path)
        else:
            self.provider = provider

        log.info(
            f"Initialized DataAcquisition with provider: {type(self.provider).__name__}"
        )

    def fetch_stock_universe(
        self, trade_date: str | date | None = None
    ) -> pl.DataFrame:
        """
        Get tradeable A-share universe for a given date.

        Filters:
        - Remove ST/*ST stocks
        - Remove suspended stocks
        - Remove stocks < 60 days from IPO
        """
        return self.provider.fetch_stock_universe(trade_date)

    def fetch_market_data(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "hfq"
    ) -> pl.DataFrame:
        """
        Daily OHLCV + adjustment factors.
        """
        return self.provider.fetch_market_data(symbol, start_date, end_date, adjust)

    def fetch_quarterly_income_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly income statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_lrb_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        return self.provider.fetch_quarterly_income_statement(report_date)

    def fetch_quarterly_balance_sheet(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly balance sheet snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_zcfz_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        return self.provider.fetch_quarterly_balance_sheet(report_date)

    def fetch_quarterly_cashflow_statement(self, report_date: str) -> pl.DataFrame:
        """Fetch quarterly cashflow statement snapshot for all A-shares.

        Uses Eastmoney batch API:
          - ak.stock_xjll_em(date="YYYY0331|YYYY0630|YYYY0930|YYYY1231")

        Returns a per-stock table keyed by (ts_code, announcement_date).
        """
        return self.provider.fetch_quarterly_cashflow_statement(report_date)

    def fetch_quarterly_fundamentals(self, report_date: str) -> pl.DataFrame:
        """Fetch and merge income/balance/cashflow for the given quarter.

        Result is keyed by (ts_code, announcement_date).
        """
        return self.provider.fetch_quarterly_fundamentals(report_date)

    def fetch_northbound_flow(self, start_date: str, end_date: str) -> pl.DataFrame:
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
        return self.provider.fetch_northbound_flow(start_date, end_date)

    def fetch_industry_classification(
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

        return self.provider.fetch_industry_classification(
            trade_date, sw3_codes=sw3_codes, max_sw3=max_sw3
        )

    def fetch_index_valuation_csindex(self, symbol: str = "000300") -> pl.DataFrame:
        """Fetch index valuation from the csindex valuation endpoint.

        Uses (listed in akentry):
          - ak.stock_zh_index_value_csindex(symbol=...)

        Intended to support CSI 300 (`symbol="000300"`) for `docs/table.md` fields like `csi300_pe`.
        We normalize by best-effort column detection and return a time series.
        """

        return self.provider.fetch_index_valuation_csindex(symbol)

    def fetch_margin_balance(self, start_date: str, end_date: str) -> pl.DataFrame:
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

        return self.provider.fetch_margin_balance(start_date, end_date)

    def fetch_fundamental_data(self, symbol: str) -> pl.DataFrame:
        """
        Fetch financial statement data.
        Note: AkShare provides multiple interfaces.
        """
        return self.provider.fetch_fundamental_data(symbol)

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
            northbound = self.fetch_northbound_flow(start_date, end_date)
            df = df.join(northbound, on="date", how="left")
        except (AttributeError, KeyError) as e:
            log.warning(f"Northbound flow not available: {e}")

        # Margin balance
        try:
            margin = self.fetch_margin_balance(start_date, end_date)
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

    def fetch_stock_info(self, symbol: str) -> pl.DataFrame:
        """
        Fetch individual stock information from Xueqiu (Snowball Finance).
        """
        return self.provider.fetch_stock_info(symbol)

    def fetch_stock_name_code_map(self) -> pl.DataFrame:
        """
        Fetch A-share stock code and name mapping.
        """
        return self.provider.fetch_stock_name_code_map()


if __name__ == "__main__":
    # Test
    acq = DataAcquisition()
    universe = acq.fetch_stock_universe()
    print(f"Universe size: {len(universe)}")
    print(universe.head())

    # Test market data
    if len(universe) > 0:
        sample_symbol = universe["ts_code"][0]
        hist = acq.fetch_market_data(sample_symbol, "20230101", "20231231")
        print(f"History for {sample_symbol}:")
        print(hist.head())
