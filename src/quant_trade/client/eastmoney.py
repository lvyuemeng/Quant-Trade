"""
East Money data provider with separated concerns.

Architecture:
- Fetcher: Independent HTTP fetching (network only)
- Parser: Converts raw JSON to structured data
- Builder: Constructs final output DataFrame
- ReportPipe: Specialized pipe for quarterly financial reports
- KlinePipe: Specialized pipe for stock historical data
- EastMoney: Facade that composes all components (no inner Fetcher exposed)

Uses generic traits from trait.py for reusable patterns.
"""

import random
import time
from dataclasses import dataclass
from typing import Final
import warnings

import polars as pl

warnings.filterwarnings(action="ignore", category=FutureWarning)

# Import protocols and base classes from trait.py
from .trait import (
    BaseBuilder,
    BaseFetch,
    BaseParser,
    ColumnSpec,
)

# =============================================================================
# Constants
# =============================================================================

EAST_MONEY_API_URL: Final[str] = "https://datacenter-web.eastmoney.com/api/data/v1/get"
KLINE_API_URL: Final[str] = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# =============================================================================
# Helper Functions
# =============================================================================


def _build_quarterly_date(year: int, quarter: int) -> str:
    """
    Build formatted date string from year and quarter.

    Args:
        year: Report year (e.g., 2024)
        quarter: Report quarter (1, 2, 3, or 4)

    Returns:
        Formatted date string (e.g., "2024-03-31")
    """
    quarter_end_months = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    month, day = quarter_end_months[quarter].split("-")
    return f"{year}-{month}-{day}"


def _build_quarterly_params(report_name: str, formatted_date: str) -> dict:
    """
    Build params for quarterly report API.

    Args:
        report_name: Report name (e.g., "RPT_DMSK_FN_INCOME")
        formatted_date: Formatted date string (e.g., "2024-03-31")

    Returns:
        Dictionary of API parameters
    """
    return {
        "sortColumns": "NOTICE_DATE,SECURITY_CODE",
        "sortTypes": "-1,-1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": report_name,
        "columns": "ALL",
        "filter": f"""(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")(REPORT_DATE='{formatted_date}')""",
    }


def _build_kline_params(
    symbol: str,
    period: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> tuple[str, dict]:
    """
    Build params for kline API.

    Args:
        symbol: Stock code (e.g., "000001")
        period: "daily", "weekly", or "monthly"
        start_date: Start date in "YYYYMMDD" format
        end_date: End date in "YYYYMMDD" format
        adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

    Returns:
        Tuple of (secid, params_dict)
    """
    # Determine market code
    market_code = 1 if symbol.startswith("6") else 0

    # Build parameter mappings
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}

    # Build params
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_dict[period],
        "fqt": adjust_dict[adjust],
        "secid": f"{market_code}.{symbol}",
        "beg": start_date,
        "end": end_date,
    }

    return f"{market_code}.{symbol}", params


# =============================================================================
# Concrete Implementations
# =============================================================================


class EastMoneyFetch(BaseFetch):
    """
    EastMoney-specific Fetcher - extends BaseFetch.

    Anti-blocking features:
    - Randomized delays between requests
    - User-Agent rotation
    - Session pooling with retry strategy
    - Concurrent fetching with ThreadPoolExecutor
    """

    # User agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/2010",
    ]

    def fetch_initial(self, url: str, params: dict) -> dict:
        """Fetch initial page to get total pages."""
        self._rate_limit()
        response = self._session.get(
            url,
            params=params,
            headers=self._get_headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def fetch_page(self, url: str, params: dict, page: int) -> dict:
        """Fetch single page."""
        self._rate_limit()
        params_copy = params.copy()
        params_copy["pageNumber"] = str(page)

        response = self._session.get(
            url,
            params=params_copy,
            headers=self._get_headers(),
            timeout=(10, 30),
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the session."""
        self._session.close()


class FundemantalParser(BaseParser):
    """
    EastMoney-specific Parser - extends BaseParser.

    DATA_PATH configured to extract data from EastMoney API response.
    """

    DATA_PATH = ("result", "data")

    def clean(self, data: list[dict]) -> pl.DataFrame:
        """Clean and convert to Polars DataFrame with deduplication."""
        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        return df


class KlineParser(BaseParser):
    """
    Parser for stock kline (historical) data.

    DATA_PATH configured for kline API response structure.
    Data is returned as comma-separated strings that need parsing.
    """

    DATA_PATH = ("data", "klines")

    def parse(self, raw: dict) -> list[dict]:
        """
        Parse kline response and extract data list.

        Args:
            raw: Raw JSON response dictionary

        Returns:
            List of data dictionaries
        """
        if not raw.get("data"):
            return []

        klines = raw.get("data", {}).get("klines", [])
        if not klines:
            return []

        # Parse comma-separated strings into dictionaries
        column_names = [
            "date", "open", "close", "high", "low",
            "volume", "amount", "amplitude", "change_pct",
            "change_amt", "turnover"
        ]

        data = []
        for kline in klines:
            values = kline.split(",")
            if len(values) == len(column_names):
                data.append(dict(zip(column_names, values)))

        return data


# =============================================================================
# Builders - Use ColumnSpec pattern from trait.py
# =============================================================================


class IncomeBuilder(BaseBuilder):
    """
    Builder for LRB (Income Statement) data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("SECURITY_CODE", "stock_code"),
        ColumnSpec("SECURITY_NAME_ABBR", "stock_name"),
        ColumnSpec("NOTICE_DATE", "notice_date"),
        ColumnSpec("NETPROFIT", "net_profit", pl.Float64),
        ColumnSpec("TOTAL_OPERATE_INCOME", "total_operate_income", pl.Float64),
        ColumnSpec("TOTAL_OPERATE_COST", "total_operate_cost", pl.Float64),
        ColumnSpec("OPERATE_COST", "operate_cost", pl.Float64),
        ColumnSpec("SALE_EXPENSE", "sale_expense", pl.Float64),
        ColumnSpec("MANAGE_EXPENSE", "manage_expense", pl.Float64),
        ColumnSpec("FINANCE_EXPENSE", "finance_expense", pl.Float64),
        ColumnSpec("OPERATE_PROFIT", "operate_profit", pl.Float64),
        ColumnSpec("TOTAL_PROFIT", "total_profit", pl.Float64),
        ColumnSpec("TOTAL_OPERATE_INCOME_SQ", "total_operate_income_yoy", pl.Float64),
        ColumnSpec("NETPROFIT_SQ", "net_profit_yoy", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "seq",
        "stock_code",
        "stock_name",
        "net_profit",
        "net_profit_yoy",
        "total_operate_income",
        "total_operate_income_yoy",
        "operate_cost",
        "sale_expense",
        "manage_expense",
        "finance_expense",
        "total_operate_cost",
        "operate_profit",
        "total_profit",
        "notice_date",
    ]

    DATE_COL = "notice_date"
    DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")


class BalanceSheetBuilder(BaseBuilder):
    """
    Builder for Balance Sheet data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("SECURITY_CODE", "stock_code"),
        ColumnSpec("SECURITY_NAME_ABBR", "stock_name"),
        ColumnSpec("NOTICE_DATE", "notice_date"),
        ColumnSpec("TOTAL_ASSETS", "total_assets", pl.Float64),
        ColumnSpec("MONETARY_CAPITAL", "monetary_capital", pl.Float64),
        ColumnSpec("ACCOUNTS_RECEIVABLE", "accounts_receivable", pl.Float64),
        ColumnSpec("INVENTORY", "inventory", pl.Float64),
        ColumnSpec("TOTAL_LIABILITIES", "total_liabilities", pl.Float64),
        ColumnSpec("ACCOUNTS_PAYABLE", "accounts_payable", pl.Float64),
        ColumnSpec("ADVANCE_RECEIPTS", "advance_receipts", pl.Float64),
        ColumnSpec("TOTAL_EQUITY", "total_equity", pl.Float64),
        ColumnSpec("TOTAL_ASSETS_YOY", "total_assets_yoy", pl.Float64),
        ColumnSpec("TOTAL_LIABILITIES_YOY", "total_liabilities_yoy", pl.Float64),
        ColumnSpec("DEBT_ASSET_RATIO", "debt_asset_ratio", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "seq",
        "stock_code",
        "stock_name",
        "total_assets",
        "total_assets_yoy",
        "monetary_capital",
        "accounts_receivable",
        "inventory",
        "total_liabilities",
        "total_liabilities_yoy",
        "accounts_payable",
        "advance_receipts",
        "debt_asset_ratio",
        "total_equity",
        "notice_date",
    ]

    DATE_COL = "notice_date"
    DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")


class CashFlowBuilder(BaseBuilder):
    """
    Builder for Cash Flow Statement data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("SECURITY_CODE", "stock_code"),
        ColumnSpec("SECURITY_NAME_ABBR", "stock_name"),
        ColumnSpec("NOTICE_DATE", "notice_date"),
        ColumnSpec("NET_CASH_FLOW", "net_cash_flow", pl.Float64),
        ColumnSpec("NET_CASH_FLOW_YOY", "net_cash_flow_yoy", pl.Float64),
        ColumnSpec("OPERATE_CASH_FLOW", "operate_cash_flow", pl.Float64),
        ColumnSpec("OPERATE_CASH_FLOW_RATIO", "operate_cash_flow_ratio", pl.Float64),
        ColumnSpec("INVEST_CASH_FLOW", "invest_cash_flow", pl.Float64),
        ColumnSpec("INVEST_CASH_FLOW_RATIO", "invest_cash_flow_ratio", pl.Float64),
        ColumnSpec("FINANCE_CASH_FLOW", "finance_cash_flow", pl.Float64),
        ColumnSpec("FINANCE_CASH_FLOW_RATIO", "finance_cash_flow_ratio", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "seq",
        "stock_code",
        "stock_name",
        "net_cash_flow",
        "net_cash_flow_yoy",
        "operate_cash_flow",
        "operate_cash_flow_ratio",
        "invest_cash_flow",
        "invest_cash_flow_ratio",
        "finance_cash_flow",
        "finance_cash_flow_ratio",
        "notice_date",
    ]

    DATE_COL = "notice_date"
    DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")


class KlineBuilder(BaseBuilder):
    """
    Builder for stock kline (historical) data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("date", "date"),
        ColumnSpec("open", "open", pl.Float64),
        ColumnSpec("close", "close", pl.Float64),
        ColumnSpec("high", "high", pl.Float64),
        ColumnSpec("low", "low", pl.Float64),
        ColumnSpec("volume", "volume", pl.Float64),
        ColumnSpec("amount", "amount", pl.Float64),
        ColumnSpec("amplitude", "amplitude", pl.Float64),
        ColumnSpec("change_pct", "change_pct", pl.Float64),
        ColumnSpec("change_amt", "change_amt", pl.Float64),
        ColumnSpec("turnover", "turnover", pl.Float64),
    ]

    OUTPUT_ORDER = [
        "date",
        "stock_code",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "change_pct",
        "change_amt",
        "turnover",
    ]

    DATE_COL = "date"
    DUPLICATE_COLS = ("stock_code", "date")


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class ReportConfig:
    """
    Configuration for quarterly financial reports.

    Attributes:
        report_name: Report name for API (e.g., "RPT_DMSK_FN_INCOME")
        parser_class: Parser class to use
        builder_class: Builder class to use
        url: API endpoint URL
    """

    report_name: str
    parser_class: type[BaseParser]
    builder_class: type[BaseBuilder]
    url: str = EAST_MONEY_API_URL


# =============================================================================
# Specialized Pipes
# =============================================================================


class ReportPipe:
    """
    Pipe for fetching quarterly financial reports.

    Handles the common pattern of:
    1. Building params from year/quarter
    2. Fetching initial page
    3. Fetching remaining pages concurrently
    4. Parsing and building DataFrame
    """

    def __init__(self, fetcher: EastMoneyFetch, config: ReportConfig):
        """
        Initialize ReportPipe.

        Args:
            fetcher: Fetcher instance
            config: Report configuration
        """
        self._fetcher = fetcher
        self._config = config

    def fetch(self, year: int, quarter: int) -> pl.DataFrame:
        """
        Fetch quarterly report data.

        Args:
            year: Report year (e.g., 2024)
            quarter: Report quarter (1, 2, 3, or 4)

        Returns:
            Polars DataFrame with report data
        """
        # Build date and params
        formatted_date = _build_quarterly_date(year, quarter)
        params = _build_quarterly_params(self._config.report_name, formatted_date)

        # Create parser and builder instances
        parser = self._config.parser_class()
        builder = self._config.builder_class()

        # Fetch initial page
        raw = self._fetcher.fetch_initial(self._config.url, params)

        # Parse and get total pages
        data = parser.parse(raw)
        if not data:
            return pl.DataFrame()

        total_pages = raw.get("result", {}).get("pages", 1)
        print(f"Total pages to fetch: {total_pages}")

        # Fetch remaining pages concurrently
        if total_pages > 1:
            pages_to_fetch = list(range(2, total_pages + 1))
            page_results = self._fetcher.fetch_pages_concurrent(
                self._config.url, params, pages_to_fetch
            )

            for page_raw in page_results:
                page_data = parser.parse(page_raw)
                data.extend(page_data)

        # Build DataFrame
        df = parser.clean(data)
        df = builder.normalize(df)
        return df


class KlinePipe:
    """
    Pipe for fetching stock historical data (kline).

    Handles the pattern of:
    1. Building params from symbol, period, dates
    2. Fetching data
    3. Parsing and building DataFrame
    """

    def __init__(self, fetcher: EastMoneyFetch):
        """
        Initialize KlinePipe.

        Args:
            fetcher: Fetcher instance
        """
        self._fetcher = fetcher

    def fetch(
        self,
        symbol: str = "000001",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        adjust: str = "",
    ) -> pl.DataFrame:
        """
        Fetch stock historical data (kline).

        Args:
            symbol: Stock code (e.g., "000001")
            period: "daily", "weekly", or "monthly"
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format
            adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

        Returns:
            Polars DataFrame with historical stock data
        """
        # Build params
        secid, params = _build_kline_params(symbol, period, start_date, end_date, adjust)

        # Fetch data
        raw = self._fetcher.fetch_initial(KLINE_API_URL, params)

        # Parse using KlineParser
        parser = KlineParser()
        data = parser.parse(raw)
        if not data:
            return pl.DataFrame()

        # Build DataFrame
        df = parser.clean(data)

        # Add stock_code column
        df = df.with_columns(pl.lit(symbol).alias("stock_code"))

        # Apply builder transformations
        builder = KlineBuilder()
        df = builder.normalize(df)

        return df


# =============================================================================
# Main Class (No Inner Fetcher Exposed)
# =============================================================================


class EastMoney:
    """
    Main interface for East Money data.

    Composes Fetcher, Parser, Builder internally.
    Fetcher is not exposed in the public interface.

    Usage:
        with EastMoney() as client:
            df = client.quarterly_income(2024, 1)
            print(df)
    """

    def __init__(
        self,
        delay_range: tuple[float, float] = (0.5, 1.5),
        max_retries: int = 3,
        max_workers: int = 3,
    ):
        """
        Initialize EastMoney client.

        Args:
            delay_range: (min, max) delay between requests in seconds
            max_retries: Number of retry attempts on failure
            max_workers: Maximum concurrent workers for page fetching
        """
        self._fetcher = EastMoneyFetch(
            delay_range=delay_range,
            max_retries=max_retries,
            max_workers=max_workers,
        )

        # Create report configurations
        self._income_config = ReportConfig(
            report_name="RPT_DMSK_FN_INCOME",
            parser_class=FundemantalParser,
            builder_class=IncomeBuilder,
        )
        self._balance_config = ReportConfig(
            report_name="RPT_DMSK_FN_BALANCE",
            parser_class=FundemantalParser,
            builder_class=BalanceSheetBuilder,
        )
        self._cashflow_config = ReportConfig(
            report_name="RPT_DMSK_FN_CASHFLOW",
            parser_class=FundemantalParser,
            builder_class=CashFlowBuilder,
        )

    def quarterly_income(self, year: int, quarter: int) -> pl.DataFrame:
        """
        Fetch quarterly income statement data.

        Args:
            year: Report year (e.g., 2024)
            quarter: Report quarter (1, 2, 3, or 4)

        Returns:
            Polars DataFrame with income statement data
        """
        pipe = ReportPipe(self._fetcher, self._income_config)
        return pipe.fetch(year, quarter)

    def quarterly_balance_sheet(self, year: int, quarter: int) -> pl.DataFrame:
        """
        Fetch quarterly balance sheet data.

        Args:
            year: Report year (e.g., 2024)
            quarter: Report quarter (1, 2, 3, or 4)

        Returns:
            Polars DataFrame with balance sheet data
        """
        pipe = ReportPipe(self._fetcher, self._balance_config)
        return pipe.fetch(year, quarter)

    def quarterly_cashflow(self, year: int, quarter: int) -> pl.DataFrame:
        """
        Fetch quarterly cash flow statement data.

        Args:
            year: Report year (e.g., 2024)
            quarter: Report quarter (1, 2, 3, or 4)

        Returns:
            Polars DataFrame with cash flow data
        """
        pipe = ReportPipe(self._fetcher, self._cashflow_config)
        return pipe.fetch(year, quarter)

    def stock_hist(
        self,
        symbol: str = "000001",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        adjust: str = "",
    ) -> pl.DataFrame:
        """
        Fetch stock historical data (kline).

        Args:
            symbol: Stock code (e.g., "000001")
            period: "daily", "weekly", or "monthly"
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format
            adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)

        Returns:
            Polars DataFrame with historical stock data
        """
        pipe = KlinePipe(self._fetcher)
        return pipe.fetch(symbol, period, start_date, end_date, adjust)

    def close(self) -> None:
        """Close the fetcher."""
        self._fetcher.close()

    def __enter__(self) -> "EastMoney":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Backward Compatibility
# =============================================================================


def stock_lrb_em(date: str = "20240331", **kwargs) -> pl.DataFrame:
    """
    East Money Income Statement (Backward Compatible).

    Args:
        date: Report date in format "YYYYMMDD"
        max_workers: Concurrent workers (default: 3)
        delay_range: (min, max) delay between requests (default: 0.5-1.5)

    Returns:
        Polars DataFrame (was pandas in original)
    """
    # Parse date
    year = int(date[:4])
    month = int(date[4:6])
    quarter = (month - 1) // 3 + 1

    # Extract kwargs
    max_workers = kwargs.get("max_workers", 3)
    delay_range = kwargs.get("delay_range", (0.5, 1.5))

    # Use context manager
    with EastMoney(
        delay_range=delay_range,
        max_workers=max_workers,
    ) as client:
        return client.quarterly_income(year, quarter)


def stock_lrb_em_batch(
    dates: list[str],
    max_workers: int = 3,
    delay_range: tuple[float, float] = (0.5, 1.5),
) -> dict[str, pl.DataFrame]:
    """
    Fetch multiple dates with adaptive rate limiting.

    Args:
        dates: List of date strings ["20240331", "20240630", ...]
        max_workers: Concurrent workers per request
        delay_range: (min, max) delay between pages

    Returns:
        Dictionary mapping date to DataFrame
    """
    results = {}

    for i, date in enumerate(dates):
        print(f"\nFetching {date} ({i + 1}/{len(dates)})...")
        try:
            # Add longer delay between different dates
            if i > 0:
                time.sleep(random.uniform(2, 4))

            with EastMoney(delay_range=delay_range, max_workers=max_workers) as client:
                year = int(date[:4])
                month = int(date[4:6])
                quarter = (month - 1) // 3 + 1
                results[date] = client.quarterly_income(year, quarter)
        except Exception as e:
            print(f"Failed to fetch {date}: {e}")
            results[date] = pl.DataFrame()

    return results


def stock_zcfz_em(date: str = "20240331", **kwargs) -> pl.DataFrame:
    """
    East Money Balance Sheet (Backward Compatible).

    Args:
        date: Report date in format "YYYYMMDD"
        max_workers: Concurrent workers (default: 3)
        delay_range: (min, max) delay between requests (default: 0.5-1.5)

    Returns:
        Polars DataFrame (was pandas in original)
    """
    # Parse date
    year = int(date[:4])
    month = int(date[4:6])
    quarter = (month - 1) // 3 + 1

    # Extract kwargs
    max_workers = kwargs.get("max_workers", 3)
    delay_range = kwargs.get("delay_range", (0.5, 1.5))

    # Use context manager
    with EastMoney(
        delay_range=delay_range,
        max_workers=max_workers,
    ) as client:
        return client.quarterly_balance_sheet(year, quarter)


def stock_xjll_em(date: str = "20240331", **kwargs) -> pl.DataFrame:
    """
    East Money Cash Flow Statement (Backward Compatible).

    Args:
        date: Report date in format "YYYYMMDD"
        max_workers: Concurrent workers (default: 3)
        delay_range: (min, max) delay between requests (default: 0.5-1.5)

    Returns:
        Polars DataFrame (was pandas in original)
    """
    # Parse date
    year = int(date[:4])
    month = int(date[4:6])
    quarter = (month - 1) // 3 + 1

    # Extract kwargs
    max_workers = kwargs.get("max_workers", 3)
    delay_range = kwargs.get("delay_range", (0.5, 1.5))

    # Use context manager
    with EastMoney(
        delay_range=delay_range,
        max_workers=max_workers,
    ) as client:
        return client.quarterly_cashflow(year, quarter)


def stock_zh_a_hist(
    symbol: str = "000001",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
    timeout: float | None = None,
    **kwargs
) -> pl.DataFrame:
    """
    East Money Stock Historical Data (Backward Compatible).

    Args:
        symbol: Stock code (e.g., "000001")
        period: "daily", "weekly", or "monthly"
        start_date: Start date in "YYYYMMDD" format
        end_date: End date in "YYYYMMDD" format
        adjust: "" (no adjustment), "qfq" (forward), "hfq" (backward)
        timeout: Timeout value (not used, kept for compatibility)
        **kwargs: Additional arguments (max_workers, delay_range)

    Returns:
        Polars DataFrame with historical stock data
    """
    # Extract kwargs
    max_workers = kwargs.get("max_workers", 3)
    delay_range = kwargs.get("delay_range", (0.5, 1.5))

    # Use context manager
    with EastMoney(
        delay_range=delay_range,
        max_workers=max_workers,
    ) as client:
        return client.stock_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
