"""
Generic data pipe components for Quant-Trade.

Extracts reusable patterns from data providers into composable traits:
- Fetcher: HTTP fetching protocol
- Parser: JSON parsing protocol
- Builder: DataFrame transformation protocol
- DataPipe: Orchestrates the full data pipeline

Usage:
    from quant_trade.client.trait import Fetcher, Parser, Builder, DataPipe

    # Create custom implementations
    class MyFetcher(BaseFetch):
        def fetch_initial(self, url: str, params: dict) -> dict:
            ...
        def fetch_page(self, url: str, params: dict, page: int) -> dict:
            ...

    class MyParser(BaseParser):
        DATA_PATH = ("result", "data")

    # Use DataPipe to orchestrate
    pipe = DataPipe(fetcher=MyFetcher(), parser=MyParser(), builder=MyBuilder())
    df = pipe.run(url, params)
"""

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import polars as pl
import requests
import tqdm
from requests.adapters import HTTPAdapter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from urllib3.util.retry import Retry

from quant_trade.config.logger import log
from quant_trade.transform import normalize_date_column

# =============================================================================
# Protocols - Duck Typing Interfaces
# =============================================================================


class Fetcher(Protocol):
    """
    Protocol for fetching - duck typing interface.

    Implementations must provide:
    - fetch_initial(url, params) -> dict
    - fetch_page(url, params, page) -> dict
    - close() -> None
    """

    def fetch_initial(self, url: str, params: dict) -> dict: ...
    def fetch_page(self, url: str, params: dict, page: int) -> dict: ...
    def close(self) -> None: ...


class Parser(Protocol):
    """
    Protocol for parsing.

    Implementations must provide:
    - parse(raw) -> list[dict]
    - clean(data) -> pl.DataFrame
    """

    def parse(self, raw: dict) -> list[dict]: ...
    def clean(self, data: list[dict]) -> pl.DataFrame: ...


class Builder(Protocol):
    """
    Protocol for building output.

    Implementations must provide:
    - rename(df) -> pl.DataFrame
    - convert_types(df) -> pl.DataFrame
    - reorder(df) -> pl.DataFrame
    """

    def rename(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def convert_types(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def reorder(self, df: pl.DataFrame) -> pl.DataFrame: ...


# =============================================================================
# Base Implementations
# =============================================================================


class _RateLimiter:
    """
    Thread-safe token-bucket-like rate limiter with jitter.
    """

    def __init__(self, min_delay: float, max_delay: float):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def wait(self, penalty: float = 0.0):
        with self._lock:
            now = time.monotonic()
            base_delay = random.uniform(self.min_delay, self.max_delay)
            delay = base_delay + penalty

            if now < self._next_allowed:
                sleep_time = self._next_allowed - now + delay
            else:
                sleep_time = delay

            self._next_allowed = max(self._next_allowed, now) + delay

        if sleep_time > 0:
            time.sleep(sleep_time)


class BaseFetcher:
    """
    Advanced Base Fetcher using httpx + tenacity.
    """

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
    ]

    _HEADER_PROFILES = [
        {"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        {"Accept-Language": "en-US,en;q=0.9"},
        {"Accept-Language": "en-GB,en;q=0.9"},
    ]

    def __init__(
        self,
        delay_range: tuple[float, float] = (0.5, 1.5),
        max_retries: int = 3,
        max_workers: int = 3,
        timeout: float = 10.0,
    ):
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.max_workers = max_workers

        self._rate_limiter = _RateLimiter(*delay_range)

        self._penalty = 0.0
        self._penalty_lock = threading.Lock()
        self._recent_errors = deque(maxlen=20)

        self._client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(timeout),
            headers=self._get_headers(),
        )

    # ----------------------------
    # Headers
    # ----------------------------

    def _get_headers(self) -> dict:
        profile = random.choice(self._HEADER_PROFILES)
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json, text/plain, */*",
            "Connection": "keep-alive",
            **profile,
        }

    # ----------------------------
    # Adaptive penalty
    # ----------------------------

    def _apply_rate_limit(self):
        self._rate_limiter.wait(self._penalty)

    def _record_result(self, response: httpx.Response | None, exc: Exception | None):
        with self._penalty_lock:
            if exc is not None:
                self._penalty = min(self._penalty + 0.3, 10.0)
                return

            if response is None:
                return

            if response.status_code in (403, 429):
                self._penalty = min(self._penalty + 1.0, 10.0)
            else:
                self._penalty = max(self._penalty - 0.1, 0.0)

    # ----------------------------
    # Retry wrapper
    # ----------------------------

    def _retry_decorator(self):
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(min=0.5, max=5),
            retry=retry_if_exception_type(
                (httpx.TransportError, httpx.HTTPStatusError)
            ),
            reraise=True,
        )

    # ----------------------------
    # Public API (unchanged)
    # ----------------------------

    def fetch_initial(self, url: str, params: dict) -> dict:
        raise NotImplementedError

    def fetch_page(self, url: str, params: dict, page: int) -> dict:
        raise NotImplementedError

    def fetch_pages_concurrent(
        self,
        url: str,
        params: dict,
        pages: list[int],
    ) -> list[dict]:
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self._safe_fetch_page, url, params, page): page
                for page in pages
            }

            n = len(pages)
            for future in tqdm.tqdm(
                as_completed(future_to_page),
                total=n,
                desc="Fetching data",
                position=0,
            ):
                page = future_to_page[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"[WARN] page={page} failed: {e}")

        return results

    # ----------------------------
    # Internal guarded fetch
    # ----------------------------

    def _safe_fetch_page(self, url: str, params: dict, page: int):
        self._apply_rate_limit()

        retryable = self._retry_decorator()

        @retryable
        def _call():
            return self.fetch_page(url, params, page)

        try:
            result = _call()
            self._record_result(None, None)
            return result
        except httpx.HTTPStatusError as e:
            self._record_result(e.response, e)
            raise
        except Exception as e:
            self._record_result(None, e)
            raise

    # ----------------------------
    # Lifecycle
    # ----------------------------

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BaseParser:
    """
    Base Parser - converts raw JSON to structured data.

    Provides:
    - Configurable data path traversal via DATA_PATH
    - Total pages extraction via get_total_pages()
    - Basic DataFrame cleaning

    Usage:
        class MyParser(BaseParser):
            DATA_PATH = ("result", "data")

            def parse(self, raw: dict) -> list[dict]:
                # Custom parsing logic
                ...
    """

    # Keys to traverse to get data list, e.g., ("result", "data")
    # Override in subclass for different API structures
    DATA_PATH: tuple[str, ...] = ("result", "data")

    # Key for total pages in response
    PAGES_KEY: str = "pages"

    def parse(self, raw: dict) -> list[dict]:
        """
        Parse response and extract data list.

        Args:
            raw: Raw JSON response dictionary

        Returns:
            List of data dictionaries
        """
        if not raw.get("result"):
            return []

        # Navigate to data using DATA_PATH
        data = raw
        for key in self.DATA_PATH:
            data = data.get(key, {})
            if data is None:
                return []

        return data if isinstance(data, list) else []

    def get_total_pages(self, raw: dict) -> int:
        """
        Get total pages from response.

        Args:
            raw: Raw JSON response dictionary

        Returns:
            Total number of pages (default: 1)
        """
        return raw.get("result", {}).get(self.PAGES_KEY, 1)

    def clean(self, data: list[dict]) -> pl.DataFrame:
        """
        Clean and convert to Polars DataFrame.

        Args:
            data: List of data dictionaries

        Returns:
            Polars DataFrame
        """
        if not data:
            return pl.DataFrame()

        # Use infer_schema_length=None to scan all rows for proper type inference
        # This prevents issues with large numeric values causing overflow
        return pl.DataFrame(data, infer_schema_length=None)


@dataclass
class ColumnSpec:
    """
    Specification for a column transformation.

    Attributes:
        source: Original column name from API
        target: Target column name in output
        dtype: Target Polars dtype (optional)
    """

    source: str
    target: str
    dtype: type[pl.DataType] | None = None


class BaseBuilder:
    """
    Base Builder - generic DataFrame transformation.

    Provides:
    - Column renaming via COLUMN_SPECS
    - Type conversion for numeric and date columns
    - Column reordering with automatic sequence numbering

    Usage:
        class MyBuilder(BaseBuilder):
            COLUMN_SPECS = [
                ColumnSpec("SECURITY_CODE", "stock_code"),
                ColumnSpec("NOTICE_DATE", "notice_date", pl.Date),
                ColumnSpec("NETPROFIT", "net_profit", pl.Float64),
            ]

            OUTPUT_ORDER = ["seq", "stock_code", "stock_name", "notice_date"]
            DATE_COL = "notice_date"
            DUPLICATE_COLS = ("SECURITY_CODE", "NOTICE_DATE")
    """

    # Column specifications - override in subclass
    COLUMN_SPECS: list[ColumnSpec] = field(default_factory=list)

    # Final column order - override in subclass
    OUTPUT_ORDER: list[str] = field(default_factory=list)

    # Name of date column if exists - override in subclass
    DATE_COL: str | None = None

    # Columns for deduplication - override in subclass
    DUPLICATE_COLS: tuple[str, ...] | None = None

    @property
    def COLUMN_MAPPING(self) -> dict[str, str]:
        """Generate column mapping from specs."""
        return {spec.source: spec.target for spec in self.COLUMN_SPECS}

    @property
    def NUMERIC_COLS(self) -> list[str]:
        """Get numeric column names from specs."""
        return [
            spec.target
            for spec in self.COLUMN_SPECS
            if spec.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

    def rename(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply column renaming based on COLUMN_SPECS.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with renamed columns
        """
        log.debug(f"Renaming columns using mapping: {self.COLUMN_MAPPING}")
        rename_map = {k: v for k, v in self.COLUMN_MAPPING.items() if k in df.columns}
        return df.rename(rename_map)

    def convert_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert column types based on COLUMN_SPECS.

        - Numeric columns: cast to Float64
        - Date columns: parse as date strings

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with converted types
        """
        log.debug(
            f"Converting types for columns: {self.NUMERIC_COLS} and date column: {self.DATE_COL}"
        )
        result = df
        for col in self.NUMERIC_COLS:
            if col in result.columns:
                result = result.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        if self.DATE_COL and self.DATE_COL in result.columns:
            result = normalize_date_column(result, self.DATE_COL)
        return result

    def reorder(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reorder columns and add sequence number.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with columns in OUTPUT_ORDER and seq column
        """
        # Select only columns that exist in OUTPUT_ORDER
        available = [c for c in self.OUTPUT_ORDER if c in df.columns]

        # Add sequence column at position 0
        if df.height > 0:
            result = df.select(available).with_columns(
                pl.arange(1, df.height + 1).alias("seq")
            )
        else:
            result = df.select(available)

        # Reorder to match OUTPUT_ORDER
        ordered = [c for c in self.OUTPUT_ORDER if c in result.columns]
        return result.select(ordered)

    def deduplicate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove duplicate rows based on DUPLICATE_COLS.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        if self.DUPLICATE_COLS is None:
            return df

        cols = [c for c in self.DUPLICATE_COLS if c in df.columns]
        if cols:
            return df.unique(subset=cols)
        return df

    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply full normalization: rename, convert types, deduplicate, reorder.

        Args:
            df: Input DataFrame

        Returns:
            Normalized DataFrame
        """
        log.debug(f"Normalizing df columns: {df.columns}")
        result = self.rename(df)
        result = self.convert_types(result)
        result = self.deduplicate(result)
        result = self.reorder(result)
        log.debug(f"Normalized df columns: {result.columns}")
        return result


# =============================================================================
# Data Pipe - Orchestrates Fetcher -> Parser -> Builder
# =============================================================================


class DataPipe:
    """
    Generic data pipe that orchestrates Fetcher -> Parser -> Builder.

    This is the core abstraction that enables reusable data fetching patterns.

    Features:
    - Full pipeline orchestration: fetch -> parse -> clean -> transform
    - Concurrent page fetching support
    - Context manager support

    Usage:
        pipe = DataPipe(
            fetcher=MyFetcher(),
            parser=MyParser(),
            builder=MyBuilder(),
        )

        with pipe as p:
            df = p.run(url, params)

        # Or without context manager:
        df = pipe.run(url, params)
        pipe.fetcher.close()
    """

    def __init__(
        self,
        fetcher: BaseFetcher,
        parser: BaseParser,
        builder: Builder,
    ):
        """
        Initialize DataPipe.

        Args:
            fetcher: BaseFetch implementation (provides fetch_pages_concurrent)
            parser: BaseParser implementation (provides get_total_pages)
            builder: Builder implementation
        """
        self.fetcher = fetcher
        self.parser = parser
        self.builder = builder

    def run(
        self,
        url: str,
        params: dict,
        concurrent_pages: bool = True,
    ) -> pl.DataFrame:
        """
        Execute the full data pipeline.

        Args:
            url: API endpoint URL
            params: Query parameters
            concurrent_pages: Whether to fetch pages concurrently

        Returns:
            Transformed Polars DataFrame
        """
        # Fetch initial page
        raw = self.fetcher.fetch_initial(url, params)

        # Parse and get total pages
        data = self.parser.parse(raw)
        if not data:
            return pl.DataFrame()

        total_pages = self.parser.get_total_pages(raw)

        # Fetch remaining pages concurrently
        if concurrent_pages and total_pages > 1:
            pages_to_fetch = list(range(2, total_pages + 1))
            page_results = self.fetcher.fetch_pages_concurrent(
                url, params, pages_to_fetch
            )

            for page_raw in page_results:
                page_data = self.parser.parse(page_raw)
                data.extend(page_data)

        # Build DataFrame
        df = self.parser.clean(data)

        # Apply builder transformations
        df = self.builder.rename(df)
        df = self.builder.convert_types(df)
        df = self.builder.reorder(df)
        return df

    def __enter__(self) -> "DataPipe":
        return self

    def __exit__(self, *args) -> None:
        self.fetcher.close()


# =============================================================================
# Utility Functions
# =============================================================================


def build_data_path(data_path: tuple[str, ...]) -> Callable[[dict], Any]:
    """
    Create a function to extract data from a nested dictionary.

    Args:
        data_path: Tuple of keys to traverse

    Returns:
        Function that extracts data from a dict

    Example:
        extract_data = build_data_path(("result", "data"))
        data = extract_data(response)  # Returns response["result"]["data"]
    """

    def extract(data: dict) -> Any:
        result = data
        for key in data_path:
            if isinstance(result, dict):
                result = result.get(key, {})
            else:
                return None
        return result if result else None

    return extract


def create_retry_session(
    max_retries: int = 3,
    backoff_factor: float = 1,
    pool_connections: int = 10,
    pool_maxsize: int = 20,
) -> requests.Session:
    """
    Create a requests Session with retry strategy.

    Args:
        max_retries: Maximum retry attempts
        backoff_factor: Backoff multiplier between retries
        pool_connections: Number of pool connections
        pool_maxsize: Maximum pool size

    Returns:
        Configured requests Session
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry_strategy,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
