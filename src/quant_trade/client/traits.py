"""Generic data pipe components for Quant-Trade.

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

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import polars as pl
import tqdm
import tqdm.asyncio
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quant_trade.config.logger import log
from quant_trade.transform import normalize_date_column

# =============================================================================
# Protocols - Duck Typing Interfaces
# =============================================================================


class Fetcher(Protocol):
    """Protocol for fetching - duck typing interface.

    Implementations must provide:
    - fetch_initial(url, params) -> dict
    - fetch_page(url, params, page) -> dict
    - close() -> None
    """

    def fetch_initial(self, url: str, params: dict) -> dict: ...
    def fetch_page(self, url: str, params: dict, page: int) -> dict: ...
    def close(self) -> None: ...


class Parser(Protocol):
    """Protocol for parsing."""

    def parse(self, raw: Any) -> Any: ...
    def clean(self, data: Any) -> pl.DataFrame: ...


class Builder(Protocol):
    """Protocol for building output.

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
#
@dataclass
class _AdaptiveController:
    """EWMA-based controller to manage backpressure and circuit breaking."""

    alpha: float = 0.2
    failure_threshold: float = 0.7  # Trip circuit breaker if pressure > 0.7

    def __post_init__(self):
        self.value = 0.0
        self._lock = asyncio.Lock()

    async def record_success(self):
        async with self._lock:
            self.value = (1 - self.alpha) * self.value

    async def record_failure(self, weight: float = 1.0):
        async with self._lock:
            # Standard EWMA formula
            self.value = min(1.0, (1 - self.alpha) * self.value + self.alpha * weight)

    @property
    def is_tripped(self) -> bool:
        return self.value > self.failure_threshold


class BaseFetcher:
    """Advanced Base Fetcher using httpx + tenacity.

    Caveat: **Do not** use **multi-thread** **in** session but outside of it
    due to single thread constraint of event loop.
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
        base_delay: float = 0.5,
        max_retries: int = 3,
        concurrency: int = 5,
        timeout: float = 10.0,
        # proxies: dict[str,str]|None = None
    ):
        self.concurrency = concurrency
        self.base_delay = base_delay
        self.max_retries = max_retries

        # Adaptive components
        self.controller = _AdaptiveController()
        self.semaphore = asyncio.Semaphore(concurrency)

        # Client configuration
        self.client = httpx.AsyncClient(
            http2=True,  # Critical for modern sites
            timeout=httpx.Timeout(timeout),
            # proxies=proxies,
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=concurrency, max_connections=concurrency * 2
            ),
        )

    def _get_headers(self) -> dict:
        profile = random.choice(self._HEADER_PROFILES)
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/json, text/plain, */*",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Cache-Control": "no-cache",
            "DNT": "1",
            **profile,
        }

    async def _handle_backoff(self):
        """Dynamic delay based on EWMA pressure."""
        # Base jittered delay + exponential pressure penalty
        pressure_penalty = self.controller.value * 5.0
        delay = random.uniform(self.base_delay, self.base_delay * 2) + pressure_penalty

        if self.controller.is_tripped:
            log.warning(
                f"Circuit breaker active (Pressure: {self.controller.value:.2f}). Cooling down..."
            )
            delay += 10.0

        await asyncio.sleep(delay)

    async def _fetch_once(self, url: str, params: dict) -> Any:
        raise NotImplementedError

    async def fetch_once(self, url: str, params: dict) -> dict:
        """Generic low-level async request with rate limiting & retries."""
        async with self.semaphore:
            await self._handle_backoff()

            @retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type(
                    (httpx.RequestError, httpx.HTTPStatusError)
                ),
                reraise=True,
            )
            async def _do():
                return await self._fetch_once(url, params)

            try:
                res = await _do()
                await self.controller.record_success()
                return res
            except httpx.HTTPStatusError as e:
                # 429 and 403 are critical signals in scraping
                weight = 1.0 if e.response.status_code in (429, 403) else 0.5
                await self.controller.record_failure(weight=weight)
                log.error(
                    f"Blocked ({e.response.status_code}). Pressure: {self.controller.value:.2f}"
                )
                return {}
            except Exception as e:
                await self.controller.record_failure(
                    weight=1.0 if "429" in str(e) else 0.5
                )
                return {}

    async def _fetch_page(self, url: str, params: dict, page: int) -> Any:
        raise NotImplementedError

    async def fetch_pages_concurrent(
        self, url: str, params: dict, pages: list[int]
    ) -> list[dict]:
        """Public API to fetch multiple pages efficiently."""
        tasks = [self._fetch_page(url, params, p) for p in pages]

        results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Scraping Quant Data")
        return [r for r in results if r is not None]

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


@dataclass
class ColumnSpec:
    """Specification for a column transformation.

    Attributes:
        source: Original column name from API
        target: Target column name in output
        dtype: Target Polars dtype (optional)
    """

    source: str
    target: str
    dtype: type[pl.DataType] | None = None


class BaseBuilder:
    """Base Builder - generic DataFrame transformation.

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
    def column_map(self) -> dict[str, str]:
        """Generate column mapping from specs."""
        return {spec.source: spec.target for spec in self.COLUMN_SPECS}

    @property
    def numeric_cols(self) -> list[str]:
        """Get numeric column names from specs."""
        return [
            spec.target
            for spec in self.COLUMN_SPECS
            if spec.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

    def rename(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply column renaming based on COLUMN_SPECS.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with renamed columns
        """
        log.debug(f"Renaming columns using mapping: \n {self.column_map}")
        rename_map = {k: v for k, v in self.column_map.items() if k in df.columns}
        return df.rename(rename_map)

    def convert_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert column types based on COLUMN_SPECS.

        - Numeric columns: cast to Float64
        - Date columns: parse as date strings

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with converted types
        """
        log.debug(
            f"Converting types: numeric columns: \n {self.numeric_cols} \n date column: \n {self.DATE_COL}"
        )
        result = df
        for col in self.numeric_cols:
            if col in result.columns:
                result = result.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        if self.DATE_COL and self.DATE_COL in result.columns:
            result = normalize_date_column(result, self.DATE_COL).sort(self.DATE_COL)
        return result

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Reorder columns and add sequence number.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with columns in OUTPUT_ORDER and seq column
        """
        # Select only columns that exist in OUTPUT_ORDER
        # available = [c for c in self.OUTPUT_ORDER if c in df.columns]

        # # Add sequence column at position 0
        # if df.height > 0:
        #     result = df.select(available).with_columns(
        #         pl.arange(1, df.height + 1).alias("seq")
        #     )
        # else:
        #     result = df.select(available)

        # Reorder to match OUTPUT_ORDER
        ordered = [c for c in self.OUTPUT_ORDER if c in df.columns]
        return df.select(ordered)

    def deduplicate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate rows based on DUPLICATE_COLS.

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
        """Apply full normalization: rename, convert types, deduplicate, reorder.

        Args:
            df: Input DataFrame

        Returns:
            Normalized DataFrame
        """
        if df.columns:
            log.debug(f"Normalizing df columns: {df.columns}")
        result = self.rename(df)
        result = self.extract(result)
        result = self.deduplicate(result)
        result = self.convert_types(result)
        log.debug(f"Normalized df columns: {result.columns}")
        return result


# =============================================================================
# Data Pipe - Orchestrates Fetcher -> Parser -> Builder
# =============================================================================


# class DataPipe:
#     """Generic data pipe that orchestrates Fetcher -> Parser -> Builder.

#     This is the core abstraction that enables reusable data fetching patterns.

#     Features:
#     - Full pipeline orchestration: fetch -> parse -> clean -> transform
#     - Concurrent page fetching support
#     - Context manager support

#     Usage:
#         pipe = DataPipe(
#             fetcher=MyFetcher(),
#             parser=MyParser(),
#             builder=MyBuilder(),
#         )

#         with pipe as p:
#             df = p.run(url, params)

#         # Or without context manager:
#         df = pipe.run(url, params)
#         pipe.fetcher.close()
#     """

#     def __init__(
#         self,
#         fetcher: BaseFetcher,
#         parser: BaseParser,
#         builder: Builder,
#     ):
#         """Initialize DataPipe.

#         Args:
#             fetcher: BaseFetch implementation (provides fetch_pages_concurrent)
#             parser: BaseParser implementation (provides get_total_pages)
#             builder: Builder implementation
#         """
#         self.fetcher = fetcher
#         self.parser = parser
#         self.builder = builder

#     def run(
#         self,
#         url: str,
#         params: dict,
#         concurrent_pages: bool = True,
#     ) -> pl.DataFrame:
#         """Execute the full data pipeline.

#         Args:
#             url: API endpoint URL
#             params: Query parameters
#             concurrent_pages: Whether to fetch pages concurrently

#         Returns:
#             Transformed Polars DataFrame
#         """
#         # Fetch initial page
#         raw = self.fetcher.fetch_initial(url, params)

#         # Parse and get total pages
#         data = self.parser.parse(raw)
#         if not data:
#             return pl.DataFrame()

#         total_pages = self.parser.get_total_pages(raw)

#         # Fetch remaining pages concurrently
#         if concurrent_pages and total_pages > 1:
#             pages_to_fetch = list(range(2, total_pages + 1))
#             page_results = self.fetcher.fetch_pages_concurrent(
#                 url, params, pages_to_fetch
#             )

#             for page_raw in page_results:
#                 page_data = self.parser.parse(page_raw)
#                 data.extend(page_data)

#         # Build DataFrame
#         df = self.parser.clean(data)

#         # Apply builder transformations
#         df = self.builder.rename(df)
#         df = self.builder.convert_types(df)
#         df = self.builder.reorder(df)
#         return df

#     def __enter__(self) -> "DataPipe":
#         return self

#     def __exit__(self, *args) -> None:
#         self.fetcher.close()


# =============================================================================
# Utility Functions
# =============================================================================


# def build_data_path(data_path: tuple[str, ...]) -> Callable[[dict], Any]:
#     """Create a function to extract data from a nested dictionary.

#     Args:
#         data_path: Tuple of keys to traverse

#     Returns:
#         Function that extracts data from a dict

#     Example:
#         extract_data = build_data_path(("result", "data"))
#         data = extract_data(response)  # Returns response["result"]["data"]
#     """

#     def extract(data: dict) -> Any:
#         result = data
#         for key in data_path:
#             if isinstance(result, dict):
#                 result = result.get(key, {})
#             else:
#                 return None
#         return result if result else None

#     return extract


# def create_retry_session(
#     max_retries: int = 3,
#     backoff_factor: float = 1,
#     pool_connections: int = 10,
#     pool_maxsize: int = 20,
# ) -> requests.Session:
#     """Create a requests Session with retry strategy.

#     Args:
#         max_retries: Maximum retry attempts
#         backoff_factor: Backoff multiplier between retries
#         pool_connections: Number of pool connections
#         pool_maxsize: Maximum pool size

#     Returns:
#         Configured requests Session
#     """
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=max_retries,
#         backoff_factor=backoff_factor,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=["GET"],
#     )
#     adapter = HTTPAdapter(
#         pool_connections=pool_connections,
#         pool_maxsize=pool_maxsize,
#         max_retries=retry_strategy,
#     )
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)
#     return session
