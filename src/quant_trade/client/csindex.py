import asyncio
from io import BytesIO

import polars as pl
from pyparsing import Any

from quant_trade.config.logger import log

from .traits import BaseBuilder, BaseFetcher, ColumnSpec


def csi_index_url(symbol: str) -> str:
    return (
        f"https://oss-ch.csindex.com.cn/static/"
        f"html/csindex/public/uploads/file/autofile/cons/{symbol}cons.xls"
    )


class CSIndexFetcher(BaseFetcher):
    """CSIndex-specific Fetcher - extends BaseFetch."""

    # User agents for rotation
    USER_AGENTS: list[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/2010",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a private event loop for this instance to handle the async bridge
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError as e:
            log.error(f"Async runtime error: {e}")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def run(self, future) -> Any:
        return self._loop.run_until_complete(future)

    async def _fetch_once(self, url: str, params: dict) -> bytes:
        """Fetch initial page to get total pages."""
        response = await self.client.get(
            url,
            params=params,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.content

    def sclose(self) -> None:
        """Gracefully close the underlying async connection pool."""
        if self.client:
            self._loop.run_until_complete(self.client.aclose())


class CSIndexParser:
    def parse(self, flow: bytes) -> pl.DataFrame:
        df = pl.read_excel(BytesIO(flow), has_header=True)
        return df


class CSIndexBuilder(BaseBuilder):
    """Builder for LRB (Income Statement) data.
    Uses ColumnSpec pattern for flexible column definitions.
    """

    COLUMN_SPECS = [
        ColumnSpec("日期Date", "date"),
        ColumnSpec("成份券代码Constituent Code", "ts_code"),
        ColumnSpec("成份券名称Constituent Name", "name"),
    ]

    OUTPUT_ORDER = ["date", "ts_code", "name"]

    DATE_COL = "date"


class CSIndexPipe:
    def __init__(self, fetcher: CSIndexFetcher):
        self._fetcher = fetcher

    def fetch(self, symbol: str) -> pl.DataFrame:
        fetcher = self._fetcher
        raw = fetcher.run(fetcher.fetch_once(csi_index_url(symbol), params={}))
        parser = CSIndexParser()
        builder = CSIndexBuilder()
        df = parser.parse(raw)
        df = builder.normalize(df)
        return df


class CSIndex:
    """Main interface for CSIndex data.

    Composes Fetcher, Parser, Builder internally.
    Fetcher is not exposed in the public interface.

    Usage:
        with EastMoney() as client:
            df = client.quarterly_income(2024, 1)
            print(df)
    """

    def __init__(
        self,
        max_retries: int = 3,
    ):
        """Initialize EastMoney client.

        Args:
            max_retries: Number of retry attempts on failure
        """
        self._fetcher = CSIndexFetcher(max_retries=max_retries)

    def index_cons(self, symbol: str) -> pl.DataFrame:
        log.info(f"Fetching {symbol} conponents")
        pipe = CSIndexPipe(self._fetcher)
        return pipe.fetch(symbol)

    def close(self) -> None:
        """Close the fetcher."""
        self._fetcher.sclose()

    def __enter__(self) -> "CSIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()
