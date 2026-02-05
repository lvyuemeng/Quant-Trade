import datetime
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import (
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

import polars as pl
from pyparsing import Callable

import quant_trade.provider.akshare as ak
import quant_trade.provider.baostock as bs
from quant_trade.config.arctic import ArcticAdapter, ArcticDB, Lib
from quant_trade.config.logger import log
from quant_trade.feature.process import MarginShort, Northbound, Shibor
from quant_trade.provider.traits import Source

from .process import (
    Behavioral,
    Fundamental,
    SectorGroup,
)
from .query import BookKey, MacroBook, QuarterBook, RecordBook, TickerBook


@runtime_checkable
class BookFetcher[B: BookKey](Protocol):
    """Protocol for classes that can fetch data by BookKey."""

    def _fetch_raw(self, book: B) -> pl.DataFrame: ...
    def _process(self, book: B, raw: pl.DataFrame) -> pl.DataFrame:
        return raw


@runtime_checkable
class BatchBookFetcher[B: BookKey](BookFetcher[B], Protocol):
    """Protocol for classes that can fetch data by BookKey in batch."""

    def _fetch_batch_raw(self, books: Sequence[B]) -> list[pl.DataFrame]: ...


class AssetLib:
    """Base class for ArcticDB-backed asset data."""

    REGION: str
    SYMBOL: str

    def __init__(self, db: ArcticDB):
        self._lib = db.get_lib(self.lib_name())

    @classmethod
    def lib_name(cls) -> str:
        return f"{cls.REGION}_{cls.SYMBOL}"

    @property
    def raw(self) -> Lib:
        """Direct access to ArcticDB library (advanced use only)."""
        return self._lib

    def _try_read(self: Self, book: BookKey) -> pl.DataFrame | None:
        key = book.to_key()
        try:
            df = ArcticAdapter.from_read(self._lib.read(key))
            return df if not df.is_empty() else pl.DataFrame()
        except Exception as e:
            log.warning(f"Database read failed for {key}: {e}")
            return pl.DataFrame()

    def _try_write(self: Self, book: BookKey, df: pl.DataFrame) -> None:
        key = book.to_key()
        try:
            self._lib.write(key, ArcticAdapter.to_write(df))
        except Exception as e:
            log.error(f"Database write failed for {key}: {e}")


class BookLib[B: BookKey](BookFetcher[B], AssetLib):
    """Mixin for ArcticDB-backed data by BookKey."""

    def fresh(self: Self, book: B) -> None:
        log.info(f"Fetching & processing data for {book.to_key()}")
        raw = self._fetch_raw(book)
        if raw.is_empty():
            log.warning(f"No raw data for {book.to_key()}")
            return
        feat = self._process(book, raw)
        self._try_write(book, feat)

    def read(self: Self, book: B, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._lib.has_symbol(book.to_key()):
            self.fresh(book)
        if self._lib.has_symbol(book.to_key()):
            return ArcticAdapter.from_read(self._lib.read(book.to_key()))
        raise KeyError(f"Data for {book.to_key()} not found in {self.lib_name()}")


class BatchBookLib[B: BookKey](BatchBookFetcher[B], AssetLib):
    def batch_fresh(self: Self, books: Sequence[B]) -> None:
        batch = self._fetch_batch_raw(books)
        for book, df in zip(books, batch):
            if df.is_empty():
                log.warning(f"No market data for {book}")
                continue
            feat = self._process(book, df)
            self._try_write(book, feat)

    def batch_read(
        self: Self, books: Iterable[B], *, fresh: bool = False
    ) -> list[pl.DataFrame]:
        books_list = list(books)

        data: dict[B, pl.DataFrame] = {}
        to_fetch: list[B] = []

        # Phase 1: read cache
        if not fresh:
            for book in books_list:
                df = self._try_read(book)
                if df is not None and not df.is_empty():
                    data[book] = df
                else:
                    to_fetch.append(book)
        else:
            to_fetch.extend(books_list)

        # Phase 2: batch fetch missing
        if to_fetch:
            for book, raw in zip(to_fetch, self._fetch_batch_raw(to_fetch)):
                if raw.is_empty():
                    continue
                df = self._process(book, raw)
                data[book] = df
                self._try_write(book, df)

        # Phase 3: preserve order
        return [
            df
            for book in books_list
            if (df := data.get(book)) is not None and not df.is_empty()
        ]


class CNStockPool(BookLib[RecordBook]):
    REGION = "CN"
    SYMBOL = "stock"

    type Book = Literal["stock_code", "industry_code"]
    type Universe = Literal["csi500"]

    @staticmethod
    def index_universe(universe: Universe, date: datetime.date) -> RecordBook:
        return RecordBook(f"{universe}", date)

    @staticmethod
    def _fetch_stock_code() -> pl.DataFrame:
        return ak.AkShareUniverse().stock_whole()

    @staticmethod
    def _fetch_industry_code() -> pl.DataFrame:
        return ak.SWIndustryCls().stock_l1_industry_cls()

    @staticmethod
    def _fetch_csi500_code() -> pl.DataFrame:
        return bs.BaoMacro().csi500_cons()

    def _fetch_raw(self, book: RecordBook) -> pl.DataFrame:
        match book:
            case _ if book.universe == "stock_code":
                return self._fetch_stock_code()
            case _ if book.universe == "industry_code":
                return self._fetch_industry_code()
            case _ if book.universe.startswith("csi500"):
                return self._fetch_csi500_code()
        raise ValueError(f"Unknown book fetch: {book}")

    def read_codes(self, book: Book, fresh: bool = False) -> pl.DataFrame:
        return self.read(RecordBook(book), fresh=fresh)

    def read_pool(
        self,
        universe: Universe,
        date: datetime.date,
        industry_cls: bool = False,
        fresh: bool = False,
    ) -> pl.DataFrame:
        book = self.index_universe(universe, date)
        df = self.read(book, fresh=fresh)
        if industry_cls:
            ind = self.read_codes("industry_code", fresh=fresh)
            df = df.join(ind, on="ts_code", how="inner")
            return df
        return df


class CNMarket(BookLib[TickerBook], BatchBookLib[TickerBook]):
    REGION = "CN"
    SYMBOL = "market"

    def __init__(self, db: ArcticDB, source: Source = "akshare"):
        super().__init__(db)
        self._source: Source = source

    def _fetch_raw(self, book: TickerBook) -> pl.DataFrame:
        ts_code = book.ts_code
        match self._source:
            case "akshare":
                return ak.AkShareMicro().market_ohlcv(ts_code, "daily")
            case "baostock":
                return bs.BaoMicro().market_ohlcv(ts_code, "daily")

    def _fetch_batch_raw(self, books: Iterable[TickerBook]) -> list[pl.DataFrame]:
        ts_codes = [book.ts_code for book in books]
        match self._source:
            case "akshare":
                return ak.AkShareMicro().batch_market_ohlcv(ts_codes, "daily")
            case "baostock":
                return bs.BaoMicro().batch_market_ohlcv(ts_codes, "daily")

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return Behavioral().metrics(df)

    @staticmethod
    def _normalize(df: pl.DataFrame) -> pl.DataFrame:
        return df.sort("date").unique(subset=["date"], keep="last")

    def _process(self, book: TickerBook, raw: pl.DataFrame) -> pl.DataFrame:
        feat = self._extract_features(raw)
        feat = self._normalize(feat)
        return feat

    def stack_read(
        self, books: Iterable[TickerBook], fresh: bool = False
    ) -> pl.DataFrame:
        frames = self.batch_read(books=books, fresh=fresh)
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="vertical_relaxed")
        return df

    def range_read(
        self,
        books: Iterable[TickerBook],
        start: datetime.date,
        end: datetime.date,
        fresh: bool = False,
    ) -> pl.DataFrame:
        stack = self.stack_read(books=books, fresh=fresh)
        df = stack.filter(pl.col("date").is_between(start, end))
        return df


class CNFundamental(BookLib[QuarterBook], BatchBookLib[QuarterBook]):
    REGION = "CN"
    SYMBOL = "fundamental"

    def _fetch_raw(self, book: QuarterBook) -> pl.DataFrame:
        return ak.AkShareMicro().quarterly_fundamentals(book.year, book.literal_quarter)

    def _fetch_batch_raw(self, books: Iterable[QuarterBook]) -> list[pl.DataFrame]:
        books = list(books)
        frames = ak.AkShareMicro().batch_quarterly_fundamentals(
            yqs=[(book.year, book.literal_quarter) for book in books]
        )
        return frames

    def range_fresh(self, start: datetime.date, end: datetime.date) -> None:
        books = QuarterBook.date_range(start, end)
        self.batch_fresh(books=list(books))

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return (
            Fundamental()
            .metrics(df)
            .sort("ts_code")
            .unique(subset=["ts_code"], keep="last")
        )

    def _process(self, book: QuarterBook, raw: pl.DataFrame) -> pl.DataFrame:
        return self._extract_features(raw)

    def range_read(
        self, start: datetime.date, end: datetime.date, fresh: bool = False
    ) -> pl.DataFrame:
        books = QuarterBook.date_range(start, end)
        frames = self.batch_read(books=books, fresh=fresh)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")



class CNMacro(BookLib[MacroBook]):
    REGION = "CN"
    SYMBOL = "macro"

    _INITIALIZED: ClassVar[bool] = False
    _FETCHERS: ClassVar[dict[str, Callable[[], pl.DataFrame]]] = {}
    _FEATURES: ClassVar[dict[str, Callable[[pl.DataFrame], pl.DataFrame]]] = {}

    def __init__(self, db: ArcticDB):
        self.db = db
        self._init_components_once()

    def _init_components_once(self) -> None:
        """Lazy one-time initialization of all fetcher & feature callables"""
        if hasattr(self.__class__, "_components_initialized"):
            return

        macro = ak.AkShareMacro()
        self.__class__._FETCHERS.update(
            {
                "northbound": macro.northbound_flow,
                "marginshort": macro.market_margin_short,
                "shibor": macro.shibor,
                "index": macro.csi1000_daily_ohlcv,
                "qvix": macro.csi1000qvix_daily_ohlc,
            }
        )
        nb = Northbound()
        ms = MarginShort()
        sh = Shibor()
        beh = Behavioral()
        self.__class__._FEATURES.update(
            {
                "northbound": nb.metrics,
                "marginshort": ms.metrics,
                "shibor": sh.metrics,
                "index": beh.metrics,
                "qvix": beh.metrics,
            }
        )

        self.__class__._INITIALIZED = True

    def _fetch_raw(self, book: MacroBook) -> pl.DataFrame:
        try:
            return self._FETCHERS[book.macro]()
        except KeyError:
            raise ValueError(f"Unknown macro book fetch: {book.macro}")

    def _process(self, book: MacroBook, raw: pl.DataFrame) -> pl.DataFrame:
        try:
            return self._FEATURES[book.macro](raw)
        except KeyError:
            raise ValueError(f"Unknown macro book process: {book.macro}")

    def read_all(self, *, fresh: bool = False) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for book in MacroBook.list():
            try:
                m = self.read(book, fresh=fresh)
                m = m.rename({c: f"{c}_{book}" for c in m.columns if c != "date"})
                frames.append(m)
            except KeyError as e:
                log.warning(f"Failed to read book '{book}': {e}")
                pass
        df = reduce(lambda df_1, df_2: df_1.join(df_2, on="date", how="inner"), frames)
        return df


class CNIndustrySectorGroup(SectorGroup):
    INDUS_COL: Final[list[str]] = ["sw_l1_code"]

    @classmethod
    def default_fundamental_metric(cls) -> list[str]:
        """
        Default fundamental metrics
        """
        return [
            # Profitability & margins (industry structure driven)
            "gross_margin",
            "operate_margin",
            "net_margin",
            # Capital efficiency (very industry dependent)
            "roe",
            "roa",
            "asset_turnover",
            # Leverage & balance sheet structure
            "debt_to_equity",
            "debt_asset_ratio",
            # Liquidity ratios (business-model specific)
            "current_ratio",
            "quick_ratio",
            # Cash flow efficiency
            "cfo_yield",
            "ttm_cfo_yield",
            # Accrual / earnings quality (sector accounting conventions)
            "accrual_ratio",
            # Growth rates (structural growth differences)
            "revenue_growth_yoy",
            "net_profit_growth_yoy",
            "asset_growth_yoy",
            "debt_growth_yoy",
            # TTM profitability
            "ttm_roe",
        ]

    def __init__(
        self,
        db: ArcticDB,
        factors: list[str],
        std_suffix: str = "_z",
        min_group_size: int = 5,
        skip_winsor: bool = False,
        by: list[str] | None = None,
        winsor_limits: tuple[float, float] = (0.01, 0.99),
    ):
        self.stock_pool = CNStockPool(db)
        self.industry_df: pl.DataFrame | None = None
        by = self.INDUS_COL + by if by else self.INDUS_COL
        super().__init__(
            by=by,
            factors=factors,
            std_suffix=std_suffix,
            min_group_size=min_group_size,
            skip_winsor=skip_winsor,
            winsor_limits=winsor_limits,
        )

    def __call__(self, df: pl.DataFrame, fresh: bool = False) -> pl.DataFrame:
        self.industry_df = self.stock_pool.read_codes("industry_code", fresh)
        required = ["ts_code"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"factor_df missing required columns: {missing}")

        merged = df.join(self.industry_df, on=required, how="inner").filter(
            pl.col(self.INDUS_COL).is_null()
        )
        original_rows = len(df)
        merged_rows = len(merged)
        if merged_rows < original_rows:
            log.info(
                f"Industry join: {merged_rows}/{original_rows} rows retained "
                f"({merged_rows / original_rows * 100:.1f}%)"
            )

        if merged_rows == 0:
            log.warning("No data after industry join. Check industry data coverage.")
            return df

        return self.normalize(merged)
