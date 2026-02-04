from dataclasses import asdict, astuple, dataclass, fields, is_dataclass
from datetime import date
from functools import reduce
from typing import Any, Final, Iterator, Literal, Protocol, Self, cast, final, runtime_checkable

import polars as pl

import quant_trade.provider.akshare as ak
import quant_trade.provider.baostock as bs
from quant_trade.config.arctic import ArcticAdapter, ArcticDB, Lib
from quant_trade.config.logger import log
from quant_trade.feature.process import MarginShort, Northbound, Shibor
from quant_trade.provider.utils import Quarter
from quant_trade.provider.traits import Source

from .process import (
    Behavioral,
    Fundamental,
    SectorGroup,
)

def book_key[T](cls: type[T]) -> type[T]:
    """
    Class decorator that auto-generates BookKey implementation.
    
    Joins all fields with "_" to create storage key.
    Nested BookKey objects are recursively converted.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")
    
    original_repr = getattr(cls, '__repr__', None)
    
    def is_bookkey(obj: Any) -> bool:
        return isinstance(obj, BookKey) if hasattr(obj, 'to_key') else False
    
    # Define methods
    def to_key(self: Any) -> str:
        parts = []
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if is_bookkey(val):
                parts.append(val.to_key())
            elif isinstance(val, date):
                parts.append(val.isoformat())
            else:
                parts.append(str(val))
        return "_".join(parts)
    
    def __iter__(self: Any) -> Iterator[Any]:
        """Enable unpacking: year, q = book"""
        yield from astuple(self)  
    
    @property  
    def _fields(self: Any) -> tuple[str, ...]:
        """Return field names."""
        return tuple(f.name for f in fields(self))  
    
    def to_dict(self: Any) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)  
    
    cls.to_key = to_key  # type: ignore[attr-assign]
    cls.__iter__ = __iter__  # type: ignore[method-assign]
    cls._fields = _fields  # type: ignore[attr-defined]
    cls.to_dict = to_dict  # type: ignore[attr-defined]
    
    return cls
        
@runtime_checkable
class BookKey(Protocol):
    """Structural trait: anything with to_key() is a valid book identifier."""
    
    def to_key(self) -> str: ...
            

@final
@book_key
@dataclass(frozen=True,slots=True)
class QuarterBook:
    """Year-quarter book identifier: '2024_Q1'"""
    year: int
    quarter: Quarter

    def to_key(self) -> str:
        return f"{self.year}_Q{self.quarter}"

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


class CNStockPool(AssetLib):
    REGION = "CN"
    SYMBOL = "stock"

    type Book = Literal["stock_code", "industry_code"]
    type Universe = Literal["csi500"]

    @staticmethod
    def index_book(universe: Universe, date: date) -> str:
        return f"{universe}_{date}"

    @staticmethod
    def _fetch_stock_code() -> pl.DataFrame:
        return ak.AkShareUniverse().stock_whole()

    @staticmethod
    def _fetch_industry_code() -> pl.DataFrame:
        return ak.SWIndustryCls().stock_l1_industry_cls(fresh=True)

    @staticmethod
    def _fetch_csi500_code() -> pl.DataFrame:
        return bs.BaoMacro().csi500_cons()

    def fresh_codes(self, book: Book):
        """Ensure stock_code and industry_code exist in DB."""
        match book:
            case "stock_code":
                df = self._fetch_stock_code()
            case "industry_code":
                df = self._fetch_industry_code()
        log.info(f"Writing {len(df)} rows to {book}")
        self._lib.write(book, ArcticAdapter.to_write(df))

    def fresh_universe(self, date: date, universe: Universe):
        """Fetch and store stock universe for a given date."""
        book = self.index_book(universe, date)
        match universe:
            case "csi500":
                df = self._fetch_csi500_code()
        log.info(f"Writing {len(df)} rows to {book}")
        self._lib.write(book, ArcticAdapter.to_write(df))

    def read_codes(self, book: Book, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._lib.has_symbol(book):
            self.fresh_codes(book)
        if self._lib.has_symbol(book):
            return ArcticAdapter.from_read(self._lib.read(book))
        raise KeyError(f"Symbol '{book}' not found in {self.lib_name()}")

    def read_pool(
        self,
        universe: Universe,
        date: date,
        industry_cls: bool = False,
        fresh: bool = False,
    ) -> pl.DataFrame:
        book = self.index_book(universe, date)
        if fresh or not self._lib.has_symbol(book):
            self.fresh_universe(date, universe)
        if self._lib.has_symbol(book):
            df = ArcticAdapter.from_read(self._lib.read(book))
            if industry_cls:
                ind = self.read_codes("industry_code", fresh=fresh)
                df = df.join(ind, on="ts_code", how="inner")
            return df
        raise KeyError(f"Universe '{book}' not found in {self.lib_name()}")


class CNMarket(AssetLib):
    REGION = "CN"
    SYMBOL = "market"

    def __init__(self, db: ArcticDB, source: Source = "baostock"):
        super().__init__(db)
        self._source: Source = source

    def _fetch_raw(self, ts_code: str) -> pl.DataFrame:
        match self._source:
            case "akshare":
                return ak.AkShareMicro().market_ohlcv(ts_code, "daily")
            case "baostock":
                return bs.BaoMicro().market_ohlcv(ts_code, "daily")

    def _fetch_batch_raw(self, ts_codes: list[str]) -> list[pl.DataFrame]:
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

    def fresh(
        self,
        ts_code: str,
    ) -> None:
        log.info(f"Fetching & processing market data for {ts_code}")
        raw = self._fetch_raw(ts_code)
        if raw.is_empty():
            log.warning(f"No market data for {ts_code}")
            return
        feat = self._extract_features(raw)
        feat = self._normalize(feat)
        self._lib.write(ts_code, ArcticAdapter.to_write(feat))

    def fresh_batch(self, ts_codes: list[str]) -> None:
        log.info(
            f"Fetching & processing market data for batch of {len(ts_codes)} stocks"
        )
        raw = self._fetch_batch_raw(ts_codes)
        for code, df in zip(ts_codes, raw):
            if df.is_empty():
                log.warning(f"No market data for {code}")
                continue
            feat = self._extract_features(df)
            feat = self._normalize(feat)
            self._lib.write(code, ArcticAdapter.to_write(feat))

    def read(self, ts_code: str, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._lib.has_symbol(ts_code):
            self.fresh(ts_code)
        if self._lib.has_symbol(ts_code):
            return ArcticAdapter.from_read(self._lib.read(ts_code))
        raise KeyError(f"Market data for {ts_code} not found in {self.lib_name()}")

    def batch_read(self, ts_codes: list[str], fresh: bool = False) -> pl.DataFrame:
        """Batch read multiple stocks (useful for panel construction)."""
        to_fetch:list[str] = []
        cached:dict[str,pl.DataFrame] = {}
        for code in ts_codes:
            if fresh or not self._lib.has_symbol(code):
                to_fetch.append(code)
                continue
            try:
                df = ArcticAdapter.from_read(self._lib.read(code))
                if not df.is_empty():
                    cached[code] = df
            except Exception as e:
                log.warning(f"Cache read failed for {code}: {e}")
                to_fetch.append(code)

        fetched:dict[str,pl.DataFrame] = {}
        if to_fetch:
            frames = self._fetch_batch_raw(to_fetch)
            for code, df in zip(to_fetch, frames):
                if df.is_empty():
                    continue
                fetched[code] = df
                try:
                    self._lib.write(code, ArcticAdapter.to_write(df))
                except Exception as e:
                    log.error(f"Cache write failed for {code}: {e}")

        dfs:list[pl.DataFrame] = []
        for code in ts_codes:
            df = fetched.get(code) or cached.get(code)
            if df is not None:
                dfs.append(df)
        
        return (
            pl.concat(dfs, how="vertical_relaxed") if dfs else pl.DataFrame()
        )


    def range_read(
        self, ts_codes: list[str], start: date, end: date, fresh: bool = False
    ) -> pl.DataFrame:
        batch = self.batch_read(ts_codes=ts_codes, fresh=fresh)
        df = batch.filter(pl.col("date").is_between(start, end))
        return df


class CNFundamental(AssetLib):
    REGION = "CN"
    SYMBOL = "fundamental"

    @staticmethod
    def book(year: int, quarter: Quarter) -> str:
        return f"{year}Q{quarter}"

    @staticmethod
    def _fetch_raw(year: int | None, quarter: Quarter) -> pl.DataFrame:
        return ak.AkShareMicro().quarterly_fundamentals(year, quarter)

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return Fundamental().metrics(df)

    def fresh(self, year: int, quarter: Quarter) -> None:
        book = self.book(year, quarter)
        log.info(f"Fetching fundamentals {book}")

        raw = self._fetch_raw(year, quarter)
        if raw.is_empty():
            log.warning(f"No raw fundamentals for {book}")
            return

        feat = (
            self._extract_features(raw)
            .sort("ts_code")
            .unique(subset=["ts_code"], keep="last")
        )

        self._lib.write(book, ArcticAdapter.to_write(feat))

    def read(self, year: int, quarter: Quarter, *, fresh: bool = False) -> pl.DataFrame:
        book = self.book(year, quarter)

        if fresh or not self._lib.has_symbol(book):
            self.fresh(year, quarter)

        if not self._lib.has_symbol(book):
            raise KeyError(f"Fundamental {book} not found")

        return ArcticAdapter.from_read(self._lib.read(book))

    def batch_read(
        self,
        start_year: int,
        start_q: Quarter,
        end_year: int,
        end_q: Quarter,
        fresh: bool = False,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []

        y, q = start_year, start_q
        while (y, q) <= (end_year, end_q):
            try:
                frames.append(self.read(y, cast(Quarter, q), fresh=fresh))
            except KeyError:
                log.debug(f"Missing {y}Q{q}")
            q += 1
            if q > 4:
                q = 1
                y += 1

        return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()

    def range_read(
        self, start: date, end: date, *, fresh: bool = False
    ) -> pl.DataFrame:
        return self.batch_read(start.year - 1, 1, end.year, 4, fresh=fresh)

    def join_stock_asof(
        self, start: date, end: date, df: pl.DataFrame, *, fresh: bool = False
    ) -> pl.DataFrame:
        fund = self.range_read(start=start, end=end, fresh=fresh)
        return fund.join_asof(
            df,
            left_on="announcement_date",
            right_on="date",
            by="ts_code",
            strategy="backward",
        )


class CNMacro(AssetLib):
    REGION = "CN"
    SYMBOL = "macro"

    Book = Literal["northbound", "marginshort", "shibor", "index", "qvix"]

    _fetch = {
        "northbound": ak.AkShareMacro().northbound_flow,
        "marginshort": ak.AkShareMacro().market_margin_short,
        "shibor": ak.AkShareMacro().shibor,
        "index": ak.AkShareMacro().csi1000_daily_ohlcv,
        "qvix": ak.AkShareMacro().csi1000qvix_daily_ohlc,
    }

    _features = {
        "northbound": Northbound().metrics,
        "marginshort": MarginShort().metrics,
        "shibor": Shibor().metrics,
        "index": Behavioral().metrics,
        "qvix": Behavioral().metrics,
    }

    def fresh(self, book: Book) -> None:
        raw = self._fetch[book]()
        if raw.is_empty():
            log.warning(f"No macro data for {book}")
            return

        feat = self._features[book](raw).sort("date").unique("date", keep="last")

        self._lib.write(book, ArcticAdapter.to_write(feat))

    def read(self, book: Book, *, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._lib.has_symbol(book):
            self.fresh(book)

        if not self._lib.has_symbol(book):
            raise KeyError(f"Macro book {book} not found")

        return ArcticAdapter.from_read(self._lib.read(book))

    def read_all(self, *, fresh: bool = False) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for book in ("northbound", "marginshort", "shibor", "index", "qvix"):
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
            "operating_margin",
            "net_margin",
            # Capital efficiency (very industry dependent)
            "roe",
            "roa",
            "asset_turnover",
            # Leverage & balance sheet structure
            "debt_to_equity",
            "debt_to_assets",
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
            "assets_growth_yoy",
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
