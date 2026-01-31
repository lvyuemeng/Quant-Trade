import typing
from datetime import date
from typing import Literal

import narwhals as nw
import polars as pl

import quant_trade.provider.akshare as ak
from quant_trade.config.arctic import ArcticAdapter, ArcticDB, Lib
from quant_trade.config.logger import log
from quant_trade.provider.utils import Quarter

from .process import Behavioral, Fundamental, MarginShort, Northbound, Shibor


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


class CNStockMap(AssetLib):
    REGION = "CN"
    SYMBOL = "stock"

    Book = Literal["stock_code", "industry_code"]

    @staticmethod
    def _fetch_stock_code() -> pl.DataFrame:
        return ak.AkShareMicro().stock_whole()

    @staticmethod
    def _fetch_industry_code() -> pl.DataFrame:
        return ak.SWIndustryCls().stock_l1_industry_cls()

    def ensure(self, *, fresh: bool = False):
        """Ensure stock_code and industry_code exist in DB."""
        if fresh or not self._lib.has_symbol("stock_code"):
            df = self._fetch_stock_code()
            self._lib.write("stock_code", ArcticAdapter.to_write(df))

        if fresh or not self._lib.has_symbol("industry_code"):
            df = self._fetch_industry_code()
            self._lib.write("industry_code", ArcticAdapter.to_write(df))

    def read(self, book: Book, ensure: bool = True) -> nw.DataFrame:
        if ensure:
            self.ensure()
        return ArcticAdapter.from_read(self._lib.read(book))


class CNMarket(AssetLib):
    REGION = "CN"
    SYMBOL = "market"

    @staticmethod
    def _fetch_raw(ts_code: str) -> pl.DataFrame:
        return ak.AkShareMicro().market_ohlcv(ts_code, "daily")

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return Behavioral().metrics(df)

    def ensure(self, ts_code: str, *, fresh: bool = False):
        """Ensure per-stock time series exists in DB."""
        if not fresh and self._lib.has_symbol(ts_code):
            return

        raw = self._fetch_raw(ts_code)
        feat = self._extract_features(raw)
        feat = feat.sort("date").unique(subset=["date"], keep="last")
        self._lib.write(ts_code, ArcticAdapter.to_write(feat))

    def read(self, ts_code: str, ensure: bool = True) -> nw.DataFrame:
        if ensure:
            self.ensure(ts_code)
        if self._lib.has_symbol(ts_code):
            return ArcticAdapter.from_read(self._lib.read(ts_code).data).
        raise ValueError(f"CNMarket does not have the {ts_code} daily market data.")

    def batch_read(self, ts_codes: list[str], ensure: bool = True) -> nw.DataFrame:
        """Read multiple stocks at once (batch-friendly for AI)."""
        frames = []
        for ts in ts_codes:
            frames.append(self.read(ts, ensure=ensure))
        return nw.concat(frames)


class CNFundamental(AssetLib):
    REGION = "CN"
    SYMBOL = "fundamental"

    @staticmethod
    def book_index(year: int, quarter: Quarter) -> str:
        return f"{year}Q{quarter}"

    @staticmethod
    def _fetch_raw(year: int, quarter: Quarter) -> pl.DataFrame:
        return ak.AkShareMicro().quarterly_fundamentals(year, quarter)

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return Fundamental().metrics(df)

    def ensure(self, *, year: int, quarter: Quarter, fresh: bool = False):
        book = self.book_index(year, quarter)
        if not fresh and self._lib.has_symbol(book):
            return
        raw = self._fetch_raw(year, quarter)
        feat = self._extract_features(raw)
        feat = feat.sort("ts_code").unique(subset=["ts_code"], keep="last")
        self._lib.write(book, ArcticAdapter.to_write(feat))

    def read(self, year: int, quarter: Quarter, ensure: bool = True) -> nw.DataFrame:
        book = self.book_index(year, quarter)
        if ensure:
            self.ensure(year=year, quarter=quarter)
        if self._lib.has_symbol(book):
            return ArcticAdapter.from_read(self._lib.read(book).data).to_native()
        raise ValueError(f"CNFundamental does not have {year}Q{quarter} data.")

    def read_range(
        self, start_year: int, start_q: Quarter, end_year: int, end_q: Quarter
    ) -> nw.DataFrame:
        quarters = []

        # Calculate total quarters for cleaner loop
        total_quarters = (end_year - start_year) * 4 + (end_q - start_q) + 1

        curr_y, curr_q = start_year, start_q
        for _ in range(total_quarters):
            df = self.read(curr_y, typing.cast(Quarter, curr_q))
            if df is not None and not df.is_empty():
                quarters.append(df)

            # Update quarter/year
            curr_q = 1 if curr_q == 4 else curr_q + 1
            if curr_q == 1:
                curr_y += 1

        if quarters:
            return nw.concat(quarters, how="vertical")
        raise ValueError(
            f"CNFundamental does not have {start_year}Q{start_q} to {end_year}Q{end_q} data."
        )


class CNMacro(AssetLib):
    REGION = "CN"
    SYMBOL = "macro"

    Book = Literal["northbound", "marginshort", "shibor", "index", "qvix"]

    _book_map = {
        "northbound": ak.AkShareMacro().northbound_flow,
        "marginshort": ak.AkShareMacro().market_margin_short,
        "shibor": ak.AkShareMacro().shibor,
        "index": ak.AkShareMacro().csi1000_daily_ohlcv,
        "qvix": ak.AkShareMacro().csi1000qvix_daily_ohlc,
    }

    _feature_map = {
        "northbound": Northbound().metrics,
        "marginshort": MarginShort().metrics,
        "shibor": Shibor().metrics,
        "index": Behavioral().metrics,
        "qvix": Behavioral().metrics,
    }

    def ensure(self, book: Book, *, fresh: bool = False):
        if not fresh and self._lib.has_symbol(book):
            return
        raw = self._book_map[book]()
        feat = self._feature_map[book](raw)
        feat = feat.sort("date").unique(subset=["date"], keep="last")
        self._lib.write(book, ArcticAdapter.to_write(feat))

    def read(self, book: Book, ensure: bool = True) -> nw.DataFrame:
        if ensure:
            self.ensure(book)
        return ArcticAdapter.from_read(self._lib.read(book))


class CNFeatures(AssetLib):
    """Daily panel features for quant AI workflows."""

    REGION = "CN"
    SYMBOL = "features"

    def __init__(self, db: ArcticDB):
        super().__init__(db)
        self.map = CNStockMap(db)
        self.market = CNMarket(db)
        self.fundamental = CNFundamental(db)
        self.macro = CNMacro(db)

    def assemble(self, ts_codes: list[str], start: date, end: date) -> pl.DataFrame:
        """Assemble market + fundamental + macro features for a date range."""
        # --- 1. Market Data (Base Layer) ---
        panel: list[pl.DataFrame] = []
        for ts in ts_codes:
            mkt = self.market.read(ts, ensure=True).to_polars()
            if mkt.is_empty():
                continue
            mkt = mkt.filter(pl.col("date").is_between(start, end))
            if mkt.is_empty():
                continue
            mkt = mkt.with_columns(pl.lit(ts).alias("ts_code"))
            panel.append(mkt)
        df = pl.concat(panel).sort(["ts_code", "date"])

        # --- 2. Fundamental Data ---
        fund = self.fundamental.read_range(start.year - 1, 1, end.year, 4).to_polars()
        if fund.is_empty():
            log.warning(f"assemble fundamental metrics between {start} and {end}")
        else:
            fund = fund.sort(["ts_code", "announcement_date"])
            df = df.join_asof(
                fund,
                left_on="date",
                right_on="announcement_date",
                by="ts_code",
                strategy="backward",
                suffix="_fund",
            )

        # --- 3. Macro Data ---
        macro_books: list[CNMacro.Book] = [
            "northbound",
            "marginshort",
            "shibor",
            "index",
            "qvix",
        ]
        for book in macro_books:
            m_df = self.macro.read(book, ensure=True).to_polars()
            if m_df.is_empty():
                continue
            suffix = f"_{book}"
            m_df = m_df.rename({c: f"{c}{suffix}" for c in m_df.columns if c != "date"})
            df = df.join(m_df, on="date", how="left")

        return df.sort(["date", "ts_code"])

    def load_range(
        self,
        start: date,
        end: date,
        ts_codes: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load features for multiple stocks and a date range."""
        codes = (
            ts_codes
            if ts_codes is not None
            else self.map.read("stock_code").to_polars().item(column="ts_code")
        )

        return self.assemble(codes, start, end)

    def load_batch(self, date: date, ts_codes: list[str] | None = None) -> pl.DataFrame:
        """Load features for a single date (ranking/inference)."""
        df = self.load_range(date, date, ts_codes)
        return df.filter(pl.col("date") == date)
