from datetime import date
from functools import reduce
from typing import Literal, cast

import polars as pl

import quant_trade.provider.akshare as ak
import quant_trade.provider.baostock as bs
from quant_trade.feature.process import Northbound, MarginShort, Shibor
from quant_trade.config.arctic import ArcticAdapter, ArcticDB, Lib
from quant_trade.config.logger import log
from quant_trade.provider.utils import Quarter

from ..provider.traits import Source
from .process import (
    Behavioral,
    Fundamental,
    SectorGroup,
)


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

    type Book = Literal["stock_code", "industry_code", "csi500_code"]
    type Universe = Literal["whole", "csi500"]

    @staticmethod
    def _fetch_stock_code() -> pl.DataFrame:
        return ak.AkShareUniverse().stock_whole()

    @staticmethod
    def _fetch_industry_code() -> pl.DataFrame:
        return ak.SWIndustryCls().stock_l1_industry_cls(fresh=True)

    @staticmethod
    def _fetch_csi500_code() -> pl.DataFrame:
        return bs.BaoMacro().csi500_cons()

    def fresh(self, book: Book):
        """Ensure stock_code and industry_code exist in DB."""
        match book:
            case "stock_code":
                df = self._fetch_stock_code()
            case "industry_code":
                df = self._fetch_industry_code()
            case "csi500_code":
                df = self._fetch_csi500_code()
        log.info(f"Writing {len(df)} rows to {book}")
        self._lib.write(book, ArcticAdapter.to_write(df))

    def read(self, book: Book, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._lib.has_symbol(book):
            self.fresh(book)
        if self._lib.has_symbol(book):
            return ArcticAdapter.from_read(self._lib.read(book))
        raise KeyError(f"Symbol '{book}' not found in {self.lib_name()}")

    def universe(
        self, universe: Universe, industry_cls: bool = False, fresh: bool = False
    ) -> pl.DataFrame:
        match universe:
            case "csi500":
                df = self.read("csi500_code", fresh)
            case "whole":
                df = self.read("stock_code", fresh)
        if industry_cls:
            ind = self.read("industry_code", fresh)
            df = df.join(ind, on="ts_code", how="inner")
        return df


class CNMarket(AssetLib):
    REGION = "CN"
    SYMBOL = "market"

    def __init__(self, source: Source = "baostock"):
        self._source: Source = source

    def _fetch_raw(self, ts_code: str) -> pl.DataFrame:
        match self._source:
            case "akshare":
                return ak.AkShareMicro().market_ohlcv(ts_code, "daily")
            case "baostock":
                return bs.BaoMicro().market_ohlcv(ts_code, "daily")

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

    def read(self, ts_code: str, fresh: bool = False) -> pl.DataFrame:
        if fresh or not self._lib.has_symbol(ts_code):
            self.fresh(ts_code)
        if self._lib.has_symbol(ts_code):
            return ArcticAdapter.from_read(self._lib.read(ts_code))
        raise KeyError(f"Market data for {ts_code} not found in {self.lib_name()}")

    def batch_read(self, ts_codes: list[str], fresh: bool = False) -> pl.DataFrame:
        """Batch read multiple stocks (useful for panel construction)."""
        frames = []
        for ts in ts_codes:
            try:
                frames.append(self.read(ts, fresh=fresh))
            except KeyError:
                log.warning(f"Skipping {ts} — data not available")
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")

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
    INDUS_COL: list[str] = ["sw_l1_code"]

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
        self.industry_df = self.stock_pool.read("industry_code", fresh)
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
