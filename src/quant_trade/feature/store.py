from datetime import date
from typing import Literal, cast

import polars as pl

import quant_trade.provider.akshare as ak
from quant_trade.config.arctic import ArcticAdapter, ArcticDB, Lib
from quant_trade.config.logger import log
from quant_trade.provider.utils import Quarter

from .process import (
    Behavioral,
    Fundamental,
    MarginShort,
    Northbound,
    SectorGroup,
    Shibor,
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


class CNStockMap(AssetLib):
    REGION = "CN"
    SYMBOL = "stock"

    Book = Literal["stock_code", "industry_code"]

    @staticmethod
    def _fetch_stock_code() -> pl.DataFrame:
        return ak.AkShareMicro().stock_whole()

    @staticmethod
    def _fetch_industry_code() -> pl.DataFrame:
        return ak.SWIndustryCls().stock_l1_industry_cls(fresh=True)

    def ensure(self, *, fresh: bool = False):
        """Ensure stock_code and industry_code exist in DB."""
        if fresh or not self._lib.has_symbol("stock_code"):
            df = self._fetch_stock_code()
            log.info(f"Writing {len(df)} rows to stock_code")
            self._lib.write("stock_code", ArcticAdapter.to_write(df))

        if fresh or not self._lib.has_symbol("industry_code"):
            df = self._fetch_industry_code()
            log.info(f"Writing {len(df)} rows to industry_code")
            self._lib.write("industry_code", ArcticAdapter.to_write(df))

    def read(
        self, book: Book, ensure: bool = True, fresh: bool = False
    ) -> pl.DataFrame:
        if ensure:
            self.ensure(fresh=fresh)
        if self._lib.has_symbol(book):
            return ArcticAdapter.from_read(self._lib.read(book))
        raise KeyError(f"Symbol '{book}' not found in {self.lib_name()}")


class CNMarket(AssetLib):
    REGION = "CN"
    SYMBOL = "market"

    @staticmethod
    def _fetch_raw(ts_code: str) -> pl.DataFrame:
        return ak.AkShareMicro().market_ohlcv(ts_code, "daily")

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return Behavioral().metrics(df)

    def ensure(self, ts_code: str, *, fresh: bool = False) -> None:
        if not fresh and self._lib.has_symbol(ts_code):
            return
        log.info(f"Fetching & processing market data for {ts_code}")
        raw = self._fetch_raw(ts_code)
        if raw.is_empty():
            log.warning(f"No market data for {ts_code}")
            return
        feat = self._extract_features(raw)
        feat = feat.sort("date").unique(subset=["date"], keep="last")
        self._lib.write(ts_code, ArcticAdapter.to_write(feat))

    def read(self, ts_code: str, ensure: bool = True) -> pl.DataFrame:
        if ensure:
            self.ensure(ts_code)
        if self._lib.has_symbol(ts_code):
            return ArcticAdapter.from_read(self._lib.read(ts_code))
        raise KeyError(f"Market data for {ts_code} not found in {self.lib_name()}")

    def batch_read(self, ts_codes: list[str], ensure: bool = True) -> pl.DataFrame:
        """Batch read multiple stocks (useful for panel construction)."""
        frames = []
        for ts in ts_codes:
            try:
                frames.append(self.read(ts, ensure=ensure))
            except KeyError:
                log.warning(f"Skipping {ts} — data not available")
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")


class CNFundamental(AssetLib):
    REGION = "CN"
    SYMBOL = "fundamental"

    NEUTRALIZED = (
        [
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
        ],
    )

    def __init__(self, db: ArcticDB):
        super().__init__(db)
        self._stock_map = CNStockMap(db)

    @staticmethod
    def book_index(year: int, quarter: Quarter) -> str:
        return f"{year}Q{quarter}"

    @staticmethod
    def _fetch_raw(year: int | None, quarter: Quarter) -> pl.DataFrame:
        return ak.AkShareMicro().quarterly_fundamentals(year, quarter)

    @staticmethod
    def _extract_features(df: pl.DataFrame) -> pl.DataFrame:
        return Fundamental().metrics(df)

    def _neutralize(
        self,
        df: pl.DataFrame,
        factors: list[str] | str | None = None,
    ) -> pl.DataFrame:
        indus_cls = self._stock_map.read("industry_code")
        if indus_cls.is_empty():
            log.error("Industry class is empty, fail to neutralize")

        df = df.join(indus_cls, on="ts_code", how="left")
        missing = (
            df.filter(pl.col("sw_l1_code").is_null())["ts_code"].unique().to_list()
        )
        if missing:
            log.warning(f"{len(missing)} is not in industry class, excluding")
        df = df.filter(pl.col("sw_l1_code").is_not_null())
        if df.is_empty():
            log.warning("Empty dataframe, fail to newtralize")

        if isinstance(factors, str):
            factors = [factors]
        if factors is None:
            exclude = {"ts_code", "name", "announcement_date", "sw_l1_code"}
            factors = [
                c for c in df.columns if c not in exclude and df[c].dtype.is_numeric()
            ]
            log.info(f"Automatically choose factors：{factors}")

        df = SectorGroup.normalize(
            df, factors=factors, by="sw_l1_code", skip_winsor=True
        )
        return df

    def ensure(self, year: int, quarter: Quarter, *, fresh: bool = False) -> None:
        book = self.book_index(year, quarter)
        if not fresh and self._lib.has_symbol(book):
            return
        log.info(f"Fetching & processing fundamentals for {year}Q{quarter}")
        raw = self._fetch_raw(year, quarter)
        if raw.is_empty():
            log.warning(f"No fundamental data for {year}Q{quarter}")
            return
        feat = self._extract_features(raw)
        feat = feat.sort("ts_code").unique(subset=["ts_code"], keep="last")
        feat = self._neutralize(
            feat,
            factors=[
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
            ],
        )
        self._lib.write(book, ArcticAdapter.to_write(feat))

    def read(self, year: int, quarter: Quarter, ensure: bool = True) -> pl.DataFrame:
        book = self.book_index(year, quarter)
        if ensure:
            self.ensure(year, quarter)
        if self._lib.has_symbol(book):
            return ArcticAdapter.from_read(self._lib.read(book))
        raise KeyError(f"Fundamental data {year}Q{quarter} not found")

    def read_range(
        self,
        start_year: int,
        start_quarter: Quarter,
        end_year: int,
        end_quarter: Quarter,
    ) -> pl.DataFrame:
        """Read fundamentals over a quarter range (inclusive)."""
        frames = []
        y = start_year
        q: int = start_quarter
        end_q: int = end_quarter

        while (y, q) <= (end_year, end_q):
            try:
                df = self.read(y, cast(Quarter, q), ensure=False)
                if not df.is_empty():
                    frames.append(df)
            except KeyError:
                log.debug(f"Skipping missing quarter {y}Q{q}")
            q += 1
            if q > 4:
                q = 1
                y += 1

        if not frames:
            log.warning(
                f"No fundamental data found in range {start_year}Q{start_quarter} — {end_year}Q{end_quarter}"
            )
            return pl.DataFrame()

        return pl.concat(frames, how="vertical_relaxed")


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

    def ensure(self, book: Book, *, fresh: bool = False) -> None:
        if not fresh and self._lib.has_symbol(book):
            return
        log.info(f"Fetching & processing macro book: {book}")
        raw = self._book_map[book]()
        if raw.is_empty():
            log.warning(f"No raw data for macro book {book}")
            return
        feat = self._feature_map[book](raw)
        feat = feat.sort("date").unique(subset=["date"], keep="last")
        self._lib.write(book, ArcticAdapter.to_write(feat))

    def read(self, book: Book, ensure: bool = True) -> pl.DataFrame:
        if ensure:
            self.ensure(book)
        if self._lib.has_symbol(book):
            return ArcticAdapter.from_read(self._lib.read(book))
        raise KeyError(f"Macro book '{book}' not found in {self.lib_name()}")


class CNFeatures(AssetLib):
    """Daily panel features assembly for quant AI workflows."""

    REGION = "CN"
    SYMBOL = "features"

    def __init__(self, db: ArcticDB):
        super().__init__(db)
        self.stock_map = CNStockMap(db)
        self.market = CNMarket(db)
        self.fundamental = CNFundamental(db)
        self.macro = CNMacro(db)

    def assemble(
        self,
        ts_codes: list[str],
        start: date,
        end: date,
        ensure: bool = True,
    ) -> pl.DataFrame:
        """Assemble market + fundamental + macro features for given stocks and date range."""
        # 1. Market data (base panel)
        panel: list[pl.DataFrame] = []
        for ts in ts_codes:
            try:
                mkt = self.market.read(ts, ensure=ensure)
                if mkt.is_empty():
                    continue
                mkt = mkt.filter(pl.col("date").is_between(start, end))
                if mkt.is_empty():
                    continue
                mkt = mkt.with_columns(pl.lit(ts).alias("ts_code"))
                panel.append(mkt)
            except KeyError:
                log.warning(f"Market data missing for {ts}")
                continue

        if not panel:
            log.warning(
                f"No market data found for {len(ts_codes)} stocks in range {start} — {end}"
            )
            return pl.DataFrame()

        df = pl.concat(panel).sort(["ts_code", "date"])

        # 2. Fundamentals (asof join)
        fund_start_y = max(start.year - 1, 2000)  # reasonable floor
        fund = self.fundamental.read_range(fund_start_y, 1, end.year, 4)
        if not fund.is_empty():
            fund = fund.sort(["ts_code", "announcement_date"])
            df = df.join_asof(
                fund,
                left_on="date",
                right_on="announcement_date",
                by="ts_code",
                strategy="backward",
            )
        else:
            log.warning("No fundamental data available in range")

        # 3. Macro overlays (left join on date)
        macro_books: list[CNMacro.Book] = [
            "northbound",
            "marginshort",
            "shibor",
            "index",
            "qvix",
        ]
        for book in macro_books:
            try:
                m_df = self.macro.read(book, ensure=ensure)
                if m_df.is_empty():
                    continue
                # Rename to avoid column conflicts
                m_df = m_df.rename(
                    {c: f"{c}_{book}" for c in m_df.columns if c != "date"}
                )
                df = df.join(m_df, on="date", how="inner")
            except KeyError:
                log.warning(f"Macro book {book} not available")

        return df.sort(["ts_code", "date"])

    def load_range(
        self,
        start: date,
        end: date,
        ts_codes: list[str] | str | None = None,
        ensure: bool = True,
    ) -> pl.DataFrame:
        """Load assembled features for multiple stocks over a date range."""
        if ts_codes is None:
            ts_codes = self.stock_map.read("stock_code", ensure=ensure)[
                "ts_code"
            ].to_list()
        if isinstance(ts_codes, str):
            ts_codes = [ts_codes]
        return self.assemble(ts_codes, start, end, ensure=ensure)

    def load_batch(
        self,
        date: date,
        ts_codes: list[str] | str | None = None,
        ensure: bool = True,
    ) -> pl.DataFrame:
        """Load features for a single date (cross-sectional / inference ready)."""
        df = self.load_range(date, date, ts_codes, ensure=ensure)
        return df.filter(pl.col("date") == date)
