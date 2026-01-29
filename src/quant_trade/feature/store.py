from typing import Literal

import quant_trade.provider.akshare as ak
from quant_trade.config import ArcticAdapter, ArcticDB
from quant_trade.provider.utils import Quarter


class AssetLib:
    REGION: str
    SYMBOL: str

    def __init__(self, db: ArcticDB) -> None:
        self.lib = db.get_lib(self.lib_name())

    @classmethod
    def lib_name(cls) -> str:
        return f"{cls.REGION}_{cls.SYMBOL}"


class CNStockMap(AssetLib):
    REGION: str = "CN"
    SYMBOL: str = "stock"

    type Book = Literal["stock_code", "industry_code"]

    def setup(self, fresh: bool = False):
        micro = ak.AkShareMicro()
        indus = ak.SWIndustryCls()

        if fresh or not self.lib.has_symbol("stock_code"):
            self.lib.write("stock_code", ArcticAdapter.input(micro.stock_whole()))
        if fresh or not self.lib.has_symbol("industry_code"):
            self.lib.write(
                "industry_code", ArcticAdapter.input(indus.stock_l1_industry_cls())
            )


class CNMarket(AssetLib):
    REGION: str = "CN"
    SYMBOL: str = "market"
    ...


class CNFundamental(AssetLib):
    REGION: str = "CN"
    SYMBOL: str = "fundamental"

    @staticmethod
    def book_index(year: int, quarter: Quarter) -> str:
        return f"{year}Q{quarter}"

    ...


class CNMacro(AssetLib):
    REGION: str = "CN"
    SYMBOL: str = "macro"

    type Book = Literal["northbound", "marginshort", "shibor", "index", "qvix"]
    ...
