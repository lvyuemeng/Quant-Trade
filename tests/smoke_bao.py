from datetime import date

import quant_trade.provider.baostock as bs
from tests.conftest import smoke_configure

INDEX_CODE = "000001"
PERIOD = "daily"


def test_stock_daily(m: bs.BaoMicro):
    df = m.market_ohlcv(INDEX_CODE, period=PERIOD)
    print(f"stock daily: {df.columns} \n {df}")


def test_csi500_cons(m: bs.BaoUniverse):
    df = m.csi500_cons(date=date(2023, 1, 1))
    print(f"csi500 cons: {df.columns} \n {df}")


if __name__ == "__main__":
    smoke_configure()
    micro = bs.BaoMicro()
    test_stock_daily(micro)
    macro = bs.BaoUniverse()
    # test_csi500_cons(macro)
