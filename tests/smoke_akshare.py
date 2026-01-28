import quant_trade.provider.akshare as ak

INDEX_CODE = "000001"
PERIOD = "daily"

SHEET_YEAR = 2021
SHEET_QUATER = 1


def test_market_daily(m: ak.AkShareMicro):
    df = m.market_ohlcv(INDEX_CODE, period=PERIOD)
    print(f"market daily: {df}")


def test_quarter_income(m: ak.AkShareMicro):
    df = m.quarterly_income_statement(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter income: {df.columns} \n {df}")


def test_quarter_balance(m: ak.AkShareMicro):
    df = m.quarterly_balance_sheet(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter balance: {df.columns} \n {df}")


def test_quarter_cashflow(m: ak.AkShareMicro):
    df = m.quarterly_cashflow_statement(year=SHEET_YEAR, quarter=SHEET_QUATER)
    print(f"quarter cashflow: {df.columns} \n {df}")


if __name__ == "__main__":
    pro = ak.AkShareMicro()
    # test_market_daily(pro)
    # test_quarter_income(pro)
    # test_quarter_balance(pro)
    test_quarter_cashflow(pro)
