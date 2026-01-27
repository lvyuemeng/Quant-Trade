import quant_trade.data.provider.akshare as ak

if __name__ == "__main__":
    pro = ak.AkShareProvider()
    df = pro.stock_universe()
    print("data: {}", df)
