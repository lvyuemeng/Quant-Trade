中国股票指数成份
最新成份
接口: index_stock_cons

目标地址: http://vip.stock.finance.sina.com.cn/corp/view/vII_NewestComponent.php?page=1&indexid=399639

描述: 指定指数的最新成份股票信息, 注意该接口返回的数据有部分是重复会导致数据缺失, 可以调用 ak.index_stock_cons_sina() 获取主流指数数据, 或调用**ak.index_stock_cons_csindex()**获取中证指数网提供的成分数据

输入参数

名称	类型	描述
symbol	str	symbol="000300", 获取沪深 300 最新成份股, 指数代码见 股票指数信息一览表
股票指数信息一览表(可以在 AKShare 中通过如下代码获取本表)

import akshare as ak

index_stock_info_df = ak.index_stock_info()
print(index_stock_info_df)

输出参数

名称	类型	描述
品种代码	object	股票代码
品种名称	object	股票名称
纳入日期	object	成份股纳入日期
接口示例

import akshare as ak

index_stock_cons_df = ak.index_stock_cons(symbol="000300")
print(index_stock_cons_df)
数据示例

       品种代码  品种名称     纳入日期
0    601916  浙商银行  2023-12-11
1    301269  华大九天  2023-12-11
2    688256   寒武纪  2023-12-11
3    600515  海南机场  2023-12-11
4    000999  华润三九  2023-12-11
..      ...   ...         ...
295  000157  中联重科  2005-04-08
296  000069  华侨城A  2005-04-08
297  000063  中兴通讯  2005-04-08
298  000001  深发展A  2005-04-08
299  000002   万科A  2005-04-08
[300 rows x 3 columns]
输出参数-按市场归类

名称	类型	描述
品种代码	object	股票代码
品种名称	object	股票名称
纳入日期	object	成份股纳入日期
接口示例-按市场归类

import akshare as ak

index_stock_cons_df = ak.index_stock_cons(symbol="000300")  # 主要调用 ak.stock_a_code_to_symbol() 来进行转换
index_stock_cons_df['symbol'] = index_stock_cons_df['品种代码'].apply(ak.stock_a_code_to_symbol)
print(index_stock_cons_df)
数据示例-按市场归类

      品种代码  品种名称  纳入日期    symbol
0   000688  国城矿业  2020-12-14  sz000688
1   002409  雅克科技  2020-12-14  sz002409
2   002683  宏大爆破  2020-12-14  sz002683
3   002709  天赐材料  2020-12-14  sz002709
4   002064  华峰氨纶  2020-06-15  sz002064
5   002458  益生股份  2020-06-15  sz002458
6   002812  恩捷股份  2019-12-16  sz002812
7   002128  露天煤业  2019-12-16  sz002128
8   002080  中材科技  2019-12-16  sz002080
9   000708  中信特钢  2019-12-16  sz000708
10  002157  正邦科技  2019-06-17  sz002157
11  000723  美锦能源  2019-06-17  sz000723
12  000629  攀钢钒钛  2019-06-17  sz000629
13  000930  中粮生化  2019-01-02  sz000930
14  000860  顺鑫农业  2019-01-02  sz000860
15  002110  三钢闽光  2018-07-02  sz002110
16  002078  太阳纸业  2018-07-02  sz002078
17  000703  恒逸石化  2018-07-02  sz000703
18  000830  鲁西化工  2018-01-02  sz000830
19  300618  寒锐钴业  2018-01-02  sz300618
20  002408  齐翔腾达  2017-07-03  sz002408
21  300498  温氏股份  2017-01-03  sz300498
22  002299  圣农发展  2016-07-01  sz002299
23  000959  首钢股份  2016-07-01  sz000959
24  000807  云铝股份  2016-07-01  sz000807
25  002221  东华能源  2016-01-04  sz002221
26  000898  鞍钢股份  2014-07-01  sz000898
27  002340   格林美  2014-01-02  sz002340
28  002311  海大集团  2014-01-02  sz002311
29  002385   大北农  2012-07-02  sz002385
30  000876   新希望  2012-01-04  sz000876
31  000998  隆平高科  2011-11-15  sz000998
32  000983  西山煤电  2011-11-15  sz000983
33  000960  锡业股份  2011-11-15  sz000960
34  000878  云南铜业  2011-11-15  sz000878
35  000825  太钢不锈  2011-11-15  sz000825
36  000778  新兴铸管  2011-11-15  sz000778
37  000709  河北钢铁  2011-11-15  sz000709
38  000630  铜陵有色  2011-11-15  sz000630
39  000060  中金岭南  2011-11-15  sz000060
[40 rows x 4 columns]
中证指数成份股
接口: index_stock_cons_csindex

目标地址: http://www.csindex.com.cn/zh-CN/indices/index-detail/000300

描述: 中证指数网站-成份股目录，可以通过 ak.index_csindex_all() 获取所有指数

输入参数

名称	类型	描述
symbol	str	symbol="000300"; 指数代码
输出参数

名称	类型	描述
日期	object	-
指数代码	object	-
指数名称	object	-
指数英文名称	object	-
成分券代码	object	-
成分券名称	object	-
成分券英文名称	object	-
交易所	object	-
交易所英文名称	object	-
示例代码

import akshare as ak

index_stock_cons_csindex_df = ak.index_stock_cons_csindex(symbol="000300")
print(index_stock_cons_csindex_df)
数据示例

             日期    指数代码  ...      交易所                  交易所英文名称
0    2024-01-04  000300  ...  深圳证券交易所  Shenzhen Stock Exchange
1    2024-01-04  000300  ...  深圳证券交易所  Shenzhen Stock Exchange
2    2024-01-04  000300  ...  深圳证券交易所  Shenzhen Stock Exchange
3    2024-01-04  000300  ...  深圳证券交易所  Shenzhen Stock Exchange
4    2024-01-04  000300  ...  深圳证券交易所  Shenzhen Stock Exchange
..          ...     ...  ...      ...                      ...
295  2024-01-04  000300  ...  上海证券交易所  Shanghai Stock Exchange
296  2024-01-04  000300  ...  上海证券交易所  Shanghai Stock Exchange
297  2024-01-04  000300  ...  上海证券交易所  Shanghai Stock Exchange
298  2024-01-04  000300  ...  上海证券交易所  Shanghai Stock Exchange
299  2024-01-04  000300  ...  上海证券交易所  Shanghai Stock Exchange
[300 rows x 9 columns]
中证指数成份股权重
接口: index_stock_cons_weight_csindex

目标地址: http://www.csindex.com.cn/zh-CN/indices/index-detail/000300

描述: 中证指数网站-成份股权重

输入参数

名称	类型	描述
symbol	str	symbol="000300"; 指数代码
输出参数

名称	类型	描述
日期	object	-
指数代码	object	-
指数名称	object	-
指数英文名称	object	-
成分券代码	object	-
成分券名称	object	-
成分券英文名称	object	-
交易所	object	-
交易所英文名称	object	-
权重	float64	注意单位: %
示例代码

import akshare as ak

index_stock_cons_weight_csindex_df = ak.index_stock_cons_weight_csindex(symbol="000300")
print(index_stock_cons_weight_csindex_df)
数据示例

     日期    指数代码   指数名称  ...     交易所                交易所英文名称     权重
0    2023-12-29  000300  沪深300  ...  深圳证券交易所  Shenzhen Stock Exchange  0.524
1    2023-12-29  000300  沪深300  ...  深圳证券交易所  Shenzhen Stock Exchange  0.410
2    2023-12-29  000300  沪深300  ...  深圳证券交易所  Shenzhen Stock Exchange  0.486
3    2023-12-29  000300  沪深300  ...  深圳证券交易所  Shenzhen Stock Exchange  0.088
4    2023-12-29  000300  沪深300  ...  深圳证券交易所  Shenzhen Stock Exchange  0.465
..          ...     ...    ...  ...      ...                      ...    ...
295  2023-12-29  000300  沪深300  ...  上海证券交易所  Shanghai Stock Exchange  0.074
296  2023-12-29  000300  沪深300  ...  上海证券交易所  Shanghai Stock Exchange  0.136
297  2023-12-29  000300  沪深300  ...  上海证券交易所  Shanghai Stock Exchange  0.063
298  2023-12-29  000300  沪深300  ...  上海证券交易所  Shanghai Stock Exchange  0.178
299  2023-12-29  000300  沪深300  ...  上海证券交易所  Shanghai Stock Exchange  0.602
[300 rows x 10 columns]