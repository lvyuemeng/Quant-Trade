# 数据字段规范说明 / Field Mapping Specification

本文档用于说明各类金融数据在本项目中的 统一字段命名规范。
不同数据提供方（AkShare / EM / Wind / TuShare 等）返回的原始字段名称可能存在差异，
本规范仅约束"目标字段语义"，不约束来源字段命名。

This document defines the unified field naming conventions for various financial data within this project.
The original field names returned by different data providers (AkShare / EM / Wind / TuShare, etc.) may vary.
This specification constrains only the "target field semantics", not the naming of the source fields.

---

## 1. 设计原则 / Design Principles

* **目标字段唯一且稳定 / Target Fields are Unique and Stable**
  * 每一个字段代表明确、唯一的经济含义
  * Each field represents a clear and unique economic meaning.

* **来源字段不固定 / Source Fields are Not Fixed**
  * 原始字段名可能随数据源变化，不作为规范约束对象
  * Original field names may change with the data source and are not constrained by this specification.

* **字段命名采用 snake_case / Field Naming Uses snake_case English**
  * 用于程序处理、建模与跨源对齐
  * For program processing, modeling, and cross-source alignment.

* **允许组合与复用 / Composition and Reuse are Allowed**
  * 不同数据集可复用同一字段子集（Identifiers / Date / Cashflow / Rates 等）
  * Different datasets can reuse the same subset of fields (Identifiers / Date / Cashflow / Rates, etc.).

---

## 2. 通用字段（Identifiers） / Common Fields (Identifiers)

* **ts_code**
  * 含义：股票代码 / Stock Code
  * 类型：string
  * 备注：不含交易所后缀 / Without exchange suffix

* **ts_code_suffix**
  * 含义：股票代码（含交易所） / Stock Code (with Exchange)
  * 类型：string
  * 备注：如 600519.SH / e.g., 600519.SH

* **name**
  * 含义：股票简称 / Stock Short Name
  * 类型：string

---

## 3. 时间字段（Date） / Date Fields (Date)

* **date**
  * 含义：数据日期 / Data Date
  * 类型：date
  * 备注：交易日 / 统计日 / Trading day / Statistical day

* **announcement_date**
  * 含义：公告日期 / Announcement Date
  * 类型：date
  * 备注：财报或事件披露日 / Financial report or event disclosure date

* **join_date**
  * 含义：纳入时间 / Inclusion Date
  * 类型：date
  * 备注：指数 / 行业纳入日 / Index / Industry inclusion date

---

## 4. 现金流量数据（Cashflow Statement） / Cash Flow Data (Cashflow Statement)

### 4.1 净现金流（Total Cashflow） / Net Cash Flow (Total Cashflow)

* **net_cashflow**
  * 含义：净现金流量 / Net Cash Flow
  * 单位：元 / CNY
  * 备注：全部现金流合计 / Total cash flow summation

* **net_cashflow_yoy**
  * 含义：净现金流同比增长 / Net Cash Flow YoY Growth
  * 单位：%

### 4.2 经营性现金流（Operating Cashflow） / Operating Cash Flow (Operating Cashflow)

* **cfo**
  * 含义：经营活动现金流量净额 / Net Cash Flow from Operating Activities
  * 单位：元 / CNY

* **cfo_share**
  * 含义：经营性现金流占比 / Operating Cash Flow Share
  * 单位：%
  * 备注：占净现金流比例 / Proportion of net cash flow

### 4.3 投资性现金流（Investing Cashflow） / Investing Cash Flow (Investing Cashflow)

* **cfi**
  * 含义：投资活动现金流量净额 / Net Cash Flow from Investing Activities
  * 单位：元 / CNY

* **cfi_share**
  * 含义：投资性现金流占比 / Investing Cash Flow Share
  * 单位：%

### 4.4 融资性现金流（Financing Cashflow） / Financing Cash Flow (Financing Cashflow)

* **cff**
  * 含义：融资活动现金流量净额 / Net Cash Flow from Financing Activities
  * 单位：元 / CNY

* **cff_share**
  * 含义：融资性现金流占比 / Financing Cash Flow Share
  * 单位：%

---

## 5. 资金流向数据（Capital Flow） / Capital Flow Data (Capital Flow)

* **net_buy**
  * 含义：当日成交净买额 / Daily Net Buy Amount
  * 单位：元 / CNY

* **fund_inflow**
  * 含义：当日资金流入 / Daily Fund Inflow
  * 单位：元 / CNY

* **cum_net_buy**
  * 含义：历史累计净买额 / Cumulative Net Buy
  * 单位：元 / CNY

---

## 6. 融资融券数据（Margin Trading） / Margin Trading Data (Margin Trading)

* **margin_buy_amount**
  * 含义：融资买入额 / Margin Purchase Amount
  * 单位：元 / CNY

* **margin_balance**
  * 含义：融资余额 / Margin Balance
  * 单位：元 / CNY

* **short_sell_volume**
  * 含义：融券卖出量 / Short Sell Volume
  * 单位：股 / Shares

* **short_balance**
  * 含义：融券余额 / Short Balance
  * 单位：元 / CNY

* **total_margin_balance**
  * 含义：融资融券余额 / Total Margin Balance
  * 单位：元 / CNY

---

## 7. 利率数据（Money Market Rates） / Interest Rate Data (Money Market Rates)

利率字段统一采用：期限_rate / 期限_change 结构。
Interest rate fields uniformly adopt the structure: `tenor_rate` / `tenor_change`.

* **ON_rate**
  * 含义：隔夜利率 / Overnight Rate
  * 单位：%

* **ON_change**
  * 含义：隔夜利率涨跌幅 / Overnight Rate Change
  * 单位：%

* **1W_rate**
  * 含义：1 周利率 / 1 Week Rate
  * 单位：%

* **1W_change**
  * 含义：1 周利率涨跌幅 / 1 Week Rate Change
  * 单位：%

* **1M_rate**
  * 含义：1 月利率 / 1 Month Rate
  * 单位：%

* **1M_change**
  * 含义：1 月利率涨跌幅 / 1 Month Rate Change
  * 单位：%

* **3M_rate**
  * 含义：3 月利率 / 3 Month Rate
  * 单位：%

* **3M_change**
  * 含义：3 月利率涨跌幅 / 3 Month Rate Change
  * 单位：%

* **6M_rate**
  * 含义：6 月利率 / 6 Month Rate
  * 单位：%

* **6M_change**
  * 含义：6 月利率涨跌幅 / 6 Month Rate Change
  * 单位：%

* **9M_rate**
  * 含义：9 月利率 / 9 Month Rate
  * 单位：%

* **9M_change**
  * 含义：9 月利率涨跌幅 / 9 Month Rate Change
  * 单位：%

* **1Y_rate**
  * 含义：1 年期利率 / 1 Year Rate
  * 单位：%

* **1Y_change**
  * 含义：1 年期利率涨跌幅 / 1 Year Rate Change
  * 单位：%

---

## 8. 行业分类数据（Industry Classification） / Industry Classification Data (Industry Classification)

* **sw_l1_name**
  * 含义：申万一级行业 / SW Level 1 Industry

* **sw_l2_name**
  * 含义：申万二级行业 / SW Level 2 Industry

* **sw_l3_name**
  * 含义：申万三级行业 / SW Level 3 Industry

* **sw_l(1/2/3)_code**
  * 含义：申万一/二/三级行业 / SW Level 1/2/3 Industry Code

---

## 价格成交量原始字段（Price-Volume Raw Fields）
* **open**
  * 含义：开盘价 / Open Price
  * 单位：元 / CNY

* **high**
  * 含义：最高价 / High Price
  * 单位：元 / CNY

* **low**
  * 含义：最低价 / Low Price
  * 单位：元 / CNY

* **close**
  * 含义：收盘价 / Close Price
  * 单位：元 / CNY

* **volume**
  * 含义：成交量 / Volume
  * 单位：股 / Shares

* **amount**
  * 含义：成交额 / Amount
  * 单位：元 / CNY

## 1. 动量特征（Momentum Features）

### 收益率特征（Return Features）
* **ret_1**
  * 含义：1日收益率 / 1-Day Return
  * 单位：%

* **ret_4**
  * 含义：4日收益率 / 4-Day Return
  * 单位：%

* **ret_12**
  * 含义：12日收益率 / 12-Day Return
  * 单位：%

* **ret_1m**
  * 含义：1月收益率（22日） / 1-Month Return (22 days)
  * 单位：%

* **ret_1q**
  * 含义：1季度收益率（65日） / 1-Quarter Return (65 days)
  * 单位：%

* **ret_1y**
  * 含义：1年收益率（261日） / 1-Year Return (261 days)
  * 单位：%

* **roc_4**
  * 含义：4日价格变化率 / 4-Day Rate of Change
  * 单位：%

* **roc_12**
  * 含义：12日价格变化率 / 12-Day Rate of Change
  * 单位：%

### 移动平均特征（Moving Average Features）
* **ma_4**
  * 含义：4日移动平均价 / 4-Day Moving Average
  * 单位：元 / CNY

* **ma_12**
  * 含义：12日移动平均价 / 12-Day Moving Average
  * 单位：元 / CNY

* **ma_24**
  * 含义：24日移动平均价 / 24-Day Moving Average
  * 单位：元 / CNY

* **ma12_dev**
  * 含义：价格对12日移动平均偏离度 / Deviation from 12-Day MA
  * 单位：%
  * 备注：裁剪范围[-10, 10] / Clipped to [-10, 10]

* **ma24_dev**
  * 含义：价格对24日移动平均偏离度 / Deviation from 24-Day MA
  * 单位：%
  * 备注：裁剪范围[-10, 10] / Clipped to [-10, 10]

* **above_ma12**
  * 含义：是否在12日移动平均线上 / Above 12-Day MA
  * 类型：Int8
  * 取值：1（是）/ 0（否）

* **ma_cross_up**
  * 含义：4日线上穿12日线 / 4-Day MA Crosses Above 12-Day MA
  * 类型：Int8
  * 取值：1（是）/ 0（否）

### 通道特征（Channel Features）
* **ch_high**
  * 含义：20日最高价通道 / 20-Day High Channel
  * 单位：元 / CNY

* **ch_low**
  * 含义：20日最低价通道 / 20-Day Low Channel
  * 单位：元 / CNY

* **ch_pos**
  * 含义：通道内相对位置 / Channel Position
  * 单位：比率 / Ratio
  * 备注：0-1之间，裁剪范围[0, 1] / Between 0-1, clipped to [0, 1]

## 2. 波动率特征（Volatility Features）

### 波动率估计量（Volatility Estimators）
* **vola_parkinson**
  * 含义：Parkinson波动率估计（基于高低价） / Parkinson Volatility Estimator (High-Low based)
  * 单位：%
  * 备注：20日滚动平均，至少5个样本 / 20-day rolling average, min 5 samples

* **vola_gk**
  * 含义：Garman-Klass波动率估计（基于OHLC） / Garman-Klass Volatility Estimator (OHLC based)
  * 单位：%
  * 备注：20日滚动平均，至少5个样本 / 20-day rolling average, min 5 samples

* **vola_20**
  * 含义：20日已实现波动率 / 20-Day Realized Volatility
  * 单位：%
  * 备注：对数收益标准差，至少10个样本 / Log return standard deviation, min 10 samples

* **vola_60**
  * 含义：60日已实现波动率 / 60-Day Realized Volatility
  * 单位：%
  * 备注：对数收益标准差，至少20个样本 / Log return standard deviation, min 20 samples

* **vola_regime_ratio**
  * 含义：波动率区间比率（20日/60日） / Volatility Regime Ratio (20-day / 60-day)
  * 单位：比率 / Ratio
  * 备注：裁剪范围[0.1, 10] / Clipped to [0.1, 10]

### 真实波幅特征（ATR Features）
* **tr**
  * 含义：真实波幅 / True Range
  * 单位：元 / CNY

* **atr_pct**
  * 含义：14日平均真实波幅占比 / 14-Day Average True Range Percentage
  * 单位：%
  * 备注：相对收盘价，裁剪范围[0, 1] / Relative to close price, clipped to [0, 1]

## 3. 成交量特征（Volume Features）

### 成交量偏离特征（Volume Deviation Features）
* **vol_dev20**
  * 含义：成交量对20日均值偏离度 / Volume Deviation from 20-Day Average
  * 单位：%
  * 备注：裁剪范围[-5, 5] / Clipped to [-5, 5]

* **vol_dev60**
  * 含义：成交量对60日均值偏离度 / Volume Deviation from 60-Day Average
  * 单位：%
  * 备注：裁剪范围[-5, 5] / Clipped to [-5, 5]

* **vol_spike**
  * 含义：成交量是否异常放大（超过2倍20日均量） / Volume Spike (Over 2x 20-Day Average)
  * 类型：Int8
  * 取值：1（是）/ 0（否）

### 均价特征（Average Price Features）
* **vwap_dev_1d**
  * 含义：收盘价对日VWAP偏离度 / Close Price Deviation from Daily VWAP
  * 单位：%
  * 备注：裁剪范围[-0.5, 0.5] / Clipped to [-0.5, 0.5]

* **avg_trade_price**
  * 含义：平均成交价（成交额/成交量） / Average Trade Price (Amount/Volume)
  * 单位：元 / CNY

## 4. 结构特征（Structure Features）

### 方差比率特征（Variance Ratio Features）
* **vr_2**
  * 含义：方差比率（2日/1日） / Variance Ratio (2-day / 1-day)
  * 单位：比率 / Ratio
  * 备注：60日窗口，至少20个样本，裁剪范围[0.1, 5] / 60-day window, min 20 samples, clipped to [0.1, 5]

### 自相关特征（Autocorrelation Features）
* **ac_1**
  * 含义：1阶自相关系数 / 1st-order Autocorrelation
  * 单位：相关系数 / Correlation Coefficient
  * 备注：60日滚动相关，至少20个样本 / 60-day rolling correlation, min 20 samples

* **ac_5**
  * 含义：5阶自相关系数 / 5th-order Autocorrelation
  * 单位：相关系数 / Correlation Coefficient
  * 备注：60日滚动相关，至少20个样本 / 60-day rolling correlation, min 20 samples

### 高低位置特征（High-Low Position Features）
* **near_high**
  * 含义：接近50日最高价程度 / Proximity to 50-Day High
  * 单位：%
  * 备注：裁剪范围[-1, 0.5] / Clipped to [-1, 0.5]

* **near_low**
  * 含义：接近50日最低价程度 / Proximity to 50-Day Low
  * 单位：%
  * 备注：裁剪范围[-0.5, 1] / Clipped to [-0.5, 1]

* **gap**
  * 含义：开盘缺口 / Opening Gap
  * 单位：%
  * 备注：相对前收盘价，裁剪范围[-0.2, 0.2] / Relative to previous close, clipped to [-0.2, 0.2]

## 5. 基础面原始字段（Fundamental Raw Fields）

### Composition

* **eps_ttm**
  * 含义：每股收益（TTM） / Earnings Per Share (TTM)
  * 单位：元 / CNY
  * 公式：归属母公司股东的净利润TTM/最新总股本

* **total_shares**
  * 含义：总股本 / Total Shares Outstanding
  * 单位：股 / Shares
  
* **float_shares**
  * 含义：流通股本 / Floating Shares
  * 单位：股 / Shares

* **total_equity**
  * 含义：股东权益 / Total Equity
  * 单位：元 / CNY

### Profit/Asset

* **gross_profit**
  * 含义：毛利润 / Gross Profit
  * 单位：元 / CNY

* **total_revenue**
  * 含义：营业收入 / Total Revenue
  * 单位：元 / CNY

* **operating_profit**
  * 含义：营业利润 / Operating Profit
  * 单位：元 / CNY

* **operating_cost**
  * 含义：营业成本 / Operating Cost
  * 单位：元 / CNY

* **net_profit**
  * 含义：净利润 / Net Profit
  * 单位：元 / CNY

* **total_assets**
  * 含义：总资产 / Total Assets
  * 单位：元 / CNY

* **total_debts**
  * 含义：总负债 / Total Debts
  * 单位：元 / CNY

* **cash**
  * 含义：现金 / Cash
  * 单位：元 / CNY

* **accounts_receivable**
  * 含义：应收账款 / Accounts Receivable
  * 单位：元 / CNY

* **inventory**
  * 含义：存货 / Inventory
  * 单位：元 / CNY

* **accounts_payable**
  * 含义：应付账款 / Accounts Payable
  * 单位：元 / CNY

* **advance_receipts**
  * 含义：预收款项 / Advance Receipts
  * 单位：元 / CNY

* **finance_cost**
  * 含义：财务费用 / Finance Cost
  * 单位：元 / CNY

* **cfo**
  * 含义：经营活动现金流量净额 / Cash Flow from Operations
  * 单位：元 / CNY

* **cfi**
  * 含义：投资活动现金流量净额 / Cash Flow from Investing
  * 单位：元 / CNY

## Growth Metric

* **gross_margin**
  * 含义：毛利率 / Gross Margin
  * 单位：%
  * 公式：gross_profit / total_revenue

* **operating_margin**
  * 含义：营业利润率 / Operating Margin
  * 单位：%
  * 公式：operating_profit / total_revenue

* **net_margin**
  * 含义：净利率 / Net Margin
  * 单位：%
  * 公式：net_profit / total_revenue

* **net_profit_yoy**
  * 含义：净利润同比增长 / Net Profit YoY Growth
  * 单位：%

* **net_profit_parent_yoy**
  * 含义：归属母公司股东净利润同比增长率 / Net Profit Attributable to Parent YoY Growth Rate
  * 单位：%
  * 公式：(本期归属母公司股东净利润-上年同期归属母公司股东净利润)/上年同期归属母公司股东净利润的绝对值*100%

* **eps_basic_yoy**
  * 含义：基本每股收益同比增长率 / Basic EPS YoY Growth Rate
  * 单位：%
  * 公式：(本期基本每股收益-上年同期基本每股收益)/上年同期基本每股收益的绝对值*100%

* **total_revenue_yoy**
  * 含义：营业收入同比增长 / Total Revenue YoY Growth
  * 单位：%

* **total_assets_yoy**
  * 含义：总资产同比增长 / Total Assets YoY Growth
  * 单位：%

* **total_debts_yoy**
  * 含义：总负债同比增长 / Total Debts YoY Growth
  * 单位：%

* **total_equity_yoy**
  * 含义：净资产同比增长率 / Equity YoY Growth Rate
  * 单位：%
  * 公式：(本期净资产-上年同期净资产)/上年同期净资产的绝对值*100%

* **roe**
  * 含义：净资产收益率 / Return on Equity
  * 单位：%
  * 公式：net_profit / total_equity
  * 备注：分母为负时返回null / Returns null when denominator is negative

* **roa**
  * 含义：总资产收益率 / Return on Assets
  * 单位：%
  * 公式：net_profit / total_assets
  * 备注：分母为负时返回null / Returns null when denominator is negative

## 账款周转 (Turnover)

* **receivable_turnover**
  * 含义：应收账款周转率 / Receivables Turnover Ratio
  * 单位：次 / Times
  * 公式：营业收入/[(期初应收票据及应收账款净额+期末应收票据及应收账款净额)/2]

* **receivable_turnover_days**
  * 含义：应收账款周转天数 / Receivables Turnover Days
  * 单位：天 / Days
  * 公式：季报天数/应收账款周转率

* **inventory_turnover**
  * 含义：存货周转率 / Inventory Turnover Ratio
  * 单位：次 / Times
  * 公式：营业成本/[(期初存货净额+期末存货净额)/2]

* **inventory_turnover_days**
  * 含义：存货周转天数 / Inventory Turnover Days
  * 单位：天 / Days
  * 公式：季报天数/存货周转率

* **current_assets_turnover**
  * 含义：流动资产周转率 / Current Assets Turnover Ratio
  * 单位：次 / Times
  * 公式：营业总收入/[(期初流动资产+期末流动资产)/2]

* **total_assets_turnover**
  * 含义：资产周转率 / Asset Turnover
  * 单位：次 / Times
  * 公式：total_revenue / total_assets

## Balance

* **current_assets_to_total_assets**
  * 含义：流动资产占总资产比例 / Current Assets to Total Assets Ratio
  * 单位：比率 / Ratio
  * 公式：流动资产除以总资产

* **non_current_assets_to_total_assets**
  * 含义：非流动资产占总资产比例 / Non-current Assets to Total Assets Ratio
  * 单位：比率 / Ratio
  * 公式：非流动资产除以总资产

* **tangible_assets_to_total_assets**
  * 含义：有形资产占总资产比例 / Tangible Assets to Total Assets Ratio
  * 单位：比率 / Ratio
  * 公式：有形资产除以总资产

* **ebit_to_interest**
  * 含义：已获利息倍数 / Times Interest Earned
  * 单位：倍 / Times
  * 公式：息税前利润/利息费用

* **cfo_to_revenue**
  * 含义：经营活动现金流占营业收入比例 / CFO to Revenue Ratio
  * 单位：比率 / Ratio
  * 公式：经营活动产生的现金流量净额除以营业收入

* **cfo_to_net_profit**
  * 含义：经营性现金净流量占净利润比例 / CFO to Net Profit Ratio
  * 单位：比率 / Ratio
  * 公式：经营性现金净流量除以净利润

* **cfo_to_total_revenue**
  * 含义：经营性现金净流量占营业总收入比例 / CFO to Total Revenue Ratio
  * 单位：比率 / Ratio
  * 公式：经营性现金净流量除以营业总收入

## 杜邦分析指标（DuPont Analysis Metrics）

### 杜邦分解
* **roe**
  * 含义：净资产收益率（杜邦） / Return on Equity (DuPont)
  * 单位：%
  * 公式：归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%

* **assets_to_equity**
  * 含义：权益乘数 / Equity Multiplier
  * 单位：倍 / Times
  * 公式：平均总资产/平均归属于母公司的股东权益

* **total_assets_turnover**
  * 含义：总资产周转率（杜邦） / Asset Turnover (DuPont)
  * 单位：次 / Times
  * 公式：营业总收入/[(期初资产总额+期末资产总额)/2]

* **parent_profit_ratio**
  * 含义：归属母公司股东净利润占比 / Parent Company Profit Ratio
  * 单位：比率 / Ratio
  * 公式：归属母公司股东的净利润/净利润

* **net_margin**
  * 含义：净利率 / Net Profit Margin
  * 单位：比率 / Ratio
  * 公式：净利润/营业总收入

* **tax_burden**
  * 含义：税收负担率 / Tax Burden Ratio
  * 单位：比率 / Ratio
  * 公式：净利润/利润总额

* **interest_burden**
  * 含义：利息负担率 / Interest Burden Ratio
  * 单位：比率 / Ratio
  * 公式：利润总额/息税前利润

* **ebit_margin**
  * 含义：息税前利润率 / EBIT Margin
  * 单位：比率 / Ratio
  * 公式：息税前利润/营业总收入

## 组合使用说明（Composable Notation） / Usage for Composition (Composable Notation)

在代码中，字段规范以"语义模块"组合使用，而非依赖具体 API。
In code, the field specification is used by composing "semantic modules", rather than relying on specific APIs.

**示例（概念） / Examples (Conceptual):**

* **季度现金流量 / Quarterly Cash Flow**
  = Identifiers + Date + Cashflow

* **日度市场资金流 / Daily Market Capital Flow**
  = Date + Capital Flow

* **货币市场利率 / Money Market Rates**
  = Date + Rates

> 字段组合仅决定输出 schema，原始字段名由各数据源自行适配，不纳入本规范。
> Field composition only determines the output schema. The adaptation of original field names is handled by each data source and is not part of this specification.