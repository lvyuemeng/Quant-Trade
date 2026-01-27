# Comprehensive Data & Feature Specification Tables

---

## **Table 1: Raw Data Requirements - Micro Level (Stock-Specific)**

| Category | Data Field | Source Table | Update Frequency | Point-in-Time Critical? | Description | Why We Need It |
|----------|-----------|--------------|------------------|------------------------|-------------|----------------|
| **Price & Volume** | open, high, low, close | stock_daily | Daily | No | OHLC prices | Base for returns, momentum |
| | pre_close | stock_daily | Daily | No | Previous day close | Calculate daily returns |
| | volume | stock_daily | Daily | No | Trading volume (shares) | Liquidity analysis |
| | amount | stock_daily | Daily | No | Trading value (CNY) | Money flow analysis |
| | turnover_rate | stock_daily | Daily | No | 換手率 (% of float traded) | **Critical for A-shares**: Retail activity indicator |
| | adj_factor | stock_daily | Daily | No | Adjustment factor for splits/dividends | Backtest price accuracy |
| **Market Microstructure** | is_limit_up | stock_daily | Daily | No | Hit 10% upper limit | Indicates buying pressure, affects tradeability |
| | is_limit_down | stock_daily | Daily | No | Hit 10% lower limit | Selling pressure indicator |
| | is_suspended | stock_daily | Daily | **YES** | Trading halted | Cannot trade suspended stocks |
| **Income Statement** | total_revenue | fundamentals | Quarterly | **YES** | Total revenue | Growth metric |
| | operating_revenue | fundamentals | Quarterly | **YES** | Revenue from operations | Quality check vs total revenue |
| | gross_profit | fundamentals | Quarterly | **YES** | Revenue - COGS | Margin analysis |
| | net_profit | fundamentals | Quarterly | **YES** | Bottom line profit | ROE denominator, valuation |
| | net_profit_parent | fundamentals | Quarterly | **YES** | Profit attributable to shareholders | Preferred for per-share metrics |
| **Balance Sheet** | total_assets | fundamentals | Quarterly | **YES** | All assets | ROA denominator |
| | total_liabilities | fundamentals | Quarterly | **YES** | All liabilities | Debt analysis |
| | total_equity | fundamentals | Quarterly | **YES** | Shareholders' equity | ROE denominator, P/B |
| | total_debt | fundamentals | Quarterly | **YES** | Short + long term debt | Leverage calculation |
| | current_assets | fundamentals | Quarterly | **YES** | Assets < 1 year | Liquidity ratio |
| | current_liabilities | fundamentals | Quarterly | **YES** | Liabilities < 1 year | Liquidity ratio |
| **Cash Flow** | operating_cf | fundamentals | Quarterly | **YES** | Cash from operations | Quality of earnings |
| | investing_cf | fundamentals | Quarterly | **YES** | Cash from investments | Capex analysis |
| | financing_cf | fundamentals | Quarterly | **YES** | Cash from financing | Debt/equity activity |
| **Classification** | sw_l1_code | industry_classification | Monthly | No | Shenwan Level 1 industry code (28 sectors) | **Critical**: For cross-sectional normalization |
| | sw_l1_name | industry_classification | Monthly | No | Industry name (e.g., "银行") | Human-readable |
| | sw_l2_code | industry_classification | Monthly | No | Shenwan Level 2 (104 sub-sectors) | Finer industry analysis |
| **Corporate Actions** | list_date | stock_basic | Static | No | IPO date | Filter recent IPOs |
| | delist_date | stock_basic | Event-driven | No | Delisting date | Survivorship bias handling |

---

## **Table 2: Raw Data Requirements - Macro Level (Market-Wide)**

| Category | Data Field | Source | Update Frequency | Description | Why We Need It | Usage in Model |
|----------|-----------|--------|------------------|-------------|----------------|----------------|
| **Foreign Capital Flow** | northbound_flow | hsgt_top10 | Daily | Net buy from HK (陸股通淨流入) | **Very important for A-shares**: Smart money indicator | Market sentiment feature |
| | northbound_flow_sh | hsgt_top10 | Daily | Shanghai connect flow | Differentiate Shanghai vs Shenzhen | Additional granularity |
| | northbound_flow_sz | hsgt_top10 | Daily | Shenzhen connect flow | Tech/growth bias indicator | Sector rotation signal |
| **liquidity** | shibor_1w | shibor | daily | 1-week interbank rate | short-term liquidity | risk-on/off signal |
| | shibor_3m | shibor | daily | 3-month interbank rate | **primary indicator**: credit availability | discount rate proxy |
| | margin_balance | margin | daily | 融資融券餘額 (margin debt) | retail leverage level | speculation indicator |
| | margin_balance_chg | margin | daily | change in margin balance | increasing = bullish sentiment | momentum confirmation |
| **index valuation** | csi300_pe | index_dailybasic | daily | csi 300 p/e ratio | market valuation level | mean reversion signal |
| | csi300_pb | index_dailybasic | daily | csi 300 p/b ratio | book value anchor | cycle indicator |
| | csi300_dividend_yield | index_dailybasic | daily | csi 300 dividend yield | income opportunity cost | bond vs equity |
| | csi500_pe | index_dailybasic | daily | csi 500 p/e (mid-caps) | small-cap valuation | style rotation |
| **market breadth** | pct_above_ma20 | calculated | daily | % of stocks > 20-day ma | market health | breadth divergence |
| | pct_above_ma200 | calculated | daily | % of stocks > 200-day ma | long-term trend | bull/bear regime |
| | advance_decline_ratio | calculated | daily | (up stocks) / (down stocks) | daily sentiment | overbought/oversold |
| **volatility** | market_turnover | index_dailybasic | daily | total a-share turnover | activity level | risk appetite |
| | csi300_volatility | calculated | daily | 20-day realized vol of csi 300 | market risk | position sizing input |
| | vix_china | index_dailybasic | daily | china vix equivalent (if available) | fear gauge | crisis detection |
| **policy/rates** | 10y_treasury_yield | bond_daily | daily | 10-year government bond yield | risk-free rate | equity risk premium |
| | m2_growth_yoy | macro | monthly | money supply growth | liquidity environment | long-term regime |
| | ppi_yoy | macro | monthly | producer price index yoy | inflation pressure | margin pressure for industrials |
| | pmi_manufacturing | macro | monthly | manufacturing pmi | economic activity | cyclical vs defensive |

---

## **Table 3: Engineered Features - Fundamental (1-Year Anchor)**

| Feature Name | Calculation Formula | Data Requirements | Normalization | Description | Why Important | Horizon Relevance |
|--------------|---------------------|-------------------|---------------|-------------|---------------|-------------------|
| **Quality Metrics** |
| `roe` | net_profit / total_equity | fundamentals | Cross-sectional within industry | Return on equity | **Core quality metric**: Profitability efficiency | 3m-1y |
| `roa` | net_profit / total_assets | fundamentals | Cross-sectional within industry | Return on assets | Asset efficiency, less affected by leverage | 3m-1y |
| `roic` | NOPAT / invested_capital | fundamentals | Cross-sectional within industry | Return on invested capital | True economic return | 1y |
| `gross_margin` | gross_profit / revenue | fundamentals | Cross-sectional within industry | Gross profit margin | Pricing power indicator | 3m-1y |
| `operating_margin` | operating_profit / revenue | fundamentals | Cross-sectional within industry | Operating efficiency | Core business profitability | 3m-1y |
| `net_margin` | net_profit / revenue | fundamentals | Cross-sectional within industry | Bottom-line efficiency | Overall profitability | 3m-1y |
| **Leverage & Safety** |
| `debt_to_asset` | total_debt / total_assets | fundamentals | Cross-sectional within industry | Leverage ratio | **Critical in China**: "Three Red Lines" policy | 6m-1y |
| `debt_to_equity` | total_debt / total_equity | fundamentals | Cross-sectional within industry | Financial leverage | Bankruptcy risk | 6m-1y |
| `current_ratio` | current_assets / current_liabilities | fundamentals | Cross-sectional within industry | Liquidity ratio | Short-term solvency | 3m-6m |
| `interest_coverage` | EBIT / interest_expense | fundamentals | Cross-sectional within industry | Debt service ability | Default risk | 6m-1y |
| **Growth Metrics** |
| `revenue_growth_yoy` | (revenue_q - revenue_q-4) / revenue_q-4 | fundamentals | Cross-sectional within industry | Revenue growth rate | Top-line momentum | 1m-6m |
| `revenue_growth_qoq` | (revenue_q - revenue_q-1) / revenue_q-1 | fundamentals | Cross-sectional within industry | Quarter-on-quarter growth | Recent acceleration | 1m-3m |
| `profit_growth_yoy` | (net_profit_q - net_profit_q-4) / net_profit_q-4 | fundamentals | Cross-sectional within industry | Earnings growth | **Primary growth metric** | 1m-6m |
| `profit_growth_qoq` | (net_profit_q - net_profit_q-1) / net_profit_q-1 | fundamentals | Cross-sectional within industry | Earnings acceleration | Short-term surprise | 1m-3m |
| `asset_growth_yoy` | (total_assets - total_assets_y-1) / total_assets_y-1 | fundamentals | Cross-sectional within industry | Asset expansion rate | Investment intensity | 6m-1y |
| **Cash Flow Quality** |
| `cf_to_profit` | operating_cf / net_profit | fundamentals | Cross-sectional within industry | Cash conversion ratio | **Quality check**: Earnings quality | 3m-1y |
| `fcf_yield` | (operating_cf - capex) / market_cap | fundamentals + market | Cross-sectional within industry | Free cash flow yield | Shareholder return capacity | 6m-1y |
| `capex_to_revenue` | capex / revenue | fundamentals | Cross-sectional within industry | Investment intensity | Growth sustainability | 1y |

---

## **Table 4: Engineered Features - Valuation (Selection Filter)**

| Feature Name | Calculation Formula | Data Requirements | Normalization | Description | Why Important | Horizon Relevance |
|--------------|---------------------|-------------------|---------------|-------------|---------------|-------------------|
| **Price Multiples** |
| `pe_ttm` | market_cap / net_profit_ttm | fundamentals + market | Cross-sectional within industry | P/E ratio (trailing 12m) | **Primary valuation metric** | 1m-1y |
| `pe_forward` | market_cap / consensus_eps_next_year | fundamentals + estimates | Cross-sectional within industry | Forward P/E | Market expectations | 3m-1y |
| `pb` | market_cap / total_equity | fundamentals + market | Cross-sectional within industry | Price-to-book | Asset valuation | 6m-1y |
| `ps` | market_cap / revenue_ttm | fundamentals + market | Cross-sectional within industry | Price-to-sales | Growth stock valuation | 3m-1y |
| `pcf` | market_cap / operating_cf_ttm | fundamentals + market | Cross-sectional within industry | Price-to-cash-flow | Quality-adjusted valuation | 3m-1y |
| **Yield Metrics** |
| `dividend_yield` | annual_dividend / price | fundamentals + market | Cross-sectional within industry | Dividend yield | Income component | 6m-1y |
| `earnings_yield` | 1 / pe_ttm | fundamentals + market | Cross-sectional within industry | Inverse P/E | Bond comparison | 3m-1y |
| `fcf_yield` | free_cash_flow_ttm / market_cap | fundamentals + market | Cross-sectional within industry | FCF yield | True owner earnings | 6m-1y |
| **Relative Valuation** |
| `pe_percentile_5y` | Percentile rank of PE in own 5y history | fundamentals + market | Time-series | Historical PE position | Mean reversion signal | 6m-1y |
| `pb_percentile_5y` | Percentile rank of PB in own 5y history | fundamentals + market | Time-series | Historical PB position | Cyclical bottom/top | 6m-1y |
| `pe_vs_industry` | (pe_stock - pe_industry_median) / pe_industry_std | fundamentals + market | Already normalized | Relative to sector | Peer comparison | 3m-1y |

---

## **Table 5: Engineered Features - Behavioral (1m-3m Trigger)**

| Feature Name | Calculation Formula | Data Requirements | Normalization | Description | Why Important | Horizon Relevance |
|--------------|---------------------|-------------------|---------------|-------------|---------------|-------------------|
| **Turnover Analysis** |
| `turnover_rate` | volume / float_shares | stock_daily | Cross-sectional within industry | 換手率 (% of float) | **Very important in A-shares**: Retail activity | 1m-3m |
| `turnover_ma20` | Mean turnover of last 20 days | stock_daily | Cross-sectional within industry | Average turnover | Baseline for comparison | 1m-3m |
| `turnover_ratio` | turnover_rate / turnover_ma20 | stock_daily | Cross-sectional within industry | Abnormal turnover | **Key signal**: Institutional entry or blow-off top | 1m |
| `turnover_volatility` | Std dev of turnover last 20 days | stock_daily | Cross-sectional within industry | Turnover stability | Regime change detection | 1m-3m |
| `consecutive_high_turnover` | Days with turnover > 2× MA | stock_daily | No normalization | Sustained activity | Bubble warning or accumulation | 1m |
| **Momentum** |
| `return_1m` | Price return last 20 days | stock_daily | Cross-sectional | 1-month momentum | Short-term trend | 1m |
| `return_3m` | Price return last 60 days | stock_daily | Cross-sectional | 3-month momentum | Medium-term trend | 1m-3m |
| `return_6m` | Price return last 120 days | stock_daily | Cross-sectional | 6-month momentum | Longer trend | 3m-6m |
| `return_12m` | Price return last 252 days | stock_daily | Cross-sectional | 12-month momentum | Long-term trend | 6m-1y |
| `excess_return_20d` | return_20d - csi300_return_20d | stock_daily | Cross-sectional | 1-month excess return | **Relative performance** vs market | 1m |
| `excess_return_60d` | return_60d - csi300_return_60d | stock_daily | Cross-sectional | 3-month excess return | Medium-term alpha | 1m-3m |
| `excess_return_120d` | return_120d - csi300_return_120d | stock_daily | Cross-sectional | 6-month excess return | Longer alpha | 3m-6m |
| **Volume-Price Dynamics** |
| `volume_ma20` | Mean volume of last 20 days | stock_daily | Cross-sectional within industry | Average volume | Baseline liquidity | 1m-3m |
| `abnormal_volume` | volume / volume_ma20 | stock_daily | Cross-sectional within industry | Volume surge | **Important**: Smart money or dumb money? | 1m |
| `amount_ma20` | Mean trading value last 20 days | stock_daily | Cross-sectional within industry | Average $ volume | True liquidity measure | 1m-3m |
| `abnormal_amount` | amount / amount_ma20 | stock_daily | Cross-sectional within industry | Money flow surge | Institutional activity | 1m |
| `price_volume_corr` | Correlation(price, volume) last 20d | stock_daily | No normalization | P-V relationship | Healthy rally = positive corr | 1m-3m |
| **Volatility** |
| `volatility_20d` | Std dev of daily returns last 20d | stock_daily | Cross-sectional within industry | Short-term volatility | Risk measure | 1m |
| `volatility_60d` | Std dev of daily returns last 60d | stock_daily | Cross-sectional within industry | Medium-term volatility | Stability indicator | 1m-3m |
| `volatility_ratio` | volatility_20d / volatility_60d | stock_daily | Cross-sectional | Recent vs historical vol | Regime change | 1m |
| `max_drawdown_20d` | Max peak-to-trough in 20 days | stock_daily | Cross-sectional | Recent max loss | Downside risk | 1m |
| **Technical Indicators** |
| `rsi_14` | RSI with 14-day period | stock_daily | No normalization (0-100) | Relative strength index | Overbought/oversold | 1m |
| `macd` | MACD line | stock_daily | Cross-sectional | MACD momentum | Trend confirmation | 1m-3m |
| `macd_signal` | MACD signal line | stock_daily | Cross-sectional | MACD signal | Crossover timing | 1m |
| `dist_to_ma20` | (price - ma20) / ma20 | stock_daily | Cross-sectional | Distance to 20-day MA | Short-term position | 1m |
| `dist_to_ma60` | (price - ma60) / ma60 | stock_daily | Cross-sectional | Distance to 60-day MA | Medium-term position | 1m-3m |
| `dist_to_ma200` | (price - ma200) / ma200 | stock_daily | Cross-sectional | Distance to 200-day MA | Long-term trend | 3m-1y |

---

## **Table 6: Engineered Features - Macro Context (Same for All Stocks)**

| Feature Name | Calculation Formula | Data Requirements | Normalization | Description | Why Important | Usage Pattern |
|--------------|---------------------|-------------------|---------------|-------------|---------------|---------------|
| **Liquidity State** |
| `northbound_flow` | Daily net buy by HK investors | hsgt_top10 | Time-series Z-score (60d) | Foreign capital flow | **Critical for A-shares**: Smart money | Direct feature |
| `northbound_flow_ma5` | 5-day MA of northbound flow | hsgt_top10 | Time-series Z-score (60d) | Smoothed flow | Trend confirmation | Direct feature |
| `northbound_cumulative_20d` | Sum of last 20 days | hsgt_top10 | Time-series Z-score (60d) | Accumulated flow | Persistent trend | Direct feature |
| `shibor_3m` | 3-month interbank rate | shibor | Time-series Z-score (252d) | Funding cost | **Key indicator**: Credit cycle | Direct feature |
| `shibor_3m_change` | Change vs 20 days ago | shibor | Time-series Z-score (60d) | Rate direction | Tightening/easing signal | Direct feature |
| `margin_balance` | Total margin debt | margin | Time-series Z-score (60d) | Retail leverage | Speculation level | Direct feature |
| `margin_balance_pct_chg` | % change last 5 days | margin | Time-series Z-score (60d) | Leverage change | Sentiment shift | Direct feature |
| **Market Valuation** |
| `csi300_pe` | CSI 300 P/E ratio | index_dailybasic | Time-series percentile (5y) | Blue-chip valuation | Market expensive/cheap | Direct feature |
| `csi300_pb` | CSI 300 P/B ratio | index_dailybasic | Time-series percentile (5y) | Book value level | Cycle position | Direct feature |
| `csi500_pe` | CSI 500 P/E ratio | index_dailybasic | Time-series percentile (5y) | Small-cap valuation | Style indicator | Direct feature |
| `pe_spread_large_small` | csi300_pe - csi500_pe | index_dailybasic | Time-series Z-score (252d) | Valuation gap | Style rotation signal | Direct feature |
| **Market Activity** |
| `market_turnover` | Total A-share turnover | Aggregated | Time-series Z-score (60d) | Total activity | Risk appetite | Direct feature |
| `csi300_volatility` | 20d realized vol of CSI 300 | index_daily | Time-series Z-score (60d) | Market risk | VIX equivalent | Direct feature |
| `vix_percentile` | Percentile of volatility in 1y | index_daily | Time-series percentile (252d) | Fear level | Crisis detection | Direct feature |
| **Market Breadth** |
| `pct_above_ma20` | % of all stocks > 20d MA | Calculated daily | No normalization (0-100%) | Short-term health | Breadth divergence | Direct feature |
| `pct_above_ma200` | % of all stocks > 200d MA | Calculated daily | No normalization (0-100%) | Long-term trend | Bull/bear regime | Direct feature |
| `advance_decline_ratio` | (Up stocks) / (Down stocks) | Calculated daily | Time-series Z-score (60d) | Daily sentiment | Overbought/oversold | Direct feature |
| `new_high_low_ratio` | (52w highs) / (52w lows) | Calculated daily | Time-series Z-score (60d) | Extreme sentiment | Euphoria/panic | Direct feature |
| **Policy & Rates** |
| `10y_treasury_yield` | 10-year bond yield | bond_daily | Time-series Z-score (252d) | Risk-free rate | Equity risk premium | Direct feature |
| `yield_spread_10y_1y` | 10y yield - 1y yield | bond_daily | Time-series Z-score (252d) | Yield curve slope | Recession indicator | Direct feature |
| `m2_growth_yoy` | Money supply growth | macro monthly | Time-series Z-score (60m) | Liquidity trend | Long-term regime | Direct feature |
| `pmi_manufacturing` | PMI index | macro monthly | Time-series Z-score (60m) | Economic activity | Cyclical signal | Direct feature |

---

## **Table 7: Feature Processing Pipeline**

| Stage | Step | Input | Output | Purpose | Critical Checks |
|-------|------|-------|--------|---------|----------------|
| **1. Raw Data Validation** | Check completeness | Raw tables | Validation report | Ensure no missing critical dates | - Check for gaps in daily data<br>- Verify fundamental announcements |
| | Handle missing values | Raw data | Cleaned data | Prevent NaN propagation | - Forward fill suspended stocks<br>- Drop if > 20% missing |
| | Detect outliers | Cleaned data | Flagged outliers | Identify data errors | - Flag P/E > 1000 or < 0<br>- Flag turnover > 100% |
| **2. Point-in-Time Alignment** | Align fundamentals | fundamentals table | Aligned fundamentals | **CRITICAL**: Prevent look-ahead bias | - Use announcement_date only<br>- Forward fill until next report |
| | Create lagged features | All tables | Lagged data | Ensure causality | - Returns calculated with 1-day lag<br>- Fundamentals use last announced |
| **3. Feature Calculation** | Compute ratios | Raw data | Basic features | Create fundamental ratios | - Handle division by zero<br>- Winsorize before calculation |
| | Calculate returns | Price data | Momentum features | Price-based signals | - Use adjusted prices<br>- Handle suspensions |
| | Aggregate volumes | Volume data | Liquidity features | Turnover analysis | - Normalize by float<br>- Remove limit-up days |
| **4. Winsorization** | Cap extremes | All features | Winsorized features | **CRITICAL**: Prevent outlier contamination | - Cap at 2.5%/97.5% percentile<br>- Do BEFORE normalization |
| **5. Cross-Sectional Normalization** | Group by industry | Features + industry | Industry groups | Prepare for normalization | - Verify industry codes exist<br>- Handle missing industries |
| | Calculate Z-scores | Industry groups | Normalized features | **CRITICAL**: Industry-neutral factors | - Z = (X - μ_industry) / σ_industry<br>- Check σ > 0 |
| | Handle low-variance | Normalized features | Final features | Avoid division by zero | - If σ < 1e-8, set Z = 0 |
| **6. Missing Value Imputation** | Industry median fill | Normalized features | Imputed features | Handle remaining NaNs | - Use industry median<br>- Flag imputed values |
| **7. Feature Selection** | Remove low-variance | All features | Filtered features | Reduce noise | - Drop if variance < 0.01 |
| | Check multicollinearity | Filtered features | Final feature set | Avoid redundancy | - Drop if correlation > 0.95 |
| **8. Label Generation** | Calculate forward returns | Price data | Return labels | Prediction targets | - Use adjusted prices<br>- Handle suspensions |
| | Winsorize returns | Return labels | Clean labels | Prevent black swan overfitting | - Cap at 1%/99% percentile |
| | Cross-sectional rank | Clean labels | Ranked labels | **For LambdaRank objective** | - Percentile rank within date |
| **9. Train/Test Split** | Time-based split | All data | Train/Val/Test | Prevent leakage | - No random shuffle!<br>- 60% train, 20% val, 20% test |
| **10. Final Validation** | Check feature completeness | Final dataset | Validation report | Quality assurance | - No NaN in features<br>- All dates have labels<br>- Industry codes present |

---

## **Table 8: Data Quality Checks (Critical for A-Shares)**

| Check Type | What to Check | Threshold | Action if Failed | Why Critical |
|------------|---------------|-----------|------------------|---------------|
| **Completeness** | Daily data coverage | < 5% missing days | Re-fetch data | Gaps create look-ahead bias |
| | Fundamental announcement dates | 100% have ann_date | Drop records without date | **MOST CRITICAL**: Point-in-time |
| | Industry classification | > 95% stocks classified | Manual mapping for missing | Normalization requires industry |
| **Consistency** | Price continuity | No jumps > 50% without splits | Verify adj_factor | Detect data errors |
| | Fundamental sign | ROE between -100% and 100% | Cap or investigate | Impossible values |
| | Turnover rate | Between 0% and 100% | Cap at 100% | Physical impossibility |
| **Timeliness** | Announcement delay | < 120 days from period end | Flag late announcers | Delisting risk signal |
| | Data freshness | Daily data within 1 day | Alert if stale | Production requirement |
| **Survivorship Bias** | Delisted stocks included | Check delist_date populated | Include historical data | Avoid overfitting to survivors |
| | ST stocks flagged | Cross-check with name | Verify ST designation | Different price behavior |
| **Tradeability** | Suspension days | Mark is_suspended = 1 | Exclude from portfolio | Cannot trade |
| | Limit up/down | Mark is_limit_up/down = 1 | Exclude from entry | Cannot execute |
| **Cross-Validation** | Fundamental sum check | Assets = Liabilities + Equity | Flag if difference > 1% | Accounting identity |
| | Market cap consistency | Close × shares = market_cap | Recalculate if mismatch | Valuation accuracy |

---

This comprehensive specification ensures you have:
1. **Complete data coverage** (micro + macro)
2. **Proper point-in-time alignment** (no look-ahead bias)
3. **Industry-neutral normalization** (critical for Chinese market)
4. **A-share specific features** (turnover, northbound flow, etc.)
5. **Robust quality checks** (prevent garbage in → garbage out)

Would you like me to proceed with the implementation details for any specific section?