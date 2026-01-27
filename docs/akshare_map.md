# Raw tables ↔ AkShare API mapping

This document maps the **raw tables referenced in `docs/table.md`** to concrete **AkShare** APIs (preferring the APIs already documented under `docs/akshare/*` and `docs/pause/*`).

It also clarifies:
- required input formats (symbol/date)
- what fields we can fetch directly vs derive
- what’s still missing / needs a decision

---

## Conventions

### Symbols
AkShare endpoints in this repo use **mixed symbol formats**:
- Many A-share endpoints use **6-digit code**: `"000001"`
- Some endpoints use **exchange-prefixed** symbols: `"SH601127"`, `"SZ000895"` (e.g. Xueqiu / some Eastmoney pages)

In our codebase we currently use:
- `ts_code`: 6-digit string (no exchange prefix)

### Dates
- Daily market data: `start_date/end_date` are `YYYYMMDD`
- Financial statements (eastmoney season): `date` is quarter-end `YYYY0331|YYYY0630|YYYY0930|YYYY1231`

### Point-in-time fundamentals (critical)
For quarterly fundamentals, avoid look-ahead bias:
- treat **公告日期** (announcement date) as `announcement_date`
- forward-fill each item until the next announcement

---

## Table 1: Micro-level raw data

### 1) `stock_daily` (OHLCV + turnover, etc.)
**Primary API (documented):** `ak.stock_zh_a_hist`
- Doc: `docs/akshare/stock_zh_a_hist.md`
- Example:
  - `ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date="20231231", adjust="")`

**Direct field mapping (from `stock_zh_a_hist` output columns):**
| target field | AkShare column | notes |
|---|---|---|
| `date` | `日期` | parse as datetime/date |
| `ts_code` | `股票代码` | already 6-digit |
| `open/high/low/close` | `开盘/最高/最低/收盘` | — |
| `volume` | `成交量` | **unit = 手** |
| `amount` | `成交额` | **unit = 元** |
| `turnover_rate` | `换手率` | unit = % |
| `pct_chg` | `涨跌幅` | unit = % |
| `change` | `涨跌额` | unit = 元 |

**Derived fields needed by `docs/table.md`:**
- `pre_close`:
  - derive via `pre_close = close - change` (same-day arithmetic)
  - OR via time series shift: `pre_close = close.shift(1)` (preferred for consistency)
- `is_suspended`:
  - AkShare doesn’t emit a suspension flag here; a common proxy is `volume == 0`
- `is_limit_up/is_limit_down`:
  - not directly provided; infer from `pct_chg` with China limit rules (10% for most, 20% for some boards). This needs a **board-aware** threshold later.
- `adj_factor`:
  - **gap:** `stock_zh_a_hist` supports `adjust` output (qfq/hfq) but does **not** return an explicit adjustment factor column in this documented schema.
  - proposal for now: store raw unadjusted close + optionally store an adjusted close series (via `adjust="hfq"`), and defer explicit `adj_factor` until we choose/identify a dedicated endpoint.

**Related API (documented):** `ak.stock_zh_a_spot_em`
- Doc: `docs/akshare/stock_zh_a_spot_em.md`
- Use cases:
  - get current **universe snapshot** (`代码`, `名称`, `换手率`, etc.)
  - use as an initial code list for backfilling

---

### 2) `stock_basic` (list/delist metadata)
**List date candidate (documented):** `ak.stock_individual_basic_info_xq`
- Doc: `docs/akshare/stock_individual_basic_info_xq.md`
- Returns `listed_date` (ms timestamp) for a symbol like `"SH601127"`
- **Caveat:** may require token/cookies; AkShare signature includes `token=None`.

**Delist date candidates (listed in `docs/akshare/akentry.md`, not documented here):**
- `stock_info_sh_delist`
- `stock_info_sz_delist`

**Notes:**
- Our current `DataAcquisition.fetch_stock_universe()` uses spot data + ST/volume filters; IPO-age filtering is explicitly a TODO in code.

---

### 3) `fundamentals` (income/balance/cashflow)
We can assemble quarterly fundamentals from the **Eastmoney statement batch APIs** (documented under `docs/pause/*`).

**Income statement (documented):** `ak.stock_lrb_em(date="YYYY0331")`
- Doc: `docs/pause/stock_lrb_em.md`
- Key fields:
  - `股票代码` (ts_code)
  - `净利润` (net_profit)
  - `营业总收入` (total_revenue)
  - `公告日期` (announcement_date)

**Balance sheet (documented):** `ak.stock_zcfz_em(date="YYYY0331")`
- Doc: `docs/pause/stock_zcfz_em.md`
- Key fields:
  - `资产-总资产` (total_assets)
  - `负债-总负债` (total_liabilities)
  - `股东权益合计` (total_equity)
  - `公告日期` (announcement_date)

**Cashflow statement (documented):** `ak.stock_xjll_em(date="YYYY0331")`
- Doc: `docs/pause/stock_xjll_em.md`
- Key fields:
  - `经营性现金流-现金流量净额` (operating_cf)
  - `投资性现金流-现金流量净额` (investing_cf)
  - `融资性现金流-现金流量净额` (financing_cf)
  - `公告日期` (announcement_date)

**Additional fundamentals (documented, per-stock):** `ak.stock_financial_abstract(symbol="600004")`
- Doc: `docs/akshare/stock_financial_abstract.md`
- Good for cross-checking (ROE/ROA/margins, etc.) but schema is wide and period-indexed.

**Additional fundamentals (used in code, not documented here):**
- `ak.stock_financial_analysis_indicator(symbol=...)` (currently called in `src/data/acquisition.py`)

---

### 4) `industry_classification` (Shenwan L1/L2)
**Documented APIs (repo-local):**
- `ak.sw_index_first_info()`
  - Doc: `docs/akshare/sw_index_first_info.md`
- `ak.sw_index_second_info()`
  - Doc: `docs/akshare/sw_index_second_info.md`
- `ak.sw_index_third_info()`
  - Doc: `docs/akshare/sw_index_third_info.md`

**Practical mapping approach:**
1) Use `sw_index_third_cons` to get per-stock rows, which already include:
   - `申万1级` (name)
   - `申万2级` (name)
   - `申万3级` (name)
   - `股票代码` like `600313.SH` / `000713.SZ`
2) Strip exchange suffix to get `ts_code`.
3) Use `sw_index_first_info` / `sw_index_second_info` to map **name → code** for L1/L2.
4) Persist monthly as `industry_classification(date, ts_code, sw_l1_code/sw_l1_name, sw_l2_code/sw_l2_name, ...)`.

---

## Table 2: Macro-level raw data

### 1) `hsgt_top10` / northbound flow (daily)
**Closest documented API:** `ak.stock_hsgt_fund_flow_summary_em()`
- Doc: `docs/akshare/stock_hsgt_fund_flow_summary_em.md`
- Output includes `成交净买额` and `资金净流入` (unit: **亿元**) with direction and board.

**Proposed mapping:**
- `northbound_flow` = sum of `成交净买额` where `资金方向 == "北向"`
- `northbound_flow_sh` = `板块 == "沪股通"` and `北向`
- `northbound_flow_sz` = `板块 == "深股通"` and `北向`

**Unit normalization:**
- raw is “亿元”; if we store in CNY, multiply by `1e8`.

---

### 2) `shibor` (daily)
**Documented API:** `ak.macro_china_shibor_all()`
- Doc: `docs/akshare/macro_china_shibor_all.md`

**Confirmed mapping (per repo docs):**
- `shibor_1w` ← `1W-定价`
- `shibor_3m` ← `3M-定价`

---

### 3) `margin` (daily margin financing)
**Documented APIs:**
- `ak.macro_china_market_margin_sh()`
  - Doc: `docs/akshare/macro_china_market_margin_sh.md`
- `ak.macro_china_market_margin_sz()`
  - Doc: `docs/akshare/macro_china_market_margin_sz.md`

**Confirmed mapping (per repo docs):**
- `margin_balance` ← SH/SZ `融资融券余额` (unit: 元), aggregated as `SH + SZ`.

---

### 4) `index_dailybasic` (CSI300/CSI500 valuation, turnover, etc.)
**Candidates (listed in `docs/akshare/akentry.md`, not documented here):**
- `stock_zh_index_value_csindex` (index valuation)
- `stock_zh_index_daily_em` (index prices)

**Status:** pending confirmation of exact schema/fields for PE/PB/dividend yield and whether they satisfy:
- `csi300_pe`, `csi300_pb`, `csi300_dividend_yield`, `csi500_pe`
- `market_turnover`

**Project note (CSI 300):**
- Use index id `000300` for CSI 300.
- Candidate endpoint (listed in `docs/akshare/akentry.md`): `stock_zh_index_value_csindex`.

---

### 5) `bond_daily` (10y treasury yield)
**Status:** pending confirmation of the preferred AkShare source for China 10Y yield in this project.

---

### 6) `macro` monthly (M2, PPI, PMI)
**Candidates (listed in `docs/akshare/akentry.md`, not documented here):**
- `macro_china_m2_yearly`
- `macro_china_ppi_yearly`
- `macro_china_pmi_yearly`

---

## How this maps to our codebase today

### Existing acquisition code (`src/data/acquisition.py`)
Already implemented:
- universe snapshot: `stock_zh_a_spot_em` → `fetch_stock_universe()`
- per-stock daily bars: `stock_zh_a_hist` → `fetch_market_data()`
- quarterly income statements: `stock_lrb_em` → `fetch_quarterly_income_statement()`
- quarterly balance sheets: `stock_zcfz_em` → `fetch_quarterly_balance_sheet()`
- quarterly cashflow statements: `stock_xjll_em` → `fetch_quarterly_cashflow_statement()`
- all fundamentals combined: `fetch_quarterly_fundamentals()`
- northbound flow: `stock_hsgt_fund_flow_summary_em` → `fetch_northbound_flow()`
- margin balance: `macro_china_market_margin_sh/sz` → `fetch_margin_balance()`
- shibor rates: `macro_china_shibor_all` → used in `fetch_macro_indicators()`
- CSI index valuation: `stock_zh_index_value_csindex` → `fetch_index_valuation_csindex()`
- Shenwan industry classification: `sw_index_*` APIs → `fetch_industry_classification()`
- macro indicators combined: `fetch_macro_indicators()`

Functions available for complete docs/table.md coverage:
- All basic market data (OHLCV, volume, amount, turnover_rate, etc.)
- Fundamental financial data (income, balance sheet, cash flow statements)
- Market indicators (northbound flow, margin balance, shibor)
- Industry classifications (Shenwan L1/L2/L3)
- Index valuations (CSI 300 PE, PB, dividend yield)

### Existing processing code 

- `transform_akshare_hist()` normalizes `stock_zh_a_hist` output into a schema used by notebooks/tests.
- This is **separate** from the ClickHouse-like `stock_daily` schema in `docs/table.md`.

---

## Next steps (implementation plan)
1) Implement acquisition functions for the missing macro + statement tables.
2) Decide on the Shenwan industry mapping source (`sw_l1/sw_l2`).
3) Add a small set of schema-normalization helpers so that `FeatureStore.save_raw_data()` can persist consistent parquet tables under `data/raw/`.
