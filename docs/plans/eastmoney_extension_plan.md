# East Money Extension Plan

## Overview
Migrate two additional report types into `src/quant_trade/provider/client.py`:
- `stock_zcfz_em` - Balance Sheet (资产负债表)
- `stock_xjll_em` - Cash Flow Statement (现金流量表)

## Current Pattern Analysis

### Common Structure
All three reports share the same structure:
```
API Endpoint: https://datacenter-web.eastmoney.com/api/data/v1/get
Parameters:
- reportName: Report identifier
- sortColumns, sortTypes, pageSize, pageNumber
- columns: "ALL"
- filter: SECURITY_TYPE_CODE, TRADE_MARKET_CODE, REPORT_DATE
```

### Report Types
| Report | reportName | Chinese Name |
|--------|-----------|-------------|
| Income Statement | RPT_DMSK_FN_INCOME | 利润表 |
| Balance Sheet | RPT_DMSK_FN_BALANCE | 资产负债表 |
| Cash Flow | RPT_DMSK_FN_CASHFLOW | 现金流量表 |

## Implementation Plan

### 1. Create Builder Classes

#### BalanceSheetBuilder
```python
class BalanceSheetBuilder(Builder):
    """Builder for Balance Sheet data."""
    
    COLUMN_MAPPING = {
        "SECURITY_CODE": "stock_code",
        "SECURITY_NAME_ABBR": "stock_name",
        "NOTICE_DATE": "notice_date",
        "TOTAL_ASSETS": "total_assets",
        "MONETARY_CAPITAL": "monetary_capital",
        "ACCOUNTS_RECEIVABLE": "accounts_receivable",
        "INVENTORY": "inventory",
        "TOTAL_LIABILITIES": "total_liabilities",
        "ACCOUNTS_PAYABLE": "accounts_payable",
        "ADVANCE_RECEIPTS": "advance_receipts",
        "TOTAL_EQUITY": "total_equity",
        "TOTAL_ASSETS_YOY": "total_assets_yoy",
        "TOTAL_LIABILITIES_YOY": "total_liabilities_yoy",
        "DEBT_ASSET_RATIO": "debt_asset_ratio",
    }
    
    NUMERIC_COLS = [
        "total_assets", "monetary_capital", "accounts_receivable",
        "inventory", "total_liabilities", "accounts_payable",
        "advance_receipts", "total_equity", "total_assets_yoy",
        "total_liabilities_yoy", "debt_asset_ratio",
    ]
    
    OUTPUT_COLS = [
        "seq", "stock_code", "stock_name", "total_assets",
        "total_assets_yoy", "monetary_capital", "accounts_receivable",
        "inventory", "total_liabilities", "total_liabilities_yoy",
        "accounts_payable", "advance_receipts", "debt_asset_ratio",
        "total_equity", "notice_date",
    ]
```

#### CashFlowBuilder
```python
class CashFlowBuilder(Builder):
    """Builder for Cash Flow Statement data."""
    
    COLUMN_MAPPING = {
        "SECURITY_CODE": "stock_code",
        "SECURITY_NAME_ABBR": "stock_name",
        "NOTICE_DATE": "notice_date",
        "NET_CASH_FLOW": "net_cash_flow",
        "NET_CASH_FLOW_YOY": "net_cash_flow_yoy",
        "OPERATE_CASH_FLOW": "operate_cash_flow",
        "OPERATE_CASH_FLOW_RATIO": "operate_cash_flow_ratio",
        "INVEST_CASH_FLOW": "invest_cash_flow",
        "INVEST_CASH_FLOW_RATIO": "invest_cash_flow_ratio",
        "FINANCE_CASH_FLOW": "finance_cash_flow",
        "FINANCE_CASH_FLOW_RATIO": "finance_cash_flow_ratio",
    }
    
    NUMERIC_COLS = [
        "net_cash_flow", "net_cash_flow_yoy", "operate_cash_flow",
        "operate_cash_flow_ratio", "invest_cash_flow", "invest_cash_flow_ratio",
        "finance_cash_flow", "finance_cash_flow_ratio",
    ]
    
    OUTPUT_COLS = [
        "seq", "stock_code", "stock_name", "net_cash_flow",
        "net_cash_flow_yoy", "operate_cash_flow", "operate_cash_flow_ratio",
        "invest_cash_flow", "invest_cash_flow_ratio",
        "finance_cash_flow", "finance_cash_flow_ratio",
        "notice_date",
    ]
```

### 2. Extend EastMoney Class

Add methods:
```python
class EastMoney:
    # ... existing methods ...
    
    def balance_sheet(self, year: int, quarter: int) -> pl.DataFrame:
        """Fetch quarterly balance sheet data."""
        ...
    
    def cashflow(self, year: int, quarter: int) -> pl.DataFrame:
        """Fetch quarterly cash flow data."""
        ...
```

### 3. Add Backward Compatible Functions

```python
def stock_zcfz_em(date: str = "20240331", **kwargs) -> pl.DataFrame:
    """Balance Sheet - backward compatible."""
    ...

def stock_xjll_em(date: str = "20240331", **kwargs) -> pl.DataFrame:
    """Cash Flow Statement - backward compatible."""
    ...
```

### 4. Update Tests

Add to `tests/test_client.py`:
- TestBalanceSheetBuilder
- TestCashFlowBuilder
- TestEastMoneyBalanceSheet
- TestEastMoneyCashflow
- TestBackwardCompatibilityZCFZ
- TestBackwardCompatibilityXJLL

## File Changes Summary

### src/quant_trade/provider/client.py
1. Add `BalanceSheetBuilder` class
2. Add `CashFlowBuilder` class
3. Add `EastMoney.balance_sheet()` method
4. Add `EastMoney.cashflow()` method
5. Add `stock_zcfz_em()` function
6. Add `stock_xjll_em()` function

### tests/test_client.py
1. Add tests for BalanceSheetBuilder
2. Add tests for CashFlowBuilder
3. Add tests for EastMoney integration methods
4. Add backward compatibility tests

## Benefits of This Approach

1. **Reuse Fetcher and Parser** - Both new reports use the same network layer
2. **Consistent Interface** - All three reports have the same quarterly(year, quarter) interface
3. **Maintainable** - Changes to column mappings are isolated to each Builder
4. **Testable** - Each component can be tested independently
5. **Backward Compatible** - Existing code continues to work
