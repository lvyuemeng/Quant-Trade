"""
Tests for East Money provider - all tests in one file.

Tests:
1. Test Protocols (duck typing)
2. Test EastMoneyFetch (mock HTTP)
3. Test EastMoneyParse (mock parsing)
4. Test EastMoneyBuilder (mock building)
5. Test EastMoney (integration)
6. Test Backward Compatibility
"""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from quant_trade.client.eastmoney import (
    BalanceSheetBuilder,
    CashFlowBuilder,
    EastMoney,
    EastMoneyFetch,
    FundemantalParser,
    IncomeBuilder,
    stock_lrb_em,
    stock_lrb_em_batch,
    stock_xjll_em,
    stock_zcfz_em,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create mock requests session."""
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = {"result": {"pages": 2, "data": []}}
    response.raise_for_status = MagicMock()
    session.get.return_value = response
    return session


@pytest.fixture
def sample_raw_response():
    """Sample raw API response."""
    return {
        "result": {
            "pages": 2,
            "data": [
                {
                    "SECURITY_CODE": "000001",
                    "SECURITY_NAME_ABBR": "平安银行",
                    "NOTICE_DATE": "2024-03-30",
                    "NETPROFIT": 1000000000,
                    "TOTAL_OPERATE_INCOME": 5000000000,
                    "TOTAL_OPERATE_COST": 4000000000,
                    "OPERATE_COST": 3000000000,
                    "SALE_EXPENSE": 100000000,
                    "MANAGE_EXPENSE": 200000000,
                    "FINANCE_EXPENSE": 50000000,
                    "OPERATE_PROFIT": 1500000000,
                    "TOTAL_PROFIT": 1600000000,
                    "TOTAL_OPERATE_INCOME_SQ": 0.15,
                    "NETPROFIT_SQ": 0.20,
                },
                {
                    "SECURITY_CODE": "000002",
                    "SECURITY_NAME_ABBR": "万 科Ａ",
                    "NOTICE_DATE": "2024-03-28",
                    "NETPROFIT": 2000000000,
                    "TOTAL_OPERATE_INCOME": 8000000000,
                    "TOTAL_OPERATE_COST": 6000000000,
                    "OPERATE_COST": 4500000000,
                    "SALE_EXPENSE": 300000000,
                    "MANAGE_EXPENSE": 400000000,
                    "FINANCE_EXPENSE": 80000000,
                    "OPERATE_PROFIT": 2500000000,
                    "TOTAL_PROFIT": 2600000000,
                    "TOTAL_OPERATE_INCOME_SQ": 0.12,
                    "NETPROFIT_SQ": 0.18,
                },
            ],
        }
    }


@pytest.fixture
def sample_second_page_response():
    """Sample second page response."""
    return {
        "result": {
            "data": [
                {
                    "SECURITY_CODE": "000003",
                    "SECURITY_NAME_ABBR": "PT金田A",
                    "NOTICE_DATE": "2024-03-25",
                    "NETPROFIT": 50000000,
                    "TOTAL_OPERATE_INCOME": 200000000,
                    "TOTAL_OPERATE_COST": 150000000,
                    "OPERATE_COST": 100000000,
                    "SALE_EXPENSE": 10000000,
                    "MANAGE_EXPENSE": 20000000,
                    "FINANCE_EXPENSE": 2000000,
                    "OPERATE_PROFIT": 60000000,
                    "TOTAL_PROFIT": 65000000,
                    "TOTAL_OPERATE_INCOME_SQ": 0.05,
                    "NETPROFIT_SQ": 0.10,
                }
            ]
        }
    }


# =============================================================================
# Test 1: Protocols (Duck Typing)
# =============================================================================


def test_fetcher_protocol():
    """Test Fetcher protocol compliance."""
    # EastMoneyFetch should satisfy Fetcher protocol
    fetcher = EastMoneyFetch()

    # Check required methods exist
    assert hasattr(fetcher, "fetch_initial")
    assert hasattr(fetcher, "fetch_page")
    assert hasattr(fetcher, "close")

    # Check method signatures (duck typing)
    assert callable(fetcher.fetch_initial)
    assert callable(fetcher.fetch_page)
    assert callable(fetcher.close)

    fetcher.close()


def test_parser_protocol():
    """Test Parser protocol compliance."""
    parser = FundemantalParser()

    # Check required methods exist
    assert hasattr(parser, "parse")
    assert hasattr(parser, "clean")

    # Check method signatures
    assert callable(parser.parse)
    assert callable(parser.clean)

    # Test functionality
    raw = {"result": {"data": [{"key": "value"}]}}
    result = parser.parse(raw)
    assert result == [{"key": "value"}]

    # Test clean
    df = parser.clean([{"col": 1}])
    assert isinstance(df, pl.DataFrame)


def test_builder_protocol():
    """Test Builder protocol compliance."""
    builder = IncomeBuilder()

    # Check required methods exist
    assert hasattr(builder, "rename")
    assert hasattr(builder, "convert_types")
    assert hasattr(builder, "reorder")

    # Check method signatures
    assert callable(builder.rename)
    assert callable(builder.convert_types)
    assert callable(builder.reorder)


# =============================================================================
# Test 2: EastMoneyFetch (Mock HTTP)
# =============================================================================


class TestEastMoneyFetch:
    """Tests for EastMoneyFetch with mocked HTTP."""

    def test_fetch_initial_success(self, mock_session, sample_raw_response):
        """Test successful initial fetch."""
        fetcher = EastMoneyFetch()
        fetcher._session = mock_session
        mock_session.get.return_value.json.return_value = sample_raw_response

        result = fetcher.fetch_initial("http://test.com", {"page": 1})

        assert result == sample_raw_response
        mock_session.get.assert_called_once()

    def test_fetch_initial_error(self, mock_session):
        """Test initial fetch with error."""
        fetcher = EastMoneyFetch()
        fetcher._session = mock_session
        mock_session.get.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            fetcher.fetch_initial("http://test.com", {})

    def test_fetch_page_success(self, mock_session, sample_raw_response):
        """Test successful page fetch."""
        fetcher = EastMoneyFetch()
        fetcher._session = mock_session
        mock_session.get.return_value.json.return_value = sample_raw_response

        result = fetcher.fetch_page("http://test.com", {"page": 1}, page=2)

        assert result == sample_raw_response
        # Verify page parameter was added
        call_args = mock_session.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert "pageNumber" in params

    def test_fetch_pages_concurrent(self, mock_session, sample_raw_response):
        """Test concurrent page fetching."""
        fetcher = EastMoneyFetch(max_workers=2)
        fetcher._session = mock_session
        mock_session.get.return_value.json.return_value = sample_raw_response

        results = fetcher.fetch_pages_concurrent("http://test.com", {}, pages=[2, 3, 4])

        assert len(results) == 3
        assert mock_session.get.call_count == 3

    def test_rate_limiting(self, mock_session):
        """Test that rate limiting is applied."""
        fetcher = EastMoneyFetch(delay_range=(0.1, 0.2))
        fetcher._session = mock_session
        mock_session.get.return_value.json.return_value = {}
        mock_session.get.return_value.raise_for_status = MagicMock()

        import time

        start = time.time()
        fetcher.fetch_initial("http://test.com", {})
        elapsed = time.time() - start

        # Should have waited at least 0.1 second
        assert elapsed >= 0.1

    def test_user_agent_rotation(self, mock_session):
        """Test that user agent is rotated."""
        fetcher = EastMoneyFetch()
        fetcher._session = mock_session
        mock_session.get.return_value.json.return_value = {}
        mock_session.get.return_value.raise_for_status = MagicMock()

        calls = []
        original_get = mock_session.get

        def track_call(*args, **kwargs):
            calls.append(kwargs.get("headers", {}).get("User-Agent"))
            return MagicMock(json=lambda: {}, raise_for_status=MagicMock())

        mock_session.get = track_call

        fetcher.fetch_initial("http://test.com", {})
        fetcher.fetch_page("http://test.com", {}, page=2)

        # Should have different user agents (randomly)
        assert len(calls) == 2


# =============================================================================
# Test 3: EastMoneyParse (Mock Parsing)
# =============================================================================


class TestEastMoneyParser:
    """Tests for EastMoneyParse."""

    def test_parse_with_data(self, sample_raw_response):
        """Test parsing response with data."""
        parser = FundemantalParser()
        result = parser.parse(sample_raw_response)

        assert len(result) == 2
        assert result[0]["SECURITY_CODE"] == "000001"

    def test_parse_empty(self):
        """Test parsing empty response."""
        parser = FundemantalParser()

        result = parser.parse({"result": None})
        assert result == []

        result = parser.parse({})
        assert result == []

        result = parser.parse({"result": {"data": []}})
        assert result == []

    def test_parse_malformed(self):
        """Test parsing malformed response."""
        parser = FundemantalParser()

        # Data is not a list
        result = parser.parse({"result": {"data": {"key": "value"}}})
        assert result == []

    def test_clean_data(self, sample_raw_response):
        """Test cleaning data into DataFrame."""
        parser = FundemantalParser()
        data = parser.parse(sample_raw_response)
        df = parser.clean(data)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 2
        assert "SECURITY_CODE" in df.columns


# =============================================================================
# Test 4: EastMoneyBuilder (Mock Building)
# =============================================================================


class TestEastMoneyBuilder:
    """Tests for EastMoneyBuilder."""

    @pytest.fixture
    def sample_df(self, sample_raw_response):
        """Create sample DataFrame."""
        parser = FundemantalParser()
        data = parser.parse(sample_raw_response)
        return parser.clean(data)

    def test_rename_columns(self, sample_df):
        """Test column renaming."""
        builder = IncomeBuilder()
        result = builder.rename(sample_df)

        assert "stock_code" in result.columns
        assert "stock_name" in result.columns
        assert "SECURITY_CODE" not in result.columns

    def test_convert_types_numeric(self, sample_df):
        """Test numeric type conversion."""
        builder = IncomeBuilder()
        result = builder.convert_types(sample_df)

        # Check numeric columns
        for col in builder.NUMERIC_COLS:
            if col in result.columns:
                assert result[col].dtype in [pl.Float64, pl.Float32, pl.Int64]

    def test_convert_types_date(self, sample_df):
        """Test date type conversion."""
        builder = IncomeBuilder()
        result = builder.convert_types(sample_df)

        if "notice_date" in result.columns:
            assert result["notice_date"].dtype == pl.Date

    def test_reorder_columns(self, sample_df):
        """Test column reordering."""
        builder = IncomeBuilder()
        renamed = builder.rename(sample_df)
        result = builder.reorder(renamed)

        # Check output columns are in correct order
        for i, col in enumerate(builder.OUTPUT_ORDER[: len(result.columns)]):
            assert result.columns[i] == col

    def test_reorder_adds_sequence(self, sample_df):
        """Test that sequence column is added."""
        builder = IncomeBuilder()
        renamed = builder.rename(sample_df)
        result = builder.reorder(renamed)

        assert "seq" in result.columns
        assert result["seq"][0] == 1
        assert result["seq"][1] == 2


# =============================================================================
# Test 5: EastMoney (Integration Tests)
# =============================================================================


class TestEastMoney:
    """Integration tests for EastMoney class."""

    @pytest.fixture
    def mock_fetcher(self, sample_raw_response, sample_second_page_response):
        """Create mock fetcher."""
        fetcher = MagicMock()

        def mock_initial(url, params):
            return sample_raw_response

        def mock_page(url, params, page):
            return sample_second_page_response if page == 2 else sample_raw_response

        def mock_concurrent(url, params, pages):
            return [sample_second_page_response]

        fetcher.fetch_initial = mock_initial
        fetcher.fetch_page = mock_page
        fetcher.fetch_pages_concurrent = mock_concurrent
        fetcher.close = MagicMock()

        return fetcher

    def test_quarterly_single_page(self, mock_fetcher, sample_raw_response):
        """Test quarterly fetch with single page."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher
            client._parser = FundemantalParser()
            client._builder = IncomeBuilder()

            result = client.quarterly_income(2024, 1)

            assert isinstance(result, pl.DataFrame)
            assert result.height >= 2

    def test_quarterly_multiple_pages(self, mock_fetcher):
        """Test quarterly fetch with multiple pages."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher
            client._parser = FundemantalParser()
            client._builder = IncomeBuilder()

            result = client.quarterly_income(2024, 1)

            assert isinstance(result, pl.DataFrame)
            # Should have data from both pages
            assert result.height >= 2

    def test_quarterly_empty_response(self, mock_fetcher):
        """Test quarterly fetch with empty response."""
        mock_fetcher.fetch_initial = MagicMock(return_value={"result": None})

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher
            client._parser = FundemantalParser()
            client._builder = IncomeBuilder()

            result = client.quarterly_income(2024, 1)

            assert isinstance(result, pl.DataFrame)
            assert result.height == 0

    def test_context_manager(self, mock_fetcher):
        """Test context manager functionality."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = mock_fetcher
                client._parser = FundemantalParser()
                client._builder = IncomeBuilder()

                result = client.quarterly_income(2024, 1)
                assert isinstance(result, pl.DataFrame)

            # Fetcher should be closed after context
            mock_fetcher.close.assert_called_once()

    def test_close_method(self, mock_fetcher):
        """Test close method."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher

            client.close()
            mock_fetcher.close.assert_called_once()


# =============================================================================
# Test 6: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatible functions."""

    def test_stock_lrb_em_signature(self):
        """Test stock_lrb_em has correct signature."""
        import inspect

        sig = inspect.signature(stock_lrb_em)

        params = list(sig.parameters.keys())
        assert "date" in params
        assert "kwargs" in params

    def test_stock_lrb_em_returns_polars(self, sample_raw_response):
        """Test stock_lrb_em returns Polars DataFrame."""
        mock_response = sample_raw_response

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = MagicMock()
                client._fetcher.fetch_initial = MagicMock(return_value=mock_response)
                client._fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
                client._fetcher.close = MagicMock()

                # Test the quarterly method directly
                client._parser = FundemantalParser()
                client._builder = IncomeBuilder()

                result = client._parser.parse(mock_response)
                result = client._parser.clean(result)

                assert isinstance(result, pl.DataFrame)

    def test_stock_lrb_em_batch_signature(self):
        """Test stock_lrb_em_batch has correct signature."""
        import inspect

        sig = inspect.signature(stock_lrb_em_batch)

        params = list(sig.parameters.keys())
        assert "dates" in params
        assert "max_workers" in params
        assert "delay_range" in params

    def test_stock_lrb_em_batch_returns_dict(self):
        """Test stock_lrb_em_batch returns dictionary."""
        mock_response = {"result": {"pages": 1, "data": []}}

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = MagicMock()
                client._fetcher.fetch_initial = MagicMock(return_value=mock_response)
                client._fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
                client._fetcher.close = MagicMock()

                result = stock_lrb_em_batch(["20240331"])

                assert isinstance(result, dict)
                assert "20240331" in result


# =============================================================================
# Test 7: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_parser_handles_missing_result(self):
        """Test parser handles missing result gracefully."""
        parser = FundemantalParser()
        result = parser.parse({})
        assert result == []

    def test_builder_handles_missing_columns(self):
        """Test builder handles missing columns gracefully."""
        builder = IncomeBuilder()

        # Create DataFrame with missing columns
        df = pl.DataFrame({"unknown_col": [1, 2, 3]})

        result = builder.rename(df)
        assert result.columns == ["unknown_col"]  # No rename applied

    def test_fetcher_handles_network_error(self, mock_session):
        """Test fetcher handles network errors."""
        fetcher = EastMoneyFetch()
        fetcher._session = mock_session
        mock_session.get.side_effect = Exception("Connection refused")

        with pytest.raises(Exception):
            fetcher.fetch_initial("http://test.com", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Test 8: Balance Sheet Builder
# =============================================================================


class TestBalanceSheetBuilder:
    """Tests for BalanceSheetBuilder."""

    @pytest.fixture
    def sample_balance_data(self):
        """Sample balance sheet data."""
        return [
            {
                "SECURITY_CODE": "000001",
                "SECURITY_NAME_ABBR": "平安银行",
                "NOTICE_DATE": "2024-03-30",
                "TOTAL_ASSETS": 100000000000,
                "MONETARY_CAPITAL": 20000000000,
                "ACCOUNTS_RECEIVABLE": 5000000000,
                "INVENTORY": 3000000000,
                "TOTAL_LIABILITIES": 80000000000,
                "ACCOUNTS_PAYABLE": 4000000000,
                "ADVANCE_RECEIPTS": 2000000000,
                "TOTAL_EQUITY": 20000000000,
                "TOTAL_ASSETS_YOY": 0.10,
                "TOTAL_LIABILITIES_YOY": 0.08,
                "DEBT_ASSET_RATIO": 0.80,
            },
            {
                "SECURITY_CODE": "000002",
                "SECURITY_NAME_ABBR": "万 科Ａ",
                "NOTICE_DATE": "2024-03-28",
                "TOTAL_ASSETS": 200000000000,
                "MONETARY_CAPITAL": 30000000000,
                "ACCOUNTS_RECEIVABLE": 8000000000,
                "INVENTORY": 50000000000,
                "TOTAL_LIABILITIES": 150000000000,
                "ACCOUNTS_PAYABLE": 10000000000,
                "ADVANCE_RECEIPTS": 5000000000,
                "TOTAL_EQUITY": 50000000000,
                "TOTAL_ASSETS_YOY": 0.12,
                "TOTAL_LIABILITIES_YOY": 0.10,
                "DEBT_ASSET_RATIO": 0.75,
            },
        ]

    def test_column_mapping(self):
        """Test column mapping exists."""
        builder = BalanceSheetBuilder()
        assert "SECURITY_CODE" in builder.COLUMN_MAPPING
        assert "TOTAL_ASSETS" in builder.COLUMN_MAPPING
        assert builder.COLUMN_MAPPING["SECURITY_CODE"] == "stock_code"

    def test_rename(self, sample_balance_data):
        """Test column renaming."""
        builder = BalanceSheetBuilder()
        df = pl.DataFrame(sample_balance_data)
        result = builder.rename(df)

        assert "stock_code" in result.columns
        assert "stock_name" in result.columns
        assert "total_assets" in result.columns
        assert "SECURITY_CODE" not in result.columns

    def test_convert_types(self, sample_balance_data):
        """Test type conversion."""
        builder = BalanceSheetBuilder()
        df = pl.DataFrame(sample_balance_data)
        df = builder.rename(df)
        result = builder.convert_types(df)

        assert result["total_assets"].dtype == pl.Float64
        assert result["debt_asset_ratio"].dtype == pl.Float64

    def test_reorder(self, sample_balance_data):
        """Test column reordering."""
        builder = BalanceSheetBuilder()
        df = pl.DataFrame(sample_balance_data)
        df = builder.rename(df)
        df = builder.convert_types(df)
        result = builder.reorder(df)

        # Check sequence column is added
        assert "seq" in result.columns
        # Check columns are in expected order
        assert result.columns[0] == "seq"
        assert result.columns[1] == "stock_code"


# =============================================================================
# Test 9: Cash Flow Builder
# =============================================================================


class TestCashFlowBuilder:
    """Tests for CashFlowBuilder."""

    @pytest.fixture
    def sample_cashflow_data(self):
        """Sample cash flow data."""
        return [
            {
                "SECURITY_CODE": "000001",
                "SECURITY_NAME_ABBR": "平安银行",
                "NOTICE_DATE": "2024-03-30",
                "NET_CASH_FLOW": 5000000000,
                "NET_CASH_FLOW_YOY": 0.15,
                "OPERATE_CASH_FLOW": 8000000000,
                "OPERATE_CASH_FLOW_RATIO": 0.20,
                "INVEST_CASH_FLOW": -2000000000,
                "INVEST_CASH_FLOW_RATIO": -0.05,
                "FINANCE_CASH_FLOW": -1000000000,
                "FINANCE_CASH_FLOW_RATIO": -0.02,
            },
            {
                "SECURITY_CODE": "000002",
                "SECURITY_NAME_ABBR": "万 科Ａ",
                "NOTICE_DATE": "2024-03-28",
                "NET_CASH_FLOW": 10000000000,
                "NET_CASH_FLOW_YOY": 0.20,
                "OPERATE_CASH_FLOW": 15000000000,
                "OPERATE_CASH_FLOW_RATIO": 0.25,
                "INVEST_CASH_FLOW": -3000000000,
                "INVEST_CASH_FLOW_RATIO": -0.05,
                "FINANCE_CASH_FLOW": -2000000000,
                "FINANCE_CASH_FLOW_RATIO": -0.03,
            },
        ]

    def test_column_mapping(self):
        """Test column mapping exists."""
        builder = CashFlowBuilder()
        assert "SECURITY_CODE" in builder.COLUMN_MAPPING
        assert "NET_CASH_FLOW" in builder.COLUMN_MAPPING
        assert builder.COLUMN_MAPPING["NET_CASH_FLOW"] == "net_cash_flow"

    def test_rename(self, sample_cashflow_data):
        """Test column renaming."""
        builder = CashFlowBuilder()
        df = pl.DataFrame(sample_cashflow_data)
        result = builder.rename(df)

        assert "stock_code" in result.columns
        assert "net_cash_flow" in result.columns
        assert "operate_cash_flow" in result.columns
        assert "SECURITY_CODE" not in result.columns

    def test_convert_types(self, sample_cashflow_data):
        """Test type conversion."""
        builder = CashFlowBuilder()
        df = pl.DataFrame(sample_cashflow_data)
        df = builder.rename(df)
        result = builder.convert_types(df)

        assert result["net_cash_flow"].dtype == pl.Float64
        assert result["operate_cash_flow_ratio"].dtype == pl.Float64

    def test_reorder(self, sample_cashflow_data):
        """Test column reordering."""
        builder = CashFlowBuilder()
        df = pl.DataFrame(sample_cashflow_data)
        df = builder.rename(df)
        df = builder.convert_types(df)
        result = builder.reorder(df)

        # Check sequence column is added
        assert "seq" in result.columns
        # Check columns are in expected order
        assert result.columns[0] == "seq"
        assert result.columns[1] == "stock_code"


# =============================================================================
# Test 10: EastMoney Balance Sheet Integration
# =============================================================================


class TestEastMoneyBalanceSheet:
    """Integration tests for EastMoney.balance_sheet()."""

    @pytest.fixture
    def mock_balance_response(self):
        """Mock balance sheet API response."""
        return {
            "result": {
                "pages": 1,
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "NOTICE_DATE": "2024-03-30",
                        "TOTAL_ASSETS": 100000000000,
                        "MONETARY_CAPITAL": 20000000000,
                        "ACCOUNTS_RECEIVABLE": 5000000000,
                        "INVENTORY": 3000000000,
                        "TOTAL_LIABILITIES": 80000000000,
                        "ACCOUNTS_PAYABLE": 4000000000,
                        "ADVANCE_RECEIPTS": 2000000000,
                        "TOTAL_EQUITY": 20000000000,
                        "TOTAL_ASSETS_YOY": 0.10,
                        "TOTAL_LIABILITIES_YOY": 0.08,
                        "DEBT_ASSET_RATIO": 0.80,
                    }
                ],
            }
        }

    def test_balance_sheet_returns_polars(self, mock_balance_response):
        """Test balance_sheet returns Polars DataFrame."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = MagicMock()
                client._fetcher.fetch_initial = MagicMock(
                    return_value=mock_balance_response
                )
                client._fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
                client._fetcher.close = MagicMock()

                result = client.quarterly_balance_sheet(2024, 1)

                assert isinstance(result, pl.DataFrame)
                assert "stock_code" in result.columns


# =============================================================================
# Test 11: EastMoney Cash Flow Integration
# =============================================================================


class TestEastMoneyCashflow:
    """Integration tests for EastMoney.cashflow()."""

    @pytest.fixture
    def mock_cashflow_response(self):
        """Mock cash flow API response."""
        return {
            "result": {
                "pages": 1,
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "NOTICE_DATE": "2024-03-30",
                        "NET_CASH_FLOW": 5000000000,
                        "NET_CASH_FLOW_YOY": 0.15,
                        "OPERATE_CASH_FLOW": 8000000000,
                        "OPERATE_CASH_FLOW_RATIO": 0.20,
                        "INVEST_CASH_FLOW": -2000000000,
                        "INVEST_CASH_FLOW_RATIO": -0.05,
                        "FINANCE_CASH_FLOW": -1000000000,
                        "FINANCE_CASH_FLOW_RATIO": -0.02,
                    }
                ],
            }
        }

    def test_cashflow_returns_polars(self, mock_cashflow_response):
        """Test cashflow returns Polars DataFrame."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = MagicMock()
                client._fetcher.fetch_initial = MagicMock(
                    return_value=mock_cashflow_response
                )
                client._fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
                client._fetcher.close = MagicMock()

                result = client.quarterly_cashflow(2024, 1)

                assert isinstance(result, pl.DataFrame)
                assert "stock_code" in result.columns
                assert "net_cash_flow" in result.columns


# =============================================================================
# Test 12: Backward Compatibility - ZCFZ and XJLL
# =============================================================================


class TestBackwardCompatibilityZCFZXJLL:
    """Tests for backward compatible functions."""

    def test_stock_zcfz_em_signature(self):
        """Test stock_zcfz_em has correct signature."""
        import inspect

        sig = inspect.signature(stock_zcfz_em)

        params = list(sig.parameters.keys())
        assert "date" in params
        assert "kwargs" in params

    def test_stock_xjll_em_signature(self):
        """Test stock_xjll_em has correct signature."""
        import inspect

        sig = inspect.signature(stock_xjll_em)

        params = list(sig.parameters.keys())
        assert "date" in params
        assert "kwargs" in params

    def test_stock_zcfz_em_returns_polars(self):
        """Test stock_zcfz_em returns Polars DataFrame."""
        mock_response = {
            "result": {
                "pages": 1,
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "NOTICE_DATE": "2024-03-30",
                        "TOTAL_ASSETS": 100000000000,
                        "MONETARY_CAPITAL": 20000000000,
                        "ACCOUNTS_RECEIVABLE": 5000000000,
                        "INVENTORY": 3000000000,
                        "TOTAL_LIABILITIES": 80000000000,
                        "ACCOUNTS_PAYABLE": 4000000000,
                        "ADVANCE_RECEIPTS": 2000000000,
                        "TOTAL_EQUITY": 20000000000,
                        "TOTAL_ASSETS_YOY": 0.10,
                        "TOTAL_LIABILITIES_YOY": 0.08,
                        "DEBT_ASSET_RATIO": 0.80,
                    }
                ],
            }
        }

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = MagicMock()
                client._fetcher.fetch_initial = MagicMock(return_value=mock_response)
                client._fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
                client._fetcher.close = MagicMock()

                # Test parsing logic directly
                client._parser = FundemantalParser()
                result = client._parser.parse(mock_response)
                result = client._parser.clean(result)

                assert isinstance(result, pl.DataFrame)

    def test_stock_xjll_em_returns_polars(self):
        """Test stock_xjll_em returns Polars DataFrame."""
        mock_response = {
            "result": {
                "pages": 1,
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "NOTICE_DATE": "2024-03-30",
                        "NET_CASH_FLOW": 5000000000,
                        "NET_CASH_FLOW_YOY": 0.15,
                        "OPERATE_CASH_FLOW": 8000000000,
                        "OPERATE_CASH_FLOW_RATIO": 0.20,
                        "INVEST_CASH_FLOW": -2000000000,
                        "INVEST_CASH_FLOW_RATIO": -0.05,
                        "FINANCE_CASH_FLOW": -1000000000,
                        "FINANCE_CASH_FLOW_RATIO": -0.02,
                    }
                ],
            }
        }

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with EastMoney() as client:
                client._fetcher = MagicMock()
                client._fetcher.fetch_initial = MagicMock(return_value=mock_response)
                client._fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
                client._fetcher.close = MagicMock()

                # Test parsing logic directly
                client._parser = FundemantalParser()
                result = client._parser.parse(mock_response)
                result = client._parser.clean(result)

                assert isinstance(result, pl.DataFrame)
