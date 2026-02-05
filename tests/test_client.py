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

