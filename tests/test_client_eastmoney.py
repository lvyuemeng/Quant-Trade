"""
Tests for East Money specific functionality.

This file contains tests for:
- Helper functions
- ReportConfig dataclass
- ReportPipe class
- KlinePipe class
- KlineParser (kline data parsing)
- KlineBuilder (kline data building)
- EastMoney class (main facade)
- stock_hist method (historical stock data)
- stock_zh_a_hist backward-compatible function
"""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from quant_trade.client.eastmoney import (
    EastMoney,
    EastMoneyFetch,
    KlineBuilder,
    KlineParser,
    KlinePipe,
    ReportConfig,
    ReportPipe,
    _build_kline_params,
    _build_quarterly_date,
    _build_quarterly_params,
    stock_zh_a_hist,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_kline_response():
    """Sample kline API response."""
    return {
        "data": {
            "klines": [
                "2024-01-02,10.50,10.80,11.00,10.30,1000000,10800000,6.67,2.86,0.30,0.5",
                "2024-01-03,10.80,10.90,11.10,10.70,1200000,13080000,3.70,0.93,0.10,0.6",
                "2024-01-04,10.90,11.20,11.30,10.80,1500000,16800000,4.59,2.75,0.30,0.75",
            ]
        }
    }


@pytest.fixture
def sample_kline_empty_response():
    """Sample empty kline API response."""
    return {
        "data": {
            "klines": []
        }
    }


@pytest.fixture
def sample_kline_malformed_response():
    """Sample malformed kline API response."""
    return {
        "data": {
            "klines": [
                "2024-01-02,10.50,10.80",  # Too few fields
                "2024-01-03,10.80,10.90,11.10,10.70,1200000,13080000,3.70,0.93,0.10,0.6,extra",
            ]
        }
    }


@pytest.fixture
def sample_report_response():
    """Sample quarterly report API response."""
    return {
        "result": {
            "data": [
                {
                    "SECURITY_CODE": "000001",
                    "SECURITY_NAME_ABBR": "平安银行",
                    "NOTICE_DATE": "2024-03-31",
                    "NETPROFIT": "1000000000",
                }
            ],
            "pages": 1,
        }
    }


@pytest.fixture
def sample_report_empty_response():
    """Sample empty quarterly report API response."""
    return {
        "result": {
            "data": [],
            "pages": 1,
        }
    }


@pytest.fixture
def mock_fetcher_kline(sample_kline_response):
    """Create mock fetcher for kline data."""
    fetcher = MagicMock()

    def mock_initial(url, params):
        return sample_kline_response

    fetcher.fetch_initial = mock_initial
    fetcher.close = MagicMock()

    return fetcher


@pytest.fixture
def mock_fetcher_report(sample_report_response):
    """Create mock fetcher for report data."""
    fetcher = MagicMock()

    def mock_initial(url, params):
        return sample_report_response

    fetcher.fetch_initial = mock_initial
    fetcher.fetch_pages_concurrent = MagicMock(return_value=[])
    fetcher.close = MagicMock()

    return fetcher


# =============================================================================
# Test 1: Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_build_quarterly_date(self):
        """Test building quarterly date string."""
        assert _build_quarterly_date(2024, 1) == "2024-03-31"
        assert _build_quarterly_date(2024, 2) == "2024-06-30"
        assert _build_quarterly_date(2024, 3) == "2024-09-30"
        assert _build_quarterly_date(2024, 4) == "2024-12-31"

    def test_build_quarterly_params(self):
        """Test building quarterly report params."""
        params = _build_quarterly_params("RPT_DMSK_FN_INCOME", "2024-03-31")

        assert params["reportName"] == "RPT_DMSK_FN_INCOME"
        assert params["pageNumber"] == "1"
        assert params["pageSize"] == "500"
        assert "REPORT_DATE='2024-03-31'" in params["filter"]

    def test_build_kline_params(self):
        """Test building kline params."""
        secid, params = _build_kline_params(
            symbol="000001",
            period="daily",
            start_date="20240101",
            end_date="20240131",
            adjust="qfq",
        )

        assert secid == "0.000001"
        assert params["klt"] == "101"
        assert params["fqt"] == "1"
        assert params["beg"] == "20240101"
        assert params["end"] == "20240131"

    def test_build_kline_params_sh_market(self):
        """Test building kline params for Shanghai market."""
        secid, params = _build_kline_params(
            symbol="600000",
            period="daily",
            start_date="20240101",
            end_date="20240131",
            adjust="",
        )

        assert secid == "1.600000"
        assert params["fqt"] == "0"

    def test_build_kline_params_weekly(self):
        """Test building kline params for weekly period."""
        secid, params = _build_kline_params(
            symbol="000001",
            period="weekly",
            start_date="20240101",
            end_date="20240331",
            adjust="hfq",
        )

        assert params["klt"] == "102"
        assert params["fqt"] == "2"


# =============================================================================
# Test 2: ReportConfig
# =============================================================================


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_create_config(self):
        """Test creating a ReportConfig."""
        from quant_trade.client.eastmoney import FundemantalParser, IncomeBuilder

        config = ReportConfig(
            report_name="RPT_DMSK_FN_INCOME",
            parser_class=FundemantalParser,
            builder_class=IncomeBuilder,
        )

        assert config.report_name == "RPT_DMSK_FN_INCOME"
        assert config.parser_class == FundemantalParser
        assert config.builder_class == IncomeBuilder
        assert config.url == "https://datacenter-web.eastmoney.com/api/data/v1/get"

    def test_create_config_with_custom_url(self):
        """Test creating a ReportConfig with custom URL."""
        from quant_trade.client.eastmoney import FundemantalParser, IncomeBuilder

        config = ReportConfig(
            report_name="RPT_DMSK_FN_INCOME",
            parser_class=FundemantalParser,
            builder_class=IncomeBuilder,
            url="https://custom.url/api",
        )

        assert config.url == "https://custom.url/api"


# =============================================================================
# Test 3: ReportPipe
# =============================================================================


class TestReportPipe:
    """Tests for ReportPipe class."""

    def test_fetch_single_page(self, mock_fetcher_report, sample_report_response):
        """Test fetching report with single page."""
        from quant_trade.client.eastmoney import FundemantalParser, IncomeBuilder

        config = ReportConfig(
            report_name="RPT_DMSK_FN_INCOME",
            parser_class=FundemantalParser,
            builder_class=IncomeBuilder,
        )

        pipe = ReportPipe(mock_fetcher_report, config)
        result = pipe.fetch(2024, 1)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "stock_code" in result.columns

    def test_fetch_empty_response(self, mock_fetcher_report, sample_report_empty_response):
        """Test fetching report with empty response."""
        from quant_trade.client.eastmoney import FundemantalParser, IncomeBuilder

        mock_fetcher_report.fetch_initial = MagicMock(return_value=sample_report_empty_response)

        config = ReportConfig(
            report_name="RPT_DMSK_FN_INCOME",
            parser_class=FundemantalParser,
            builder_class=IncomeBuilder,
        )

        pipe = ReportPipe(mock_fetcher_report, config)
        result = pipe.fetch(2024, 1)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0

    def test_fetch_multiple_pages(self, mock_fetcher_report, sample_report_response):
        """Test fetching report with multiple pages."""
        from quant_trade.client.eastmoney import FundemantalParser, IncomeBuilder

        # Modify response to have multiple pages
        multi_page_response = sample_report_response.copy()
        multi_page_response["result"]["pages"] = 3

        mock_fetcher_report.fetch_initial = MagicMock(return_value=multi_page_response)

        config = ReportConfig(
            report_name="RPT_DMSK_FN_INCOME",
            parser_class=FundemantalParser,
            builder_class=IncomeBuilder,
        )

        pipe = ReportPipe(mock_fetcher_report, config)
        result = pipe.fetch(2024, 1)

        # Should call fetch_pages_concurrent for pages 2 and 3
        mock_fetcher_report.fetch_pages_concurrent.assert_called_once()
        assert isinstance(result, pl.DataFrame)


# =============================================================================
# Test 4: KlinePipe
# =============================================================================


class TestKlinePipe:
    """Tests for KlinePipe class."""

    def test_fetch_daily(self, mock_fetcher_kline, sample_kline_response):
        """Test fetching daily kline data."""
        pipe = KlinePipe(mock_fetcher_kline)
        result = pipe.fetch(
            symbol="000001",
            period="daily",
            start_date="20240101",
            end_date="20240131",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3
        assert "date" in result.columns
        assert "stock_code" in result.columns
        assert result["stock_code"][0] == "000001"

    def test_fetch_weekly(self, mock_fetcher_kline):
        """Test fetching weekly kline data."""
        pipe = KlinePipe(mock_fetcher_kline)
        result = pipe.fetch(
            symbol="000001",
            period="weekly",
            start_date="20240101",
            end_date="20240331",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3

    def test_fetch_monthly(self, mock_fetcher_kline):
        """Test fetching monthly kline data."""
        pipe = KlinePipe(mock_fetcher_kline)
        result = pipe.fetch(
            symbol="000001",
            period="monthly",
            start_date="20240101",
            end_date="20241231",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3

    def test_fetch_with_adjust_qfq(self, mock_fetcher_kline):
        """Test fetching with forward adjustment."""
        pipe = KlinePipe(mock_fetcher_kline)
        result = pipe.fetch(
            symbol="000001",
            period="daily",
            start_date="20240101",
            end_date="20240131",
            adjust="qfq",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3

    def test_fetch_with_adjust_hfq(self, mock_fetcher_kline):
        """Test fetching with backward adjustment."""
        pipe = KlinePipe(mock_fetcher_kline)
        result = pipe.fetch(
            symbol="000001",
            period="daily",
            start_date="20240101",
            end_date="20240131",
            adjust="hfq",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3

    def test_fetch_empty_response(self):
        """Test fetching with empty response."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_initial = MagicMock(return_value={"data": {"klines": []}})
        mock_fetcher.close = MagicMock()

        pipe = KlinePipe(mock_fetcher)
        result = pipe.fetch(
            symbol="000001",
            period="daily",
            start_date="20240101",
            end_date="20240131",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0

    def test_fetch_sh_market(self, mock_fetcher_kline):
        """Test fetching Shanghai market stock."""
        pipe = KlinePipe(mock_fetcher_kline)
        result = pipe.fetch(
            symbol="600000",
            period="daily",
            start_date="20240101",
            end_date="20240131",
        )

        assert isinstance(result, pl.DataFrame)
        assert result["stock_code"][0] == "600000"


# =============================================================================
# Test 5: KlineParser
# =============================================================================


class TestKlineParser:
    """Tests for KlineParser."""

    def test_parse_with_data(self, sample_kline_response):
        """Test parsing kline response with data."""
        parser = KlineParser()
        result = parser.parse(sample_kline_response)

        assert len(result) == 3
        assert result[0]["date"] == "2024-01-02"
        assert result[0]["open"] == "10.50"
        assert result[0]["close"] == "10.80"
        assert result[0]["high"] == "11.00"
        assert result[0]["low"] == "10.30"
        assert result[0]["volume"] == "1000000"
        assert result[0]["amount"] == "10800000"
        assert result[0]["amplitude"] == "6.67"
        assert result[0]["change_pct"] == "2.86"
        assert result[0]["change_amt"] == "0.30"
        assert result[0]["turnover"] == "0.5"

    def test_parse_empty(self, sample_kline_empty_response):
        """Test parsing empty kline response."""
        parser = KlineParser()
        result = parser.parse(sample_kline_empty_response)

        assert result == []

    def test_parse_no_data_key(self):
        """Test parsing response without data key."""
        parser = KlineParser()
        result = parser.parse({})

        assert result == []

    def test_parse_malformed(self, sample_kline_malformed_response):
        """Test parsing malformed kline response."""
        parser = KlineParser()
        result = parser.parse(sample_kline_malformed_response)

        # Both entries are malformed (too few or too many fields)
        # First entry has 3 fields, second has 12 fields (expected 11)
        assert len(result) == 0

    def test_clean_data(self, sample_kline_response):
        """Test cleaning kline data into DataFrame."""
        parser = KlineParser()
        data = parser.parse(sample_kline_response)
        df = parser.clean(data)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert "date" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns

    def test_data_path(self):
        """Test that DATA_PATH is correctly configured."""
        parser = KlineParser()
        assert parser.DATA_PATH == ("data", "klines")


# =============================================================================
# Test 6: KlineBuilder
# =============================================================================


class TestKlineBuilder:
    """Tests for KlineBuilder."""

    @pytest.fixture
    def sample_kline_df(self, sample_kline_response):
        """Create sample kline DataFrame."""
        parser = KlineParser()
        data = parser.parse(sample_kline_response)
        return parser.clean(data)

    def test_column_specs(self):
        """Test column specifications."""
        builder = KlineBuilder()

        assert len(builder.COLUMN_SPECS) == 11
        assert builder.COLUMN_SPECS[0].source == "date"
        assert builder.COLUMN_SPECS[0].target == "date"
        assert builder.COLUMN_SPECS[1].source == "open"
        assert builder.COLUMN_SPECS[1].target == "open"
        assert builder.COLUMN_SPECS[1].dtype == pl.Float64

    def test_output_order(self):
        """Test output column order."""
        builder = KlineBuilder()

        assert len(builder.OUTPUT_ORDER) == 12
        assert builder.OUTPUT_ORDER[0] == "date"
        assert builder.OUTPUT_ORDER[1] == "stock_code"
        assert "open" in builder.OUTPUT_ORDER
        assert "close" in builder.OUTPUT_ORDER
        assert "high" in builder.OUTPUT_ORDER
        assert "low" in builder.OUTPUT_ORDER
        assert "volume" in builder.OUTPUT_ORDER
        assert "amount" in builder.OUTPUT_ORDER
        assert "amplitude" in builder.OUTPUT_ORDER
        assert "change_pct" in builder.OUTPUT_ORDER
        assert "change_amt" in builder.OUTPUT_ORDER
        assert "turnover" in builder.OUTPUT_ORDER

    def test_date_col(self):
        """Test date column configuration."""
        builder = KlineBuilder()
        assert builder.DATE_COL == "date"

    def test_duplicate_cols(self):
        """Test duplicate column configuration."""
        builder = KlineBuilder()
        assert builder.DUPLICATE_COLS == ("stock_code", "date")

    def test_rename_columns(self, sample_kline_df):
        """Test column renaming."""
        builder = KlineBuilder()
        result = builder.rename(sample_kline_df)

        # All columns should be renamed to target names
        assert "date" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

    def test_convert_types_numeric(self, sample_kline_df):
        """Test numeric type conversion."""
        builder = KlineBuilder()
        result = builder.convert_types(sample_kline_df)

        # Check numeric columns
        for col in builder.NUMERIC_COLS:
            if col in result.columns:
                assert result[col].dtype in [pl.Float64, pl.Float32, pl.Int64]

    def test_convert_types_date(self, sample_kline_df):
        """Test date type conversion."""
        builder = KlineBuilder()
        result = builder.convert_types(sample_kline_df)

        if "date" in result.columns:
            assert result["date"].dtype == pl.Date

    def test_reorder_columns(self, sample_kline_df):
        """Test column reordering."""
        builder = KlineBuilder()
        renamed = builder.rename(sample_kline_df)
        # Add stock_code column as done in stock_hist method
        renamed = renamed.with_columns(pl.lit("000001").alias("stock_code"))
        result = builder.reorder(renamed)

        # Check output columns are in correct order
        for i, col in enumerate(builder.OUTPUT_ORDER[: len(result.columns)]):
            assert result.columns[i] == col


# =============================================================================
# Test 7: EastMoney.stock_hist
# =============================================================================


class TestEastMoneyStockHist:
    """Tests for EastMoney.stock_hist method."""

    def test_stock_hist_daily(self, mock_fetcher_kline, sample_kline_response):
        """Test fetching daily stock data."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="daily",
                start_date="20240101",
                end_date="20240131",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3
            assert "date" in result.columns
            assert "stock_code" in result.columns
            assert result["stock_code"][0] == "000001"

    def test_stock_hist_weekly(self, mock_fetcher_kline):
        """Test fetching weekly stock data."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="weekly",
                start_date="20240101",
                end_date="20240331",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3

    def test_stock_hist_monthly(self, mock_fetcher_kline):
        """Test fetching monthly stock data."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="monthly",
                start_date="20240101",
                end_date="20241231",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3

    def test_stock_hist_adjust_qfq(self, mock_fetcher_kline):
        """Test fetching with forward adjustment."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="daily",
                start_date="20240101",
                end_date="20240131",
                adjust="qfq",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3

    def test_stock_hist_adjust_hfq(self, mock_fetcher_kline):
        """Test fetching with backward adjustment."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="daily",
                start_date="20240101",
                end_date="20240131",
                adjust="hfq",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3

    def test_stock_hist_no_adjust(self, mock_fetcher_kline):
        """Test fetching without adjustment."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="daily",
                start_date="20240101",
                end_date="20240131",
                adjust="",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3

    def test_stock_hist_sh_market(self, mock_fetcher_kline):
        """Test fetching Shanghai market stock (starts with 6)."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="600000",
                period="daily",
                start_date="20240101",
                end_date="20240131",
            )

            assert isinstance(result, pl.DataFrame)
            assert result["stock_code"][0] == "600000"

    def test_stock_hist_sz_market(self, mock_fetcher_kline):
        """Test fetching Shenzhen market stock (starts with 0)."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist(
                symbol="000001",
                period="daily",
                start_date="20240101",
                end_date="20240131",
            )

            assert isinstance(result, pl.DataFrame)
            assert result["stock_code"][0] == "000001"

    def test_stock_hist_empty_response(self):
        """Test fetching with empty response."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_initial = MagicMock(return_value={"data": {"klines": []}})
        mock_fetcher.close = MagicMock()

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher

            result = client.stock_hist(
                symbol="000001",
                period="daily",
                start_date="20240101",
                end_date="20240131",
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height == 0

    def test_stock_hist_default_params(self, mock_fetcher_kline):
        """Test fetching with default parameters."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_kline

            result = client.stock_hist()

            assert isinstance(result, pl.DataFrame)
            assert result.height == 3
            assert result["stock_code"][0] == "000001"


# =============================================================================
# Test 8: EastMoney.quarterly methods
# =============================================================================


class TestEastMoneyQuarterly:
    """Tests for EastMoney quarterly report methods."""

    def test_quarterly_income(self, mock_fetcher_report, sample_report_response):
        """Test fetching quarterly income statement."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_report

            result = client.quarterly_income(2024, 1)

            assert isinstance(result, pl.DataFrame)
            assert result.height == 1
            assert "stock_code" in result.columns

    def test_quarterly_balance_sheet(self, mock_fetcher_report, sample_report_response):
        """Test fetching quarterly balance sheet."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_report

            result = client.quarterly_balance_sheet(2024, 1)

            assert isinstance(result, pl.DataFrame)
            assert result.height == 1
            assert "stock_code" in result.columns

    def test_quarterly_cashflow(self, mock_fetcher_report, sample_report_response):
        """Test fetching quarterly cash flow statement."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_report

            result = client.quarterly_cashflow(2024, 1)

            assert isinstance(result, pl.DataFrame)
            assert result.height == 1
            assert "stock_code" in result.columns

    def test_quarterly_empty_response(self, mock_fetcher_report, sample_report_empty_response):
        """Test fetching quarterly report with empty response."""
        mock_fetcher_report.fetch_initial = MagicMock(return_value=sample_report_empty_response)

        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            client = EastMoney()
            client._fetcher = mock_fetcher_report

            result = client.quarterly_income(2024, 1)

            assert isinstance(result, pl.DataFrame)
            assert result.height == 0


# =============================================================================
# Test 9: stock_zh_a_hist (Backward Compatibility)
# =============================================================================


class TestStockZhAHist:
    """Tests for stock_zh_a_hist backward-compatible function."""

    def test_signature(self):
        """Test function has correct signature."""
        import inspect

        sig = inspect.signature(stock_zh_a_hist)

        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "period" in params
        assert "start_date" in params
        assert "end_date" in params
        assert "adjust" in params
        assert "timeout" in params
        assert "kwargs" in params

    def test_default_params(self, mock_fetcher_kline):
        """Test function with default parameters."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with patch.object(EastMoney, "__init__", return_value=None):
                client = MagicMock()
                client.stock_hist = MagicMock(return_value=pl.DataFrame())
                client.__enter__ = MagicMock(return_value=client)
                client.__exit__ = MagicMock(return_value=None)

                with patch("quant_trade.client.eastmoney.EastMoney", return_value=client):
                    result = stock_zh_a_hist()

                    assert isinstance(result, pl.DataFrame)

    def test_with_custom_params(self, mock_fetcher_kline):
        """Test function with custom parameters."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with patch.object(EastMoney, "__init__", return_value=None):
                client = MagicMock()
                client.stock_hist = MagicMock(return_value=pl.DataFrame())
                client.__enter__ = MagicMock(return_value=client)
                client.__exit__ = MagicMock(return_value=None)

                with patch("quant_trade.client.eastmoney.EastMoney", return_value=client):
                    result = stock_zh_a_hist(
                        symbol="600000",
                        period="weekly",
                        start_date="20240101",
                        end_date="20240331",
                        adjust="qfq",
                    )

                    assert isinstance(result, pl.DataFrame)
                    client.stock_hist.assert_called_once_with(
                        symbol="600000",
                        period="weekly",
                        start_date="20240101",
                        end_date="20240331",
                        adjust="qfq",
                    )

    def test_with_kwargs(self, mock_fetcher_kline):
        """Test function with kwargs."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with patch.object(EastMoney, "__init__", return_value=None):
                client = MagicMock()
                client.stock_hist = MagicMock(return_value=pl.DataFrame())
                client.__enter__ = MagicMock(return_value=client)
                client.__exit__ = MagicMock(return_value=None)

                with patch("quant_trade.client.eastmoney.EastMoney", return_value=client):
                    result = stock_zh_a_hist(
                        symbol="000001",
                        max_workers=5,
                        delay_range=(1.0, 2.0),
                    )

                    assert isinstance(result, pl.DataFrame)

    def test_timeout_param_ignored(self, mock_fetcher_kline):
        """Test that timeout parameter is accepted but not used."""
        with patch.object(EastMoneyFetch, "__init__", return_value=None):
            with patch.object(EastMoney, "__init__", return_value=None):
                client = MagicMock()
                client.stock_hist = MagicMock(return_value=pl.DataFrame())
                client.__enter__ = MagicMock(return_value=client)
                client.__exit__ = MagicMock(return_value=None)

                with patch("quant_trade.client.eastmoney.EastMoney", return_value=client):
                    result = stock_zh_a_hist(
                        symbol="000001",
                        timeout=30.0,
                    )

                    assert isinstance(result, pl.DataFrame)
                    # timeout should not be passed to stock_hist
                    client.stock_hist.assert_called_once()
                    call_kwargs = client.stock_hist.call_args.kwargs
                    assert "timeout" not in call_kwargs
