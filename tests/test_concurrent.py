"""Tests for concurrent request pattern."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import polars as pl

from quant_trade.provider.concurrent import (
    BatchConfig,
    Try,
    batch_fetch,
    try_call,
)


def optimal_workers(n_tasks: int) -> int:
    """Determine optimal number of workers for ThreadPoolExecutor."""
    import os

    cpu_count = os.cpu_count() or 4

    if n_tasks <= 0:
        raise ValueError(f"n_tasks {n_tasks} must be > 0")
    return min(cpu_count, n_tasks)


class TestTryCall:
    """Tests for try_call function."""

    def test_success_first_attempt(self):
        """Successful call on first attempt."""
        mock_func = MagicMock(return_value=pl.DataFrame({"a": [1]}))
        result = try_call(mock_func, retry=3, sleep=0.1)

        assert mock_func.call_count == 1
        assert result.equals(pl.DataFrame({"a": [1]}))

    def test_retry_on_failure(self):
        """Retry on failure, succeed on second attempt."""
        mock_func = MagicMock(side_effect=[Exception("fail"), pl.DataFrame({"a": [1]})])
        result = try_call(mock_func, retry=3, sleep=0.1)

        assert mock_func.call_count == 2
        assert result.equals(pl.DataFrame({"a": [1]}))

    def test_all_retries_fail(self):
        """Return empty DataFrame when all retries fail."""
        mock_func = MagicMock(side_effect=Exception("always fail"))
        result = try_call(mock_func, retry=3, sleep=0.1)

        assert mock_func.call_count == 3
        assert result.equals(pl.DataFrame())

    def test_with_args_kwargs(self):
        """Pass args and kwargs to function."""

        def func(a, b, c=None):
            return pl.DataFrame({"a": [a], "b": [b], "c": [c]})

        result = try_call(func, retry=1, a=1, b=2, c=3)
        assert result.equals(pl.DataFrame({"a": [1], "b": [2], "c": [3]}))


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_process_config(self):
        """Create ProcessPool-based config."""
        config = BatchConfig.process(workers=4)
        assert config.max_workers == 4

    def test_thread_config(self):
        """Create ThreadPool-based config."""
        config = BatchConfig.thread(workers=8)
        assert config.max_workers == 8


class TestTry:
    """Tests for Try factory."""

    def test_simple_retry(self):
        """Create wrapped function with retry."""
        mock_func = MagicMock(return_value=pl.DataFrame({"a": [1]}))
        wrapped = Try(retry=3, sleep=0.1)(mock_func)
        result = wrapped()

        assert mock_func.call_count == 1
        assert result.equals(pl.DataFrame({"a": [1]}))

    def test_retry_on_failure(self):
        """Retry on failure."""
        mock_func = MagicMock(side_effect=[Exception("fail"), pl.DataFrame({"a": [1]})])
        wrapped = Try(retry=3, sleep=0.1)(mock_func)
        result = wrapped()

        assert mock_func.call_count == 2
        assert result.equals(pl.DataFrame({"a": [1]}))

    def test_with_args(self):
        """Pass arguments to wrapped function."""

        def func(symbol):
            return pl.DataFrame({"symbol": [symbol]})

        wrapped = Try(retry=1)(func)
        result = wrapped("sh.600000")

        assert result.equals(pl.DataFrame({"symbol": ["sh.600000"]}))


class TestTryWithSession:
    """Tests for TryWithSession factory."""

    @contextmanager
    def mock_session(self):
        """Mock session context manager."""
        yield

    def test_with_session(self):
        """Wrap with session and retry."""
        mock_func = MagicMock(return_value=pl.DataFrame({"a": [1]}))
        wrapped = Try(retry=3, sleep=0.1).with_session(self.mock_session)(mock_func)
        result = wrapped()

        # Should enter session context
        assert mock_func.call_count == 1
        assert result.equals(pl.DataFrame({"a": [1]}))


class TestBatchFetch:
    """Tests for batch_fetch function."""

    def test_empty_items(self):
        """Return empty list for empty items."""
        result = batch_fetch(BatchConfig.thread(), lambda x: pl.DataFrame(), [])
        assert result == []

    def test_single_item(self):
        """Process single item."""

        def fetch(symbol: str) -> pl.DataFrame:
            return pl.DataFrame({"symbol": [symbol]})

        result = batch_fetch(
            BatchConfig.thread(workers=1),
            fetch,
            ["sh.600000"],
        )

        assert len(result) == 1
        assert result[0].equals(pl.DataFrame({"symbol": ["sh.600000"]}))

    def test_multiple_items(self):
        """Process multiple items in order."""

        def fetch(symbol: str) -> pl.DataFrame:
            return pl.DataFrame({"symbol": [symbol]})

        symbols = ["sh.600000", "sz.000001", "sh.600519"]
        result = batch_fetch(
            BatchConfig.thread(workers=2),
            fetch,
            symbols,
        )

        assert len(result) == 3
        for i, symbol in enumerate(symbols):
            assert result[i].equals(pl.DataFrame({"symbol": [symbol]}))

    def test_failure_returns_empty_df(self):
        """Failed fetch returns empty DataFrame."""

        def fetch(symbol: str):
            if symbol == "fail":
                raise Exception("intentional failure")
            return pl.DataFrame({"symbol": [symbol]})

        result = batch_fetch(
            BatchConfig.thread(workers=1),
            fetch,
            ["sh.600000", "fail", "sz.000001"],
        )

        assert len(result) == 3
        assert result[0].equals(pl.DataFrame({"symbol": ["sh.600000"]}))
        assert result[1].equals(pl.DataFrame())  # Empty for failure
        assert result[2].equals(pl.DataFrame({"symbol": ["sz.000001"]}))

    def test_order_preserved(self):
        """Results in same order as input."""
        call_order = []

        def fetch(symbol: str) -> pl.DataFrame:
            call_order.append(symbol)
            return pl.DataFrame({"symbol": [symbol]})

        symbols = ["a", "b", "c", "d", "e"]
        result = batch_fetch(
            BatchConfig.thread(workers=5),
            fetch,
            symbols,
        )

        # Verify order is preserved despite concurrent execution
        assert len(result) == 5
        for i, symbol in enumerate(symbols):
            assert result[i].equals(pl.DataFrame({"symbol": [symbol]}))

    def test_with_retry_wrapper(self):
        """Use with Try wrapper."""
        call_count = [0]

        def fetch(symbol: str) -> pl.DataFrame:
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("first fail")
            return pl.DataFrame({"symbol": [symbol]})

        wrapped = Try(retry=3, sleep=0.1)(fetch)
        result = batch_fetch(
            BatchConfig.thread(workers=1),
            wrapped,
            ["sh.600000"],
        )

        assert len(result) == 1
        assert result[0].equals(pl.DataFrame({"symbol": ["sh.600000"]}))
        assert call_count[0] == 2  # One retry


class TestIntegration:
    """Integration tests for concurrent pattern."""

    def test_end_to_end_thread(self):
        """End-to-end test with ThreadPoolExecutor."""

        def fetch_daily(symbol: str) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "symbol": [symbol],
                    "close": [100.0 + hash(symbol) % 10],
                }
            )

        symbols = ["sh.600000", "sz.000001", "sh.600519"]
        result = batch_fetch(
            BatchConfig.thread(workers=2),
            fetch_daily,
            symbols,
        )

        assert len(result) == 3
        for df in result:
            assert "symbol" in df.columns
            assert "close" in df.columns
            assert len(df) == 1
