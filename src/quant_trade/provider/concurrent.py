"""Concurrent request pattern for parallel data fetching.

This module provides a minimal, composable abstraction for parallel data fetching
with retry logic, supporting both ThreadPoolExecutor and ProcessPoolExecutor.

Example:
    from quant_trade.provider.concurrent import BatchConfig, Try, batch_fetch

    # Simple retry
    wrapped = Try(retry=3, sleep=0.5)(fetch_func)
    dfs = batch_fetch(BatchConfig.thread(), wrapped, symbols)

    # With session
    wrapped = Try(retry=3, sleep=0.5).with_session(BaoSession)(fetch_func)
    dfs = batch_fetch(BatchConfig.process(), wrapped, symbols)
"""

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, Self

import polars as pl
import tqdm

from quant_trade.config.logger import log

type FetchResult = pl.DataFrame


def optimal_workers(n_tasks: int) -> int:
    """Determine optimal number of workers for ThreadPoolExecutor."""
    import os

    cpu_count = os.cpu_count() or 4

    if n_tasks <= 0:
        raise ValueError(f"n_tasks {n_tasks} must be > 0")
    return min(cpu_count, n_tasks)


def try_call(
    fetch: Callable[..., FetchResult],
    retry: int = 3,
    sleep: float = 0.5,
    *args: Any,
    **kwargs: Any,
) -> FetchResult | pl.DataFrame:
    """Call function with retry and exponential backoff.

    Args:
        fetch: Function to call
        retry: Number of retry attempts
        sleep: Initial sleep seconds (doubles each attempt)
        *args: Positional arguments for fetch
        **kwargs: Keyword arguments for fetch

    Returns:
        Result of fetch call, or empty DataFrame on failure
    """
    for attempt in range(retry):
        try:
            return fetch(*args, **kwargs)
        except Exception as e:
            if attempt == 0:
                log.warning(f"Attempt 1 failed: {e}")
            if attempt == retry - 1:
                log.error(f"All {retry} attempts failed: {e}")
                return pl.DataFrame()
            backoff = sleep * (2**attempt)
            import time

            time.sleep(backoff)
    return pl.DataFrame()


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Batch-level configuration for concurrent fetching.

    Attributes:
        executor_factory: Factory function to create executor (ThreadPoolExecutor or ProcessPoolExecutor)
        max_workers: Maximum number of workers (None for auto-detection)
    """

    executor_factory: Callable[[int], Executor]
    max_workers: int | None = None

    @classmethod
    def process(cls, workers: int | None = None) -> Self:
        """Create ProcessPool-based configuration.

        Args:
            workers: Maximum number of processes (None for auto)

        Returns:
            BatchConfig with ProcessPoolExecutor
        """
        from concurrent.futures import ProcessPoolExecutor

        return cls(
            executor_factory=ProcessPoolExecutor,
            max_workers=workers,
        )

    @classmethod
    def thread(cls, workers: int | None = None) -> Self:
        """Create ThreadPool-based configuration.

        Args:
            workers: Maximum number of threads (None for auto)

        Returns:
            BatchConfig with ThreadPoolExecutor
        """
        from concurrent.futures import ThreadPoolExecutor

        return cls(
            executor_factory=ThreadPoolExecutor,
            max_workers=workers,
        )


@dataclass(frozen=True, slots=True)
class Try:
    """Factory for creating retry-wrapped callables.

    Attributes:
        retry: Number of retry attempts (default: 3)
        sleep: Initial sleep seconds between retries (default: 0.5)

    Example:
        retry_wrap = Try(retry=5, sleep=1.0)
        wrapped = retry_wrap(fetch_func)
        result = wrapped(arg1, arg2)  # fetch_func called with retry logic
    """

    retry: int = 3
    sleep: float = 0.5

    def __call__(
        self,
        fetch: Callable[..., FetchResult],
    ) -> Callable[..., FetchResult | pl.DataFrame]:
        """Wrap fetch function with retry logic.

        Args:
            fetch: Function to wrap

        Returns:
            Wrapper function with retry logic
        """
        return partial(self.pickable, fetch)

    def pickable(
        self, fetch: Callable[..., FetchResult], *args, **kwargs
    ) -> FetchResult:
        """Wrap fetch function with retry logic, ensuring picklability for ProcessPoolExecutor.

        Args:
            fetch: Function to wrap
        """
        return try_call(fetch, self.retry, self.sleep, *args, **kwargs)

    def with_session(
        self,
        session_factory: Callable[[], AbstractContextManager[None]],
    ) -> "TryWithSession":
        """Create TryWithSession combining retry and session.

        Args:
            session_factory: Factory function that creates session context manager

        Returns:
            TryWithSession instance
        """
        return TryWithSession(
            retry=self.retry,
            sleep=self.sleep,
            session_factory=session_factory,
        )


@dataclass(frozen=True, slots=True)
class TryWithSession:
    """Factory for retry + session wrapped callables.

    Pickleable for ProcessPoolExecutor.

    Attributes:
        retry: Number of retry attempts
        sleep: Initial sleep seconds between retries
        session_factory: Factory function for session context manager

    Example:
        wrapped = Try(retry=3, sleep=0.5).with_session(BaoSession)
        result = wrapped(fetch_func)(symbol)  # Session + retry
    """

    retry: int = 3
    sleep: float = 0.5
    session_factory: Callable[[], AbstractContextManager[None]] | None = None

    def __call__(
        self,
        fetch: Callable[..., FetchResult],
    ) -> Callable[..., FetchResult | pl.DataFrame]:
        """Wrap fetch function with session and retry logic.

        Args:
            fetch: Function to wrap

        Returns:
            Wrapper function with session + retry logic
        """

        def wrapper(*args: Any, **kwargs: Any) -> FetchResult | pl.DataFrame:
            if self.session_factory is None:
                return try_call(fetch, self.retry, self.sleep, *args, **kwargs)
            with self.session_factory():
                return try_call(fetch, self.retry, self.sleep, *args, **kwargs)

        return wrapper


def batch_fetch[T](
    config: BatchConfig,
    worker: Callable[[T], FetchResult],
    items: Sequence[T],
) -> list[FetchResult | pl.DataFrame]:
    """Execute parallel fetch for multiple items with order preservation.

    Args:
        config: BatchConfig with executor settings
        fetch_func: Pickleable function that takes one item and returns DataFrame
        items: Sequence of inputs - each item passed as single argument

    Returns:
        List of results in same order as items.
        Empty DataFrame for failed items.

    Example:
        def fetch_daily(symbol: str) -> pl.DataFrame:
            ...

        dfs = batch_fetch(
            config=BatchConfig.thread(),
            fetch_func=fetch_daily,
            items=["sh.600000", "sz.000001"],
        )
    """
    if not items:
        return []

    n = len(items)
    results: list[FetchResult | pl.DataFrame] = [pl.DataFrame()] * n

    workers = config.max_workers or optimal_workers(n)

    with config.executor_factory(max_workers=workers) as executor:  # type: ignore[arg-type]
        futures: dict[Any, int] = {}
        for idx, item in enumerate(items):
            future = executor.submit(worker, item)
            futures[future] = idx

        for future in tqdm.tqdm(
            as_completed(futures),
            total=n,
            desc="Fetching data",
            position=0,
        ):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                item = items[idx]
                log.error(f"Failed to fetch {item}: {e}")
                results[idx] = pl.DataFrame()

    return results
