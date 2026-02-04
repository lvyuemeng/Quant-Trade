# Concurrent Request Pattern

A minimal, composable abstraction for parallel data fetching with retry logic.

## Design Principles

- **Session independent**: Session handled separately, placed inside or outside executor
- **Pickle-safe**: Works with ProcessPoolExecutor (requires pickleable callables)
- **Separation of concerns**: Batch vs worker vs session
- **Factory pattern**: Configurable callables via factory classes

## Core Components

### 1. Retry Factory - `Try`

```python
from dataclasses import dataclass
from collections.abc import Callable

@dataclass(frozen=True, slots=True)
class Try:
    """
    Factory for creating retry-wrapped callables.
    
    Usage:
        retry5 = Try(retry=5, sleep=1.0)
        wrapped = retry5(fetch_func)
        wrapped(arg1, arg2)  # calls fetch_func with retry logic
    """
    retry: int = 3
    sleep: float = 0.5
    
    def __call__(self, fetch: Callable) -> Callable:
        """Wrap fetch with retry logic."""
        
        def wrapper(*args, **kwargs) -> pl.DataFrame:
            return try_call(fetch, self.retry, self.sleep, *args, **kwargs)
        
        return wrapper
    
    def with_session(self, session_cls) -> "TryWithSession":
        """Create TryWithSession combining retry and session."""
        return TryWithSession(
            retry=self.retry,
            sleep=self.sleep,
            session_cls=session_cls,
        )


@dataclass(frozen=True, slots=True)
class TryWithSession:
    """
    Factory for retry + session wrapped callables.
    Pickleable for ProcessPoolExecutor.
    """
    retry: int = 3
    sleep: float = 0.5
    session_cls = None
    
    def __call__(self, fetch: Callable) -> Callable:
        """Wrap fetch with session and retry."""
        
        def wrapper(*args, **kwargs) -> pl.DataFrame:
            with self.session_cls():
                return try_call(fetch, self.retry, self.sleep, *args, **kwargs)
        
        return wrapper
```

### 2. Batch Configuration

```python
from dataclasses import dataclass

type ExecutorFactory = Callable[[int], Executor]

@dataclass(frozen=True, slots=True)
class BatchConfig:
    """
    Batch-level configuration (executor only).
    
    Usage:
        config = BatchConfig.process()
        config = BatchConfig.thread(max_workers=8)
    """
    executor_factory: ExecutorFactory
    max_workers: int | None = None
    
    @classmethod
    def process(cls, workers: int | None = None) -> "BatchConfig":
        """Process-based configuration."""
        return cls(
            executor_factory=ProcessPoolExecutor,
            max_workers=workers,
        )
    
    @classmethod
    def thread(cls, workers: int | None = None) -> "BatchConfig":
        """Thread-based configuration."""
        return cls(
            executor_factory=ThreadPoolExecutor,
            max_workers=workers,
        )
```

### 3. Batch Fetcher

```python
def batch_fetch[T](
    config: BatchConfig,
    fetch_func: Callable[[T], pl.DataFrame],
    items: Sequence[T],
) -> list[pl.DataFrame]:
    """
    Generic parallel fetcher.
    
    Args:
        config: BatchConfig with executor settings
        fetch_func: Pickleable function that takes one T and returns DataFrame
        items: Sequence of inputs - each item passed as single argument
    
    Returns:
        list[pl.DataFrame] in same order as items
    """
    n = len(items)
    results: list[pl.DataFrame] = [pl.DataFrame()] * n
    
    workers = config.max_workers or optimal_workers(n)
    
    with config.executor_factory(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_func, item): idx
            for idx, item in enumerate(items)
        }
        
        for future in tqdm.tqdm(as_completed(futures), total=n):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                item = items[idx]
                log.error(f"Failed to fetch {item}: {e}")
                results[idx] = pl.DataFrame()
    
    return results
```

## Usage Patterns

### Pattern 1: Try Factory - Simple

```python
# Create retry wrapper
retry3 = Try(retry=3, sleep=0.5)
wrapped = retry3(market_ohlcv)

# Use in batch_fetch
dfs = batch_fetch(BatchConfig.process(), wrapped, symbols)
```

### Pattern 2: Try.with_session - Session + Retry

```python
# Create combined wrapper
with_session = Try(retry=3, sleep=0.5).with_session(BaoSession)
wrapped = with_session(market_ohlcv)

dfs = batch_fetch(BatchConfig.process(), wrapped, symbols)
```

### Pattern 3: Named Function with Try

```python
# Named function using Try factory
def market_ohlcv_worker(symbol: str, period: str = "daily") -> pl.DataFrame:
    with BaoSession():
        return Try(retry=3, sleep=0.5)(market_ohlcv)(
            symbol=symbol,
            period=period,
        )

dfs = batch_fetch(BatchConfig.process(), market_ohlcv_worker, symbols)
```

### Pattern 4: Partial + Try

```python
from functools import partial

def query(symbol: str, start: str, end: str) -> pl.DataFrame:
    ...

wrapped = partial(query, start="2024-01-01", end="2024-12-31")
wrapped = Try(retry=3, sleep=0.5)(wrapped)

dfs = batch_fetch(BatchConfig.thread(), wrapped, symbols)
```

### Pattern 5: Lambda (ThreadPool Only)

```python
# Lambdas are NOT pickleable - ThreadPool only
dfs = batch_fetch(
    config=BatchConfig.thread(),
    fetch_func=lambda s: stock_daily(s, "2024-01-01", "2024-12-31"),
    items=symbols,
)
```

## Factory Comparison

| Factory | Pickleable | Use Case |
|---------|------------|----------|
| `Try(retry=3)` | ✅ Yes | Retry only |
| `Try().with_session(BaoSession)` | ✅ Yes | Retry + Session |
| Named function | ✅ Yes | Complex logic |
| Lambda | ❌ No | Simple, ThreadPool only |

## Pickle Safety Guide

### DO (Pickleable)

```python
# Try factory
retry5 = Try(retry=5, sleep=1.0)
wrapped = retry5(fetch_func)

# Try with session
wrapped = Try().with_session(BaoSession)(fetch_func)

# Named function
def worker(symbol: str) -> pl.DataFrame:
    with BaoSession():
        return Try()(fetch_func)(symbol=symbol)
```

### DON'T (Not Pickleable)

```python
# Inline lambda
lambda s: Try()(fetch_func)(s)

# Local function
def outer():
    def inner(symbol):  # Not pickleable
        ...
    return inner
```

## When to Use ThreadPool vs ProcessPool

| Scenario | Executor | Pickleable Required? |
|----------|----------|---------------------|
| **I/O-bound, simple** | ThreadPoolExecutor | No (lambdas OK) |
| **Session-per-process** | ProcessPoolExecutor | Yes (Try, named functions) |
| **CPU-intensive** | ProcessPoolExecutor | Yes (Try, named functions) |

## Error Handling

1. **Per-item failure**: Empty DataFrame returned
2. **Batch continuation**: Other items continue
3. **Error logging**: Failed items logged with item repr
4. **Retry**: Handled by Try factory

## Configuration Summary

| Level | Component | Settings |
|-------|-----------|----------|
| **Batch** | `BatchConfig` | executor_factory, max_workers |
| **Retry** | `Try` | retry, sleep |
| **Session** | `BaoSession` | login/logout |

## Migration from Existing Code

### Before (BaoMicro)

```python
class BaoMicro:
    @staticmethod
    def _worker(symbol: str, ...) -> pl.DataFrame:
        return try_call(
            BaoMicro.market_ohlcv,
            retry=3,
            sleep=0.5,
            symbol=symbol,
            ...
        )
```

### After (Refactored)

```python
# Using Try factory with session
def market_ohlcv_worker(symbol: str, ...) -> pl.DataFrame:
    return Try(retry=3, sleep=0.5).with_session(BaoSession)(
        market_ohlcv
    )(symbol=symbol, ...)

# Or more readable
retry_wrap = Try().with_session(BaoSession)
wrapped = retry_wrap(market_ohlcv)

def market_ohlcv_worker(symbol: str, ...) -> pl.DataFrame:
    return wrapped(symbol=symbol, ...)
```

## Benefits

1. **Factory pattern**: `Try` and `TryWithSession` are configurable callables
2. **Pickle-safe**: All factories are pickleable for ProcessPoolExecutor
3. **Separation of concerns**: Batch vs retry vs session
4. **Flexible**: Multiple patterns for different complexity levels
5. **Composable**: `Try().with_session()` combines concerns
