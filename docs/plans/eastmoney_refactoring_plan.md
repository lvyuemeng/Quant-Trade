# East Money Provider Refactoring Plan (Final)

## Requirements
1. **Protocol for dispatch** (duck typing) - no "Protocol" suffix in naming
2. **Interface names**: `Fetcher`, `Parser`, `Builder` (no suffix)
3. **Concrete implementations**: `EastMoneyFetch`, `EastMoneyParse`, `EastMoneyBuilder`
4. **Use Polars** (like `akshare.py`)
5. **English naming** only
6. **Fetcher independent** - instantiated separately, no coupling
7. **No inner Fetcher exposed** in public interface
8. **All tests** in `tests/test_client.py`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EastMoney                                │
│  ┌─────────────────┐                                            │
│  │  Fetcher[Proto] │  # Duck-typing protocol                    │
│  │  ◄── EastMoney  │  # Concrete: EastMoneyFetch                 │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Parser[Proto]  │  # Duck-typing protocol                    │
│  │  ◄── EastMoney  │  # Concrete: EastMoneyParse                 │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Builder[Proto] │  # Duck-typing protocol                    │
│  │  ◄── EastMoney  │  # Concrete: EastMoneyBuilder                │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Protocol Interfaces (Duck Typing)

```python
# Protocol Fetcher - no Protocol suffix
class Fetcher:
    """Protocol for fetching - duck typing interface"""
    
    def fetch_initial(self, url: str, params: dict) -> dict: ...
    def fetch_page(self, url: str, params: dict, page: int) -> dict: ...
    def close(self) -> None: ...

# Protocol Parser
class Parser:
    """Protocol for parsing"""
    
    def parse(self, raw: dict) -> list[dict]: ...
    def clean(self, data: list[dict]) -> pl.DataFrame: ...

# Protocol Builder
class Builder:
    """Protocol for building output"""
    
    def rename(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def convert_types(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def reorder(self, df: pl.DataFrame) -> pl.DataFrame: ...
```

## Concrete Implementations

```python
class EastMoneyFetch:
    """Concrete Fetcher - independent, network only"""
    
    def __init__(
        self,
        delay_range: tuple[float, float] = (0.5, 1.5),
        max_retries: int = 3,
    ): ...
    
    def fetch_initial(self, url: str, params: dict) -> dict: ...
    def fetch_page(self, url: str, params: dict, page: int) -> dict: ...
    def close(self) -> None: ...


class EastMoneyParse:
    """Concrete Parser"""
    
    def parse(self, raw: dict) -> list[dict]: ...
    def clean(self, data: list[dict]) -> pl.DataFrame: ...


class EastMoneyBuilder:
    """Concrete Builder for LRB data"""
    
    COLUMN_MAPPING: dict[str, str]
    NUMERIC_COLS: list[str]
    OUTPUT_COLS: list[str]
    
    def rename(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def convert_types(self, df: pl.DataFrame) -> pl.DataFrame: ...
    def reorder(self, df: pl.DataFrame) -> pl.DataFrame: ...
```

## Main Class (No Inner Fetcher Exposed)

```python
class EastMoney:
    """Main interface - Fetcher not exposed, composed internally"""
    
    def __init__(
        self,
        parser: Parser | None = None,
        builder: Builder | None = None,
        delay_range: tuple[float, float] = (0.5, 1.5),
        max_retries: int = 3,
    ): ...
    
    def quarterly(self, year: int, quarter: int) -> pl.DataFrame: ...
    def close(self) -> None: ...
    
    def __enter__(self) -> EastMoney: ...
    def __exit__(self, *args) -> None: ...
```

## Backward Compatibility

```python
def stock_lrb_em(date: str = "20240331", **kwargs) -> pl.DataFrame:
    """Backward compatible - returns Polars DataFrame"""
    year = int(date[:4])
    quarter = (int(date[4:6]) - 1) // 3 + 1
    
    client = EastMoney()
    return client.quarterly(year, quarter)
```

## Test Structure (tests/test_client.py)

```python
"""Tests for East Money provider - all tests in one file."""

import pytest
import polars as pl
from unittest.mock import Mock, patch, MagicMock

from quant_trade.provider.client import (
    EastMoney,
    EastMoneyFetch,
    EastMoneyParse,
    EastMoneyBuilder,
    Fetcher,
    Parser,
    Builder,
)


# Section 1: Test Protocols (duck typing)
# Section 2: Test EastMoneyFetch (mock HTTP)
# Section 3: Test EastMoneyParse (mock parsing)
# Section 4: Test EastMoneyBuilder (mock building)
# Section 5: Test EastMoney (integration)
# Section 6: Test Backward Compatibility
```

## Anti-Blocking Features

1. Randomized delays between requests
2. User-Agent rotation
3. Session pooling with retry strategy
4. Concurrent fetching with semaphore control
