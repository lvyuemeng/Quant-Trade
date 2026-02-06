import datetime
import functools
import re
from collections.abc import Generator
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    cast,
    final,
    runtime_checkable,
)

from ..transform import Quarter


def book_key[T](cls: type[T]) -> type[T]:
    """
    Class decorator that auto-generates BookKey implementation.

    Joins all fields with "_" to create storage key.
    Nested BookKey objects are recursively converted.
    Preserves manually-defined methods (to_key, from_key, etc.).
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")

    # Pre-compute field names for performance
    field_names: tuple[str, ...] = tuple(f.name for f in fields(cls))

    # Check which methods are already defined (to avoid overwriting)
    has_to_key = "to_key" in cls.__dict__

    def _format_value(val: Any) -> str:
        """Convert a value to its string representation."""
        if isinstance(val, BookKey):
            return val.to_key()
        if isinstance(val, datetime.date):
            return val.isoformat()
        return str(val)

    if not has_to_key:

        def to_key(self: T) -> str:
            """Generate key by joining all field values with '_'."""
            parts = [_format_value(getattr(self, name)) for name in field_names]
            return "_".join(parts)

        # Cache to_key for frozen dataclasses (they're immutable)
        dc_params = getattr(cls, "__dataclass_params__", None)
        if dc_params and getattr(dc_params, "frozen", False):
            to_key = functools.lru_cache(maxsize=None)(to_key)

        setattr(cls, "to_key", to_key)

    @classmethod
    def _fields(cls_: type[T]) -> tuple[str, ...]:
        """Return field names."""
        return field_names

    setattr(cls, "_fields", _fields)

    def to_dict(self: T) -> dict[str, Any]:
        """Convert to dictionary with nested BookKey expansion."""
        result: dict[str, Any] = {}
        for name in field_names:
            val = getattr(self, name)
            if isinstance(val, BookKey):
                result[name] = val.to_key()  # Or val.to_dict() if you prefer
            else:
                result[name] = val
        return result

    setattr(cls, "to_dict", to_dict)

    # Enhanced repr showing the key
    # original_repr = cls.__repr__

    def __repr__(self: T) -> str:
        key = getattr(self, "to_key")()
        field_str = ", ".join(f"{name}={getattr(self, name)!r}" for name in field_names)
        return f"{cls.__name__}({field_str})[key={key!r}]"

    setattr(cls, "__repr__", __repr__)

    return cls


@runtime_checkable
class BookKey(Protocol):
    """Structural trait: anything with to_key() is a valid book identifier."""

    def to_key(self) -> str: ...

    @classmethod
    def parse_key(cls: type[Self], key: str) -> Self: ...


@final
@book_key
@dataclass(frozen=True, slots=True)
class QuarterBook:
    """Year-quarter book identifier: '2024_Q1'"""

    year: int
    quarter: Annotated[int, Literal[1, 2, 3, 4]]

    # Pre-compile regex for parsing (class-level optimization)
    _KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(\d{4})_Q([1-4])$")

    def to_key(self) -> str:
        """Override: More efficient than auto-generated version."""
        return f"{self.year}_Q{self.quarter}"

    @classmethod
    def parse_key(cls, key: str) -> Self:
        """Parse key into QuarterBook instance."""
        if not key:
            raise ValueError("Empty key")

        match = cls._KEY_PATTERN.match(key)
        if not match:
            raise ValueError(
                f"Invalid QuarterBook key: {key!r}. Expected format: YYYY_Q[1-4]"
            )

        year = int(match.group(1))
        quarter = int(match.group(2))
        return cls(year, quarter)

    @staticmethod
    def date_range(
        start_date: datetime.date, end_date: datetime.date
    ) -> Generator["QuarterBook"]:
        """Generate QuarterBook instances covering the date range [start_date, end_date]."""
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date")

        start_year = start_date.year
        start_quarter = (start_date.month - 1) // 3 + 1
        end_year = end_date.year
        end_quarter = (end_date.month - 1) // 3 + 1

        current = QuarterBook(start_year, start_quarter)
        end = QuarterBook(end_year, end_quarter)

        while current <= end:
            yield current
            current = current.next()

    def __le__(self, other: Self) -> bool:
        return (self.year, self.quarter) <= (other.year, other.quarter)

    def __lt__(self, other: Self) -> bool:
        return (self.year, self.quarter) < (other.year, other.quarter)

    def __gt__(self, other: Self) -> bool:
        return (self.year, self.quarter) > (other.year, other.quarter)

    def __ge__(self, other: Self) -> bool:
        return (self.year, self.quarter) >= (other.year, other.quarter)

    def next(self) -> Self:
        """Get the next quarter."""
        if self.quarter == 4:
            return type(self)(self.year + 1, 1)
        return type(self)(self.year, self.quarter + 1)

    def previous(self) -> Self:
        """Get the previous quarter."""
        if self.quarter == 1:
            return type(self)(self.year - 1, 4)
        return type(self)(self.year, self.quarter - 1)

    @property
    def literal_quarter(self) -> Quarter:
        """Get quarter as Literal type."""
        return cast(Quarter, self.quarter)

    @property
    def as_tuple(self) -> tuple[int, int]:
        return (self.year, self.quarter)

    @property
    def is_year_end(self) -> bool:
        """Check if this is Q4."""
        return self.quarter == 4


@final
@book_key
@dataclass(frozen=True, slots=True)
class RecordBook:
    """Universe + date: 'CSI500_2024-01-15'"""

    universe: str
    date: datetime.date | None = None

    def to_key(self) -> str:
        """Generate key by joining all field values with '_'."""
        if self.date is not None:
            return f"{self.universe}_{self.date}"
        return self.universe

    @classmethod
    def parse_key(cls, key: str) -> Self:
        """Parse 'CSI500_2024-01-15' -> UniverseDateBook('CSI500', date(2024, 1, 15))"""
        lstr = key.rsplit("_", 1)
        if len(lstr) == 1:
            (universe,) = lstr
            date_str = None
        elif len(lstr) == 2:
            universe, date_str = lstr
        else:
            raise ValueError(f"Invalid UniverseDateBook key: {key!r}")
        return cls(
            universe, datetime.date.fromisoformat(date_str) if date_str else None
        )


@final
class TickerBook(str):
    """A seamless string-based ticker.
    It behaves exactly like '000001_SZ' but has helper methods attached."""

    def __new__(cls, symbol: str) -> Self:
        # Ensure we are creating a string properly
        return super().__new__(cls, symbol)

    def to_key(self) -> str:
        """Returns itself, because it IS a string."""
        return self

    @classmethod
    def parse_key(cls, key: str) -> Self:
        """Parse key into self."""
        return cls(key)

    @property
    def ts_code(self) -> str:
        """Polars/ts_code format: '000001.SZ'"""
        return self.replace("_", ".")


@final
@dataclass(frozen=True, slots=True)
class MacroBook:
    """Placeholder for future macro book identifiers."""

    type Book = Literal["northbound", "marginshort", "shibor", "index", "qvix"]
    macro: Book

    @staticmethod
    def literals() -> list[Book]:
        return ["northbound", "marginshort", "shibor", "index", "qvix"]

    @staticmethod
    def list() -> list["MacroBook"]:
        return [MacroBook(macro=book) for book in MacroBook.literals()]

    def to_key(self) -> str:
        """Generate key by joining all field values with '_'."""
        return self.macro

    @classmethod
    def parse_key(cls, key: str) -> Self:
        """Parse macro book from key."""
        if key not in {"northbound", "marginshort", "shibor", "index", "qvix"}:
            raise ValueError(f"Invalid MacroBook key: {key!r}")
        return cls(macro=cast(MacroBook.Book, key))


class BookQuery:
    """Factory: users call with primitives, get structured BookKeys."""

    @staticmethod
    def quarterly(
        year: int, quarter: Annotated[int, Literal[1, 2, 3, 4]]
    ) -> QuarterBook:
        return QuarterBook(year, quarter)

    @staticmethod
    def universe_date(universe: str, date_val: datetime.date | str) -> RecordBook:
        d = (
            date_val
            if isinstance(date_val, datetime.date)
            else datetime.date.fromisoformat(date_val)
        )
        return RecordBook(universe, d)

    @staticmethod
    def ticker(symbol: str) -> TickerBook:
        return TickerBook(symbol)
