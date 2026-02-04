from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal

import polars as pl

from quant_trade.config.logger import log

type DateLike = str | date | datetime
type Period = Literal["daily", "weekly", "monthly"]
type Quarter = Literal[1, 2, 3, 4]
type AdjustCN = Literal["hfq", "qfq"]


# ────────────────────────────────────────────────
#           Date conversion helpers
# ────────────────────────────────────────────────
def to_date(d: DateLike) -> date:
    """Convert to date — no None allowed, raise on failure"""
    if isinstance(d, (date, datetime)):
        return d.date() if isinstance(d, datetime) else d

    if isinstance(d, str):
        d = d.strip()
        # Common formats in financial data
        for fmt in (
            "%Y%m%d",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
        ):
            try:
                return datetime.strptime(d, fmt).date()
            except ValueError:
                continue

    raise ValueError(f"Cannot convert to date: {d!r}")


def to_ymd_str(d: DateLike, sep: str = "") -> str:
    """Convert to YYYYMMDD (default) or YYYY-MM-DD"""
    dt = to_date(d)
    if sep:
        return dt.strftime(f"%Y{sep}%m{sep}%d")
    return dt.strftime("%Y%m%d")


# ────────────────────────────────────────────────
#         Stock code normalization (suffix control)
# ────────────────────────────────────────────────
#
#
def normalize_ts_code_str(
    code: str,
    *,
    length: int = 6,
    add_exchange: bool = False,
    exchange: Literal["SH", "SZ", "BJ"] | None = None,
    position: Literal["suffix", "prefix"] = "suffix",
    sep: str = ".",
    case: Literal["upper", "lower"] = "upper",
) -> str:
    """
    Normalize a single stock code string.

    Examples:
        "600519"        -> "600519"
        "600519.SH"     -> "600519"
        "sh600519"      -> "600519"
        add_exchange=True -> "600519.SH"
    """

    # ---- extract digits (fast path, no regex) ----
    digits = "".join(c for c in code if c.isdigit())

    if not digits:
        raise ValueError("Code digit is empty")

    # pad + truncate
    if len(digits) < length:
        digits = digits.zfill(length)
    elif len(digits) > length:
        digits = digits[-length:]

    if not add_exchange:
        return digits

    # ---- determine exchange ----
    if exchange is not None:
        exch = exchange
    else:
        first = digits[0]
        if first == "6":
            exch = "SH"
        elif first in ("0", "3"):
            exch = "SZ"
        elif first in ("4", "8"):
            exch = "BJ"
        else:
            exch = "UNKNOWN"

    if case == "lower":
        exch = exch.lower()
    else:
        exch = exch.upper()

    # ---- assemble ----
    if position == "suffix":
        return f"{digits}{sep}{exch}"
    else:
        return f"{exch}{sep}{digits}"


def normalize_ts_code(
    code: str | pl.Expr,
    *,
    length: int = 6,
    output: str | None = None,
    add_exchange: bool = False,  # whether to add SH/SZ/BJ at all
    exchange: Literal["SH", "SZ", "BJ"] | None = None,
    position: Literal["suffix", "prefix"] = "suffix",
    sep: str = ".",  # ".", "" or others
    case: Literal["upper", "lower"] = "upper",
) -> pl.Expr:
    """
    Normalize stock code to:
    - 6-digit string (default)            e.g. "600519"
    - with exchange as suffix/prefix      e.g. "600519.SH", "sh600519", "600519SH"

    Args:
        code: column name or Expr
        length: usually 6
        output: alias name
        add_exchange: whether to attach exchange
        exchange: explicit exchange (SH/SZ/BJ); if None, guessed
        position: "suffix" or "prefix"
        sep: separator between code and exchange (".", "", "_", etc.)
        case: exchange letter case
    """
    expr = pl.col(code) if isinstance(code, str) else code

    # extract digits → zero-pad → keep last `length`
    digits = expr.str.extract(r"(\d+)", 1).str.pad_start(length, "0").str.slice(-length)

    if not add_exchange:
        out = digits
    else:
        # --- exchange expr ---
        if exchange is not None:
            exch = pl.lit(exchange)
        else:
            first_digit = digits.str.slice(0, 1)
            exch = (
                pl.when(first_digit == "6")
                .then(pl.lit("SH"))
                .when(first_digit.is_in(["0", "3"]))
                .then(pl.lit("SZ"))
                .when(first_digit.is_in(["4", "8"]))
                .then(pl.lit("BJ"))
                .otherwise(pl.lit("UNKNOWN"))
            )

        # case control
        exch = exch.str.to_uppercase() if case == "upper" else exch.str.to_lowercase()

        # combine
        if position == "suffix":
            out = digits + pl.lit(sep) + exch
        else:  # prefix
            out = exch + pl.lit(sep) + digits

    return out.alias(output) if output else out


# ────────────────────────────────────────────────
#               Date column normalization
# ────────────────────────────────────────────────


def normalize_date_column(
    df: pl.DataFrame,
    date_col: str,
    *,
    target_type: Literal["date", "datetime"] = "date",
    strict: bool = False,
    null_threshold: float = 0.7,  # fraction of non-null required
) -> pl.DataFrame:
    """
    Normalize a specific date column — column must exist.

    Args:
        df: DataFrame
        date_col: MUST exist in df
        target_type: "date" or "datetime"
        strict: if True, raise on parsing failure
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    expr = pl.col(date_col)

    # Already correct type?
    if df.schema[date_col] in (pl.Date, pl.Datetime):
        if target_type == "date" and df.schema[date_col] == pl.Datetime:
            expr = expr.dt.date()
        elif target_type == "datetime" and df.schema[date_col] == pl.Date:
            expr = expr.cast(pl.Datetime)
        return df.with_columns(expr.alias(date_col))

    # String → date parsing
    if df.schema[date_col] != pl.Utf8:
        return df.with_columns(expr.cast(pl.Date, strict=strict).alias(date_col))

    n_rows = len(df)
    threshold = int(n_rows * null_threshold)  # non-null required
    best_expr = expr.str.to_date(strict=strict)  # fallback
    best_non_null = 0
    for fmt in (
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
    ):
        candidate = expr.str.strptime(pl.Date, fmt, strict=False)
        # Evaluate only the null count (cheap)
        non_null = n_rows - df.select(candidate.null_count()).item()
        if non_null > best_non_null:
            best_non_null = non_null
            best_expr = candidate
        if non_null >= threshold:
            break

    df = df.with_columns(best_expr.alias(date_col))
    if target_type == "datetime":
        df = df.with_columns(pl.col(date_col).cast(pl.Datetime, strict=strict))

    return df


# ────────────────────────────────────────────────
#                   Quarter utilities
# ────────────────────────────────────────────────


def quarter_end(year: int, quarter: Quarter, as_ymd: bool = True) -> str:
    """Explicit year + quarter → YYYYMMDD or YYYY-MM-DD"""
    ends = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    s = f"{year}{ends[quarter]}"
    if not as_ymd:
        return f"{year}-{ends[quarter][:2]}-{ends[quarter][2:]}"
    return s


def current_quarter_end(as_ymd: bool = True) -> str:
    """Most recent completed quarter end (explicit — no None handling)"""
    today = datetime.now().date()
    year = today.year
    month = today.month

    if month <= 3:
        return quarter_end(year - 1, 4, as_ymd)
    elif month <= 6:
        return quarter_end(year, 1, as_ymd)
    elif month <= 9:
        return quarter_end(year, 2, as_ymd)
    else:
        return quarter_end(year, 3, as_ymd)
