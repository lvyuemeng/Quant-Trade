from datetime import date, datetime
from typing import Literal

import polars as pl

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
    null_threshold: float = 0.7,
) -> pl.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found")

    expr = pl.col(date_col)
    dtype = df.schema[date_col]

    # ---- Fast path: already temporal ----
    if dtype in (pl.Date, pl.Datetime):
        out = expr
        if target_type == "date" and dtype == pl.Datetime:
            out = out.dt.date()
        elif target_type == "datetime" and dtype == pl.Date:
            out = out.cast(pl.Datetime)
        return df.with_columns(out.alias(date_col))

    # ---- Non-string: direct cast ----
    if dtype != pl.Utf8:
        return df.with_columns(expr.cast(pl.Date, strict=strict).alias(date_col))

    n = len(df)
    threshold = int(n * null_threshold)

    def score(e: pl.Expr) -> int:
        return n - df.select(e.null_count()).item()

    # ---- Parse candidates (ordered, unified) ----
    candidates: list[tuple[pl.Expr, type[pl.Date | pl.Datetime]]] = []

    # Date formats
    for fmt in (
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
    ):
        candidates.append((expr.str.strptime(pl.Date, fmt, strict=False), pl.Date))

    # Datetime formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        candidates.append(
            (expr.str.strptime(pl.Datetime, fmt, strict=False), pl.Datetime)
        )

    # Last-resort auto datetime
    candidates.append((expr.str.to_datetime(strict=False), pl.Datetime))

    # ---- Select best candidate ----
    best_expr = None
    best_type = None
    best_score = -1

    for cand, cand_type in candidates:
        s = score(cand)
        if s > best_score:
            best_expr, best_type, best_score = cand, cand_type, s
        if s >= threshold:
            break

    if best_expr is None or best_score <= 0:
        if strict:
            raise ValueError(f"Failed to parse '{date_col}' as date/datetime")
        return df

    # ---- Normalize output type ----
    out = best_expr
    if target_type == "date" and best_type == pl.Datetime:
        out = out.dt.date()
    elif target_type == "datetime" and best_type == pl.Date:
        out = out.cast(pl.Datetime)

    return df.with_columns(out.alias(date_col))
