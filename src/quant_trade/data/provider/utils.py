import re
from datetime import date, datetime
from typing import Literal

import polars as pl

type DateLike = str | date
type OptDateLike = DateLike | None
type Period = Literal["daily", "weekly", "monthly"]
type AdjustCN = Literal["hfq", "qfq"]


def normal_stock_code(raw_code: str, length: int = 6) -> str:
    """
    Normalize A-share stock codes by handling prefixes/suffixes and padding.

    Args:
        raw_code: Raw stock code (e.g., '600313.SH', 'SZ000001', '000002.SZ')
        expect_length: Expected length after normalization (default: 6 for A-shares)

    Returns:
        Normalized numeric stock code (e.g., '600313', '000001', '000002')

    Raises:
        ValueError: If no numeric portion found or code can't be normalized
    """
    match = re.search(r"(\d+)", raw_code)
    if not match:
        raise ValueError(f"No numeric portion found in code: '{raw_code}'")

    matched = match.group(1).zfill(length)

    # Truncate if longer than expected (unusual cases)
    if len(matched) > length:
        matched = matched[-length:]

    return matched


def _normal_stock_code_expr(
    code_col: str | pl.Expr, length: int = 6, alias: str | None = None
) -> pl.Expr:
    """
    Vectorized stock code normalization using Polars expressions.

    Args:
        code_col: Column name or expression containing stock codes
        expect_length: Expected length after normalization (default: 6 for A-shares)
        output_col: Optional output column name (default: replaces input column)

    Returns:
        Polars expression for use in with_columns()
    """
    match code_col:
        case str():
            expr = pl.col(code_col)
        case _:  # expr
            expr = code_col

    matched = (
        expr.str.extract(r"(\d+)", 1).str.pad_start(length, "0").str.slice(-length)
    )

    # Return with optional alias
    if alias:
        return matched.alias(alias)
    return matched


def normal_stock_code_df(
    df: pl.DataFrame, col: str = "ts_code", length: int = 6, inplace: bool = True
) -> pl.DataFrame:
    """
    Batch normalize stock codes in a DataFrame.

    Args:
        df: Input DataFrame
        code_col: Column name containing stock codes
        expect_length: Expected length after normalization
        inplace: If True, replaces original column; if False, adds new column

    Returns:
        DataFrame with normalized codes
    """
    output_col = col if inplace else f"{col}_normalized"

    return df.with_columns(_normal_stock_code_expr(col, length, output_col))


def date_f_datelike(date: OptDateLike) -> date:
    match date:
        case None:
            return datetime.now().date()
        case str():
            return datetime.strptime(date, "%Y%m%d").date()
        case _:  # date
            return date


def str_f_datelike(date: OptDateLike) -> str:
    match date:
        case None:
            return datetime.now().date().strftime("%Y%m%d")
        case str():
            return date
        case _:  # date
            return date.strftime("%Y%m%d")


def clean_timedf(
    df: pl.DataFrame,
    rename: dict,
    drop: list | str | None = None,
    select: list | None = None,  # Optional columns to select for consistent schema
    date_col: str | None = None,  # Column to normalize
    date_format: str | None = None,  # Optional format string
) -> pl.DataFrame:
    """
    Internal helper to clean and standardize a stock DataFrame.

    Args:
        df: Raw Polars DataFrame from akshare
        rename_map: Dictionary mapping original to new column names
        drop_cols: Column(s) to drop (string or list of strings)
        date_col: Name of the column containing date strings
        date_format: Optional format string for parsing (e.g., "%Y/%m/%d")
        select_cols: Optional list of columns to select for consistent schema
    """
    res_df = df.rename(rename)

    if drop:
        res_df = df.drop(drop)

    if select:
        existing_cols = [col for col in select if col in res_df.columns]
        res_df = res_df.select(existing_cols)

    if date_col is None or date_col not in res_df.columns:
        return res_df

    match res_df.schema[date_col]:
        case pl.Date:
            return res_df
        case pl.Datetime:
            # Convert datetime to date
            date_expr = pl.col(date_col).dt.date()
            res_df = res_df.with_columns(date_expr)
            return res_df
        case pl.Utf8:
            # Parse string to date
            if date_format:
                date_expr = pl.col(date_col).str.strptime(
                    pl.Date, date_format, strict=False
                )
            else:
                date_expr = pl.col(date_col).str.to_date(strict=False)
            res_df = res_df.with_columns(date_expr)
            return res_df
        case _:
            # Try to cast other types to date
            date_expr = pl.col(date_col).cast(pl.Date, strict=False)
            res_df = res_df.with_columns(date_expr)
            return res_df


def fallback_col(columns: list[str], candidates: list[str], label: str) -> str:
    """
    Find the first column that exists in the list of candidates.

    Args:
        columns: List of available column names
        candidates: List of possible column names to look for
        label: Label for error message

    Returns:
        First existing column name

    Raises:
        KeyError: If none of the candidates exist
    """
    for candidate in candidates:
        if candidate in columns:
            return candidate

    msg = f"Unable to find {label} column. Tried: {candidates}. Available columns: {columns}"
    raise KeyError(msg)
