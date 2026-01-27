"""Data validation and quality checks module.

Implements the data quality checks specified in docs/table.md for Chinese A-share data.
"""

from datetime import date
from typing import Any

import polars as pl

from quant_trade.utils.logger import log


class DataValidator:
    """Comprehensive data validation for raw financial data."""

    def __init__(self):
        self.quality_checks = []

    def check_completeness(
        self,
        df: pl.DataFrame,
        required_columns: list[str],
        min_completeness: float = 0.95,
    ) -> dict[str, Any]:
        """Check data completeness for required columns."""
        results = {
            "completeness_score": 0.0,
            "missing_counts": {},
            "passed": True,
            "issues": [],
        }

        total_rows = len(df)
        if total_rows == 0:
            results["passed"] = False
            results["issues"].append("DataFrame is empty")
            return results

        missing_counts = {}
        for col in required_columns:
            if col not in df.columns:
                missing_counts[col] = total_rows
            else:
                missing_counts[col] = df[col].null_count()

        results["missing_counts"] = missing_counts

        # Calculate overall completeness
        total_missing = sum(missing_counts.values())
        total_possible_values = total_rows * len(required_columns)
        if total_possible_values > 0:
            results["completeness_score"] = 1.0 - (
                total_missing / total_possible_values
            )

        if results["completeness_score"] < min_completeness:
            results["passed"] = False
            results["issues"].append(
                f"Completeness score {results['completeness_score']:.2%} below threshold {min_completeness:.2%}"
            )

        return results

    def check_daily_data_coverage(
        self, df: pl.DataFrame, min_coverage: float = 0.95
    ) -> dict[str, Any]:
        """Check daily data coverage (for daily OHLCV data)."""
        results = {"passed": True, "issues": [], "coverage": 0.0}

        if "date" not in df.columns:
            results["passed"] = False
            results["issues"].append("No 'date' column found")
            return results

        # Get date range
        date_range = df.select(
            [
                pl.col("date").min().alias("min_date"),
                pl.col("date").max().alias("max_date"),
            ]
        ).row(0)

        min_date, max_date = date_range
        if min_date is None or max_date is None:
            results["passed"] = False
            results["issues"].append("Date range is empty")
            return results

        # Calculate expected business days (approximate)
        import pandas as pd

        date_range_pd = pd.date_range(
            start=min_date, end=max_date, freq="B"
        )  # Business days
        expected_days = len(date_range_pd)
        actual_days = df.select(pl.col("date")).unique().height

        coverage = actual_days / expected_days if expected_days > 0 else 0
        results["coverage"] = coverage

        if coverage < min_coverage:
            results["passed"] = False
            results["issues"].append(
                f"Daily coverage {coverage:.2%} below threshold {min_coverage:.2%}"
            )

        return results

    def check_fundamental_announcement_dates(self, df: pl.DataFrame) -> dict[str, Any]:
        """Check that fundamental data has proper announcement dates (critical for point-in-time)."""
        results = {"passed": True, "issues": []}

        if "announcement_date" not in df.columns:
            results["passed"] = False
            results["issues"].append(
                "No 'announcement_date' column found in fundamentals data"
            )
            return results

        # Check for null announcement dates
        null_announcement_dates = df.select(
            pl.col("announcement_date").null_count()
        ).item()
        if null_announcement_dates > 0:
            results["passed"] = False
            results["issues"].append(
                f"Found {null_announcement_dates} records without announcement dates"
            )

        # Check that announcement dates are in the past relative to data date (if date column exists)
        if "date" in df.columns:
            future_announcements = df.filter(
                pl.col("announcement_date") > pl.col("date")
            ).height

            if future_announcements > 0:
                results["passed"] = False
                results["issues"].append(
                    f"Found {future_announcements} records with future announcement dates (look-ahead bias)"
                )

        return results

    def check_consistency(self, df: pl.DataFrame) -> dict[str, Any]:
        """Check data consistency (e.g., ROE between -100% and 100%, turnover rate limits)."""
        results = {"passed": True, "issues": [], "checks": {}}

        checks_performed = {}

        # Check turnover rate limits (should be between 0 and reasonable upper limit)
        if "turnover_rate" in df.columns:
            turnover_outliers = df.filter(
                (pl.col("turnover_rate") < 0) | (pl.col("turnover_rate") > 100)
            ).height
            checks_performed["turnover_rate"] = turnover_outliers
            if turnover_outliers > 0:
                results["passed"] = False
                results["issues"].append(
                    f"Found {turnover_outliers} turnover_rate values outside valid range [0, 100]"
                )

        # Check price continuity (detect jumps > 50% without splits)
        if "close" in df.columns and "adj_factor" not in df.columns:
            # Without adjustment factor, check for large price jumps
            df_with_lag = df.sort("date").with_columns(
                pl.col("close").shift(1).over("ts_code").alias("prev_close")
            )

            if "prev_close" in df_with_lag.columns:
                large_jumps = df_with_lag.filter(
                    (pl.col("prev_close").is_not_null())
                    & (pl.col("close").is_not_null())
                    & (pl.col("prev_close") != 0)
                    & ((pl.col("close") / pl.col("prev_close")).abs() > 1.5)
                    & ((pl.col("close") / pl.col("prev_close")).abs() < 0.5)
                ).height

                checks_performed["price_continuity"] = large_jumps
                if large_jumps > 0:
                    results["passed"] = False
                    results["issues"].append(
                        f"Found {large_jumps} large price jumps (>50%) that may indicate data errors"
                    )

        # Check ROE limits if available
        if "net_profit" in df.columns and "total_equity" in df.columns:
            df_with_roe = df.filter(
                (pl.col("total_equity").is_not_null()) & (pl.col("total_equity") != 0)
            ).with_columns((pl.col("net_profit") / pl.col("total_equity")).alias("roe"))

            roe_outliers = df_with_roe.filter(
                (pl.col("roe").is_not_null())
                & (
                    (pl.col("roe") < -10.0) | (pl.col("roe") > 10.0)
                )  # ROE > 1000% or < -1000%
            ).height

            checks_performed["roe_limits"] = roe_outliers
            if roe_outliers > 0:
                results["passed"] = False
                results["issues"].append(
                    f"Found {roe_outliers} ROE values outside reasonable range [-10, 10]"
                )

        results["checks"] = checks_performed
        return results

    def check_timeliness(
        self, df: pl.DataFrame, max_staleness_days: int = 1
    ) -> dict[str, Any]:
        """Check data freshness and timeliness."""
        results = {"passed": True, "issues": []}

        if "date" not in df.columns:
            results["passed"] = False
            results["issues"].append("No 'date' column found for timeliness check")
            return results

        latest_date = df.select(pl.col("date").max()).item()
        if latest_date is None:
            results["passed"] = False
            results["issues"].append("No valid dates found in data")
            return results

        today = date.today()
        days_stale = (today - latest_date).days

        if days_stale > max_staleness_days:
            results["passed"] = False
            results["issues"].append(
                f"Data is {days_stale} days stale (max allowed: {max_staleness_days})"
            )

        return results

    def check_tradeability_flags(self, df: pl.DataFrame) -> dict[str, Any]:
        """Check tradeability flags (suspended, limit up/down)."""
        results = {"passed": True, "issues": [], "flags_summary": {}}

        flags_summary = {}

        # Check for suspended stocks flag
        if "is_suspended" in df.columns:
            suspended_count = df.select(pl.col("is_suspended").sum()).item()
            flags_summary["suspended"] = suspended_count

        # Check for limit up/down flags
        if "is_limit_up" in df.columns:
            limit_up_count = df.select(pl.col("is_limit_up").sum()).item()
            flags_summary["limit_up"] = limit_up_count

        if "is_limit_down" in df.columns:
            limit_down_count = df.select(pl.col("is_limit_down").sum()).item()
            flags_summary["limit_down"] = limit_down_count

        results["flags_summary"] = flags_summary
        return results

    def check_accounting_identity(
        self, df: pl.DataFrame, tolerance: float = 0.01
    ) -> dict[str, Any]:
        """Check accounting identity: Assets = Liabilities + Equity."""
        results = {"passed": True, "issues": [], "failed_checks": 0}

        required_cols = ["total_assets", "total_liabilities", "total_equity"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            results["passed"] = False
            results["issues"].append(
                f"Missing required columns for accounting identity check: {missing_cols}"
            )
            return results

        # Filter out rows with null values
        df_filtered = df.filter(
            pl.col("total_assets").is_not_null()
            & pl.col("total_liabilities").is_not_null()
            & pl.col("total_equity").is_not_null()
        )

        if df_filtered.height == 0:
            results["issues"].append("No complete rows for accounting identity check")
            return results

        # Calculate difference
        df_with_diff = df_filtered.with_columns(
            (
                pl.col("total_assets")
                - pl.col("total_liabilities")
                - pl.col("total_equity")
            )
            .abs()
            .alias("identity_diff")
        )

        # Count failures (difference > tolerance)
        failed_checks = df_with_diff.filter(pl.col("identity_diff") > tolerance).height

        results["failed_checks"] = failed_checks

        if failed_checks > 0:
            results["passed"] = False
            results["issues"].append(
                f"Found {failed_checks} records violating accounting identity (Assets = Liabilities + Equity) with tolerance {tolerance}"
            )

        return results

    def run_all_checks(
        self, df: pl.DataFrame, data_type: str = "market"
    ) -> dict[str, Any]:
        """Run all applicable quality checks based on data type."""
        log.info(
            f"Running data quality checks for {data_type} data with {len(df)} rows"
        )

        results = {"overall_passed": True, "check_results": {}}

        # Determine required columns based on data type
        if data_type == "market":
            required_columns = [
                "date",
                "ts_code",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            completeness_result = self.check_completeness(df, required_columns)
            results["check_results"]["completeness"] = completeness_result

            daily_coverage_result = self.check_daily_data_coverage(df)
            results["check_results"]["daily_coverage"] = daily_coverage_result

            consistency_result = self.check_consistency(df)
            results["check_results"]["consistency"] = consistency_result

            timeliness_result = self.check_timeliness(df)
            results["check_results"]["timeliness"] = timeliness_result

            tradeability_result = self.check_tradeability_flags(df)
            results["check_results"]["tradeability"] = tradeability_result

        elif data_type == "fundamentals":
            required_columns = ["ts_code", "announcement_date"]
            completeness_result = self.check_completeness(df, required_columns)
            results["check_results"]["completeness"] = completeness_result

            announcement_result = self.check_fundamental_announcement_dates(df)
            results["check_results"]["announcement_dates"] = announcement_result

            consistency_result = self.check_consistency(df)
            results["check_results"]["consistency"] = consistency_result

            accounting_result = self.check_accounting_identity(df)
            results["check_results"]["accounting_identity"] = accounting_result

        else:  # General check
            completeness_result = self.check_completeness(df, df.columns)
            results["check_results"]["completeness"] = completeness_result

        # Overall result
        for check_name, check_result in results["check_results"].items():
            if not check_result.get("passed", True):
                results["overall_passed"] = False
                break

        if results["overall_passed"]:
            log.info("All data quality checks PASSED")
        else:
            log.warning("Some data quality checks FAILED")
            for check_name, check_result in results["check_results"].items():
                if not check_result.get("passed", True):
                    log.warning(
                        f"Failed check '{check_name}': {check_result.get('issues', [])}"
                    )

        return results


def validate_data_quality(
    df: pl.DataFrame, data_type: str = "market"
) -> dict[str, Any]:
    """Convenience function to run data quality checks."""
    validator = DataValidator()
    return validator.run_all_checks(df, data_type)
