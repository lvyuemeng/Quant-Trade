"""Feature engineering module for quantitative trading.

Implements the feature calculation pipeline as specified in docs/table.md,
including quality metrics, leverage ratios, growth metrics, valuation ratios,
behavioral indicators, and macro context features.
"""

import polars as pl

from quant_trade.utils.logger import log


class FeatureEngineer:
    """Feature engineering class to calculate various financial indicators."""

    def __init__(self):
        pass

    def calculate_quality_metrics(self, fundamentals_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate quality metrics as per docs/table.md Table 3."""
        log.info("Calculating quality metrics")

        # Calculate ROE, ROA, ROIC, margins
        result_df = fundamentals_df.with_columns(
            [
                # ROE (Return on Equity)
                (pl.col("net_profit") / pl.col("total_equity"))
                .alias("roe")
                .filter(
                    pl.col("total_equity").is_not_null() & (pl.col("total_equity") != 0)
                ),
                # ROA (Return on Assets)
                (pl.col("net_profit") / pl.col("total_assets"))
                .alias("roa")
                .filter(
                    pl.col("total_assets").is_not_null() & (pl.col("total_assets") != 0)
                ),
                # Gross Margin
                (pl.col("gross_profit") / pl.col("total_revenue"))
                .alias("gross_margin")
                .filter(
                    pl.col("total_revenue").is_not_null()
                    & (pl.col("total_revenue") != 0)
                    & pl.col("gross_profit").is_not_null()
                ),
                # Operating Margin
                (pl.col("operating_profit") / pl.col("total_revenue"))
                .alias("operating_margin")
                .filter(
                    pl.col("total_revenue").is_not_null()
                    & (pl.col("total_revenue") != 0)
                    & pl.col("operating_profit").is_not_null()
                ),
                # Net Margin
                (pl.col("net_profit") / pl.col("total_revenue"))
                .alias("net_margin")
                .filter(
                    pl.col("total_revenue").is_not_null()
                    & (pl.col("total_revenue") != 0)
                ),
            ]
        )

        return result_df

    def calculate_leverage_metrics(self, fundamentals_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate leverage and safety metrics as per docs/table.md Table 3."""
        log.info("Calculating leverage metrics")

        result_df = fundamentals_df.with_columns(
            [
                # Debt to Asset Ratio
                (
                    (pl.col("total_debt") / pl.col("total_assets")).alias(
                        "debt_to_asset"
                    )
                ).filter(
                    pl.col("total_assets").is_not_null()
                    & (pl.col("total_assets") != 0)
                    & pl.col("total_debt").is_not_null()
                ),
                # Debt to Equity Ratio
                (
                    (pl.col("total_debt") / pl.col("total_equity")).alias(
                        "debt_to_equity"
                    )
                ).filter(
                    pl.col("total_equity").is_not_null()
                    & (pl.col("total_equity") != 0)
                    & pl.col("total_debt").is_not_null()
                ),
                # Current Ratio
                (
                    (pl.col("current_assets") / pl.col("current_liabilities")).alias(
                        "current_ratio"
                    )
                ).filter(
                    pl.col("current_liabilities").is_not_null()
                    & (pl.col("current_liabilities") != 0)
                    & pl.col("current_assets").is_not_null()
                ),
            ]
        )

        return result_df

    def calculate_growth_metrics(self, fundamentals_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate growth metrics as per docs/table.md Table 3."""
        log.info("Calculating growth metrics")

        # This requires historical data to calculate YoY and QoQ growth
        # We'll calculate using shift operations to get previous period values
        result_df = fundamentals_df.sort(["ts_code", "announcement_date"]).with_columns(
            [
                # YoY growth (requires same quarter from previous year)
                (
                    (
                        pl.col("net_profit")
                        - pl.col("net_profit").shift(1).over("ts_code")
                    )
                    / pl.col("net_profit").shift(1).over("ts_code")
                )
                .alias("profit_growth_yoy")
                .filter(
                    pl.col("net_profit").shift(1).over("ts_code").is_not_null()
                    & (pl.col("net_profit").shift(1).over("ts_code") != 0)
                ),
                (
                    (
                        pl.col("total_revenue")
                        - pl.col("total_revenue").shift(1).over("ts_code")
                    )
                    / pl.col("total_revenue").shift(1).over("ts_code")
                )
                .alias("revenue_growth_yoy")
                .filter(
                    pl.col("total_revenue").shift(1).over("ts_code").is_not_null()
                    & (pl.col("total_revenue").shift(1).over("ts_code") != 0)
                ),
            ]
        )

        return result_df

    def calculate_valuation_metrics(
        self, market_df: pl.DataFrame, fundamentals_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate valuation metrics as per docs/table.md Table 4."""
        log.info("Calculating valuation metrics")

        # Join market and fundamentals data
        joined_df = market_df.join(fundamentals_df, on=["ts_code", "date"], how="left")

        result_df = joined_df.with_columns(
            [
                # P/E TTM (requires TTM earnings data which might be in fundamentals)
                ((pl.col("amount") / pl.col("volume")) / pl.col("net_profit"))
                .alias("pe_ttm")  # Simplified calculation
                .filter(
                    pl.col("net_profit").is_not_null() & (pl.col("net_profit") != 0)
                ),
                # P/B (Price to Book)
                ((pl.col("amount") / pl.col("volume")) / pl.col("total_equity"))
                .alias("pb")
                .filter(
                    pl.col("total_equity").is_not_null() & (pl.col("total_equity") != 0)
                ),
                # P/S (Price to Sales) - would need revenue data
                ((pl.col("amount") / pl.col("volume")) / pl.col("total_revenue"))
                .alias("ps")
                .filter(
                    pl.col("total_revenue").is_not_null()
                    & (pl.col("total_revenue") != 0)
                ),
            ]
        )

        return result_df

    def calculate_behavioral_features(self, market_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate behavioral features as per docs/table.md Table 5."""
        log.info("Calculating behavioral features")

        # Sort by date and ts_code to ensure proper time series operations
        sorted_df = market_df.sort(["ts_code", "date"])

        result_df = sorted_df.with_columns(
            [
                # 1-month return (20 trading days approx)
                (
                    pl.col("close")
                    .pct_change()
                    .rolling_mean(window_size=20)
                    .alias("return_1m")
                ),
                # 3-month return (60 trading days approx)
                (
                    pl.col("close")
                    .pct_change()
                    .rolling_mean(window_size=60)
                    .alias("return_3m")
                ),
                # 12-month return (252 trading days approx)
                (
                    pl.col("close")
                    .pct_change()
                    .rolling_mean(window_size=252)
                    .alias("return_12m")
                ),
                # Turnover rate (already in the data, but we can calculate moving averages)
                (
                    pl.col("turnover_rate")
                    .rolling_mean(window_size=20)
                    .alias("turnover_ma20")
                ),
                # Abnormal turnover
                (
                    (
                        pl.col("turnover_rate")
                        / pl.col("turnover_rate").rolling_mean(window_size=20)
                    ).alias("turnover_ratio")
                ),
                # Volatility (20-day rolling standard deviation of returns)
                (
                    pl.col("close")
                    .pct_change()
                    .rolling_std(window_size=20)
                    .alias("volatility_20d")
                ),
                # RSI (Relative Strength Index) - simplified calculation
                # Calculate up and down movements
                pl.when(pl.col("close").diff() > 0)
                .then(pl.col("close").diff())
                .otherwise(0)
                .alias("up_move"),
                pl.when(pl.col("close").diff() < 0)
                .then(pl.col("close").diff().abs())
                .otherwise(0)
                .alias("down_move"),
            ]
        )

        # Calculate RSI separately since it requires more complex logic
        rsi_calc = result_df.with_columns(
            [
                pl.col("up_move").rolling_mean(window_size=14).alias("avg_up"),
                pl.col("down_move").rolling_mean(window_size=14).alias("avg_down"),
            ]
        ).with_columns(
            [
                pl.when(pl.col("avg_down") != 0)
                .then(100 - (100 / (1 + (pl.col("avg_up") / pl.col("avg_down")))))
                .otherwise(100)
                .alias("rsi_14")
            ]
        )

        # Clean up temporary columns
        final_df = rsi_calc.drop(["up_move", "down_move", "avg_up", "avg_down"])

        return final_df

    def calculate_macro_features(self, macro_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate macro context features as per docs/table.md Table 6."""
        log.info("Calculating macro features")

        result_df = macro_df.with_columns(
            [
                # Time-series Z-scores for various macro indicators
                # Calculate Z-score as (value - mean) / std, using a rolling window
                (
                    (
                        pl.col("shibor_3m")
                        - pl.col("shibor_3m").rolling_mean(window_size=252)
                    )
                    / pl.col("shibor_3m").rolling_std(window_size=252)
                ).alias("shibor_3m_zscore"),
                (
                    (
                        pl.col("northbound_flow")
                        - pl.col("northbound_flow").rolling_mean(window_size=60)
                    )
                    / pl.col("northbound_flow").rolling_std(window_size=60)
                ).alias("northbound_flow_zscore"),
                # Time-series percentiles (simplified using rolling quantiles)
                (
                    pl.col("csi300_pe")
                    .rolling_quantile(quantile=0.5, interpolation="linear")
                    .alias("csi300_pe_percentile")
                ),
            ]
        )

        return result_df

    def winsorize_features(
        self, df: pl.DataFrame, columns: list[str], limits: tuple = (0.025, 0.975)
    ) -> pl.DataFrame:
        """Winsorize features to handle outliers as specified in docs/table.md."""
        log.info(f"Winsorizing features: {columns}")

        result_df = df.clone()

        for col in columns:
            if col in df.columns:
                # Calculate quantiles
                quantiles = df.select(
                    [
                        pl.col(col).quantile(limits[0]).alias("lower_quantile"),
                        pl.col(col).quantile(limits[1]).alias("upper_quantile"),
                    ]
                ).row(0)

                lower_q, upper_q = quantiles

                # Winsorize: cap values at quantile limits
                result_df = result_df.with_columns(
                    [
                        pl.when(pl.col(col) < lower_q)
                        .then(lower_q)
                        .when(pl.col(col) > upper_q)
                        .then(upper_q)
                        .otherwise(pl.col(col))
                        .alias(col)
                    ]
                )

        return result_df

    def cross_sectional_normalize(
        self, df: pl.DataFrame, feature_cols: list[str], group_col: str = "sw_l1_code"
    ) -> pl.DataFrame:
        """Perform cross-sectional normalization within industries as specified in docs/table.md."""
        log.info(f"Cross-sectional normalizing features: {feature_cols} by {group_col}")

        # Calculate group-wise mean and std for each feature
        stats = df.group_by(group_col).agg(
            [
                *[pl.col(col).mean().alias(f"{col}_mean") for col in feature_cols],
                *[pl.col(col).std().alias(f"{col}_std") for col in feature_cols],
            ]
        )

        # Join stats back to original dataframe
        result_df = df.join(stats, on=group_col, how="left")

        # Calculate z-scores: (X - μ_industry) / σ_industry
        for col in feature_cols:
            result_df = result_df.with_columns(
                [
                    (
                        (pl.col(col) - pl.col(f"{col}_mean")) / pl.col(f"{col}_std")
                    ).alias(f"{col}_zscore")
                ]
            )

        # Clean up temporary columns
        temp_cols = [f"{col}_mean" for col in feature_cols] + [
            f"{col}_std" for col in feature_cols
        ]
        result_df = result_df.drop(temp_cols)

        return result_df

    def calculate_all_features(
        self,
        market_data: pl.DataFrame,
        fundamentals_data: pl.DataFrame,
        macro_data: pl.DataFrame,
        industry_data: pl.DataFrame,
    ) -> dict[str, pl.DataFrame]:
        """Calculate all features using the pipeline specified in docs/table.md."""
        log.info("Starting full feature calculation pipeline")

        # Calculate individual feature sets
        quality_features = self.calculate_quality_metrics(fundamentals_data)
        leverage_features = self.calculate_leverage_metrics(fundamentals_data)
        growth_features = self.calculate_growth_metrics(fundamentals_data)
        behavioral_features = self.calculate_behavioral_features(market_data)
        macro_features = self.calculate_macro_features(macro_data)

        # Combine quality and leverage features (both from fundamentals)
        combined_fundamentals = quality_features.join(
            leverage_features, on=["ts_code", "announcement_date"], how="outer"
        )
        combined_fundamentals = combined_fundamentals.join(
            growth_features, on=["ts_code", "announcement_date"], how="outer"
        )

        # Join with market data for valuation calculations
        valuation_features = self.calculate_valuation_metrics(
            market_data, combined_fundamentals
        )

        # Apply winsorization to handle outliers
        feature_cols_to_winsorize = [
            "roe",
            "roa",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "debt_to_asset",
            "debt_to_equity",
            "current_ratio",
            "profit_growth_yoy",
            "revenue_growth_yoy",
        ]

        winsorized_fundamentals = self.winsorize_features(
            combined_fundamentals,
            [
                col
                for col in feature_cols_to_winsorize
                if col in combined_fundamentals.columns
            ],
        )

        # Perform cross-sectional normalization (industry-based)
        normalization_features = [
            "roe",
            "roa",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "debt_to_asset",
            "debt_to_equity",
            "current_ratio",
            "profit_growth_yoy",
            "revenue_growth_yoy",
        ]

        # Need to join with industry data for cross-sectional normalization
        if "sw_l1_code" in industry_data.columns:
            industry_joined = winsorized_fundamentals.join(
                industry_data.select(["ts_code", "date", "sw_l1_code"]),
                on=["ts_code", "date"],
                how="left",
            )

            normalized_features = self.cross_sectional_normalize(
                industry_joined,
                [
                    col
                    for col in normalization_features
                    if col in industry_joined.columns
                ],
            )
        else:
            normalized_features = winsorized_fundamentals

        results = {
            "fundamentals": normalized_features,
            "valuation": valuation_features,
            "behavioral": behavioral_features,
            "macro": macro_features,
            "complete_feature_set": None,  # Will be created by joining all
        }

        # Create the complete feature set by joining all data
        complete = market_data

        if len(normalized_features) > 0:
            complete = complete.join(
                normalized_features, on=["ts_code", "date"], how="left"
            )

        if len(valuation_features) > 0:
            complete = complete.join(
                valuation_features, on=["ts_code", "date"], how="left"
            )

        if len(behavioral_features) > 0:
            complete = complete.join(
                behavioral_features, on=["ts_code", "date"], how="left"
            )

        if len(macro_features) > 0:
            complete = complete.join(macro_features, on=["date"], how="left")

        results["complete_feature_set"] = complete

        log.info("Feature calculation pipeline completed")
        return results


def create_features(
    market_data: pl.DataFrame,
    fundamentals_data: pl.DataFrame,
    macro_data: pl.DataFrame,
    industry_data: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Convenience function to create all features."""
    engineer = FeatureEngineer()
    return engineer.calculate_all_features(
        market_data, fundamentals_data, macro_data, industry_data
    )
