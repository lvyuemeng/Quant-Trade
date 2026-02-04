from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

import lightgbm as lgb
import optuna
import polars as pl
from scipy import stats

from quant_trade.config.logger import log
from quant_trade.feature.process import CrossSection
from quant_trade.provider.transform import DateLike, to_date


@dataclass
class PurgedTimeSplit:
    """
    Time series split with purging to prevent lookahead bias.

    Purging removes training samples whose label period overlaps with
    test period to prevent information leakage.

    Diagram:
        [TRAIN]---[PURGE]---[TEST]---[FUTURE]
                  ↑        ↑
                embargo    horizon
    """

    horizon_days: int = 21
    embargo_days: int = 5

    def __post_init__(self):
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {self.horizon_days}")
        if self.embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {self.embargo_days}")

    def split(
        self,
        df: pl.DataFrame,
        split_date: DateLike,
        date_col: str | pl.Expr = "date",
        label_end_col: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data at a specific date with purging.

        Args:
            df: DataFrame with time series data
            date_col: Column containing observation dates
            split_date: Date to split train/test
            label_end_col: Optional column with label end date for purging

        Returns:
            Tuple of (train_df, test_df)
        """
        # Convert inputs
        if isinstance(date_col, str):
            date_expr = pl.col(date_col)
        else:
            date_expr = date_col

        split_date = to_date(split_date)
        purge_start = split_date - timedelta(days=self.horizon_days)
        test_start = split_date + timedelta(days=self.embargo_days)
        log.debug(f"purge start: {purge_start}")
        log.debug(f"test start: {test_start}")
        test_df = df.filter(date_expr >= test_start)

        if label_end_col:
            train_df = df.filter(
                (pl.col(label_end_col) < purge_start)  # Label ends before purge
                & (date_expr < purge_start)  # Observation before purge
            )
        else:
            train_df = df.filter(date_expr < purge_start)

        if len(train_df) == 0:
            raise ValueError(f"No training data before {purge_start}")
        if len(test_df) == 0:
            raise ValueError(f"No test data after {test_start}")

        log.info(f"Purging: Removed(embargo) data from {purge_start} to {split_date}")

        return train_df, test_df


@dataclass
class PurgedKFold:
    """
    K-Fold cross-validation with purging for time series data.

    Each fold has:
    1. Training period (all data before test start minus purge window)
    2. Purge window (horizon + embargo before test start)
    3. Test period

    Prevents information leakage from future to past.
    """

    n_splits: int = 5
    horizon_days: int = 21
    embargo_days: int = 5
    min_train_size: int = 0

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.horizon_days < 0:
            raise ValueError(f"horizon_days must be >= 0, got {self.horizon_days}")
        if self.embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {self.embargo_days}")

    def split(
        self,
        df: pl.DataFrame,
        date_col: str | pl.Expr = "date",
        label_end_col: str | None = None,
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate train/test splits with purging.

        Args:
            df: DataFrame with time series data
            date_col: Column containing observation dates
            label_end_col: Optional column with label end date for purging

        Yields:
            Tuples of (train_df, test_df) for each fold
        """
        if isinstance(date_col, str):
            date_expr = pl.col(date_col)
        else:
            date_expr = date_col

        dates = df.select(date_expr).unique().sort(date_expr).to_series().to_list()
        n_dates = len(dates)

        if n_dates < self.n_splits * 2:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for {self.n_splits} splits"
            )

        test_size = n_dates // (self.n_splits + 1)
        log.debug(f"Purged k fold test size {test_size}")
        for fold in range(1, self.n_splits + 1):
            start_idx = fold * test_size
            end_idx = (
                n_dates - 1 if fold == self.n_splits else start_idx + test_size - 1
            )
            test_start = dates[start_idx]
            log.debug(f"Purged k fold test start {fold}: {test_start}")
            test_end = dates[end_idx]
            log.debug(f"Purged k fold test end {fold}: {test_end}")
            purge_cutoff = test_start - timedelta(
                days=self.horizon_days + self.embargo_days
            )

            log.debug(
                f"Fold {fold}: test from {test_start} to {test_end}, purge cutoff {purge_cutoff}"
            )
            test_df = df.filter((date_expr >= test_start) & (date_expr <= test_end))

            if label_end_col is not None:
                train_df = df.filter(pl.col(label_end_col) < purge_cutoff)
            else:
                train_df = df.filter(date_expr < purge_cutoff)

            if len(train_df) < self.min_train_size:
                log.warning(
                    f"Fold {fold + 1}: only {len(train_df)} training samples, skipped"
                )
                continue

            if len(test_df) == 0:
                log.warning(f"Fold {fold + 1}: empty test set, skipped")
                continue

            yield train_df, test_df


@dataclass
class WalkForwardValidation:
    """
    Walk-forward validation with expanding/rolling window and purging.

    Common in financial time series to simulate live trading.
    """

    n_windows: int = 10
    window_size: int = 252  # ~1 trading year
    step_size: int = 63  # ~1 trading quarter
    horizon_days: int = 21
    embargo_days: int = 5
    min_train_size: int = 0
    expanding: bool = True

    def __post_init__(self):
        if self.n_windows < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.window_size}")
        if total_gap := self.horizon_days + self.embargo_days >= self.window_size:
            raise ValueError(
                f"horizon_days + embargo_days ({total_gap}), must be less than"
                f"window size ({self.window_size})"
            )
        if total_gap >= self.step_size:
            raise ValueError(
                f"horizon_days + embargo_days ({total_gap}), must be less than"
                f"step size ({self.step_size})"
            )

    def split(
        self,
        df: pl.DataFrame,
        date_col: str | pl.Expr = "date",
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate expanding/rolling walk-forward splits.
        """
        if isinstance(date_col, str):
            date_expr = pl.col(date_col)
        else:
            date_expr = date_col

        dates = (
            df.select(date_expr)
            .unique()
            .sort(date_expr)
            .get_column(date_expr.meta.output_name())
            .to_list()
        )
        n_dates = len(dates)

        for i in range(1, self.n_windows + 1):
            test_start_idx = i * self.step_size
            test_end_idx = min(test_start_idx + self.window_size, n_dates - 1)

            if test_end_idx <= test_start_idx:
                break

            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            purge_cutoff = test_start - timedelta(
                days=self.horizon_days + self.embargo_days
            )
            log.debug(
                f"Window {i}: test from {test_start} to {test_end}, purge cutoff {purge_cutoff}"
            )
            if self.expanding:
                # Expanding window: all data before purge cutoff
                train_df = df.filter(date_expr < purge_cutoff)
                log.debug(f"Window {i} expanding: train < {purge_cutoff}")
            else:
                # Rolling window: fixed size before purge cutoff
                train_start_idx = max(0, test_start_idx - self.step_size)
                train_start = dates[train_start_idx]
                train_df = df.filter(
                    (date_expr >= train_start) & (date_expr < purge_cutoff)
                )
                log.debug(
                    f"Window {i} rolling: train from {train_start} to {purge_cutoff}"
                )

            test_df = df.filter((date_expr >= test_start) & (date_expr <= test_end))

            if len(train_df) < self.min_train_size:
                log.warning(
                    f"Window {i}: only {len(train_df)} training samples, skipped"
                )
                continue
            yield train_df, test_df


class LabelBuilder(Protocol):
    def label(self, df: pl.DataFrame) -> pl.DataFrame: ...

    @property
    def label_name(self) -> str: ...
    @property
    def rank_over_name(self) -> str: ...


@dataclass
class GaussianLabelBuilder:
    """
    Builds Gaussian-transformed labels from returns for ranking models.

    This applies rank-Gaussian transformation (inverse normal transformation)
    which preserves ordering while creating normally-distributed targets.

    Formula:
    1. winsorize returns within groups
    2. rank winsorized returns: rank = r.rank().over("date")
    3. uniform transform: u = (rank - 0.5) / n
    4. Gaussian transform: y = Φ⁻¹(u) where Φ is standard normal CDF
    """

    def __init__(
        self,
        factor: str,
        rank_over: str = "date",
        group_by: list | str | None = None,
        limits: tuple[float, float] = (0.01, 0.99),
        alpha: float = 0.5,
    ):
        """
        Args:
            return_column: Column containing forward returns to transform
            date_column: Date column for cross-sectional operations
            group_columns: Additional grouping columns for winsorization
                          (e.g., ["sw_l1_code"] for industry groups)
            winsorize_quantiles: Quantiles for winsorization (lower, upper)
        """
        self.factor = factor
        self.rank_over = rank_over
        self.alpha = alpha

        # Handle group columns
        if group_by is None:
            self.group_columns = [rank_over]
        elif isinstance(group_by, str):
            self.group_columns = [rank_over, group_by]
        else:
            self.group_columns = [rank_over] + list(group_by)

        self.win_lower, self.win_upper = limits

        # Column names for intermediate steps
        self.win_col = f"{factor}_win"
        self.count_col = f"{self.win_col}_n"
        self.rank_col = f"{factor}_rank"
        self.uniform_col = f"{factor}_uniform"
        self.label_col = f"label_{factor}"

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Gaussian transformation to create labels.

        Returns:
            DataFrame with added label column containing Gaussian-transformed values
        """
        df_clean = df.filter(pl.col(self.factor).is_not_null())
        df_winsorized = df_clean.with_columns(
            CrossSection.winsorize(
                col=self.factor,
                by=self.rank_over,
                limits=(self.win_lower, self.win_upper),
                name=self.win_col,
            )
        )
        df_with_count = df_winsorized.with_columns(
            pl.col(self.win_col).count().over(self.rank_over).alias(self.count_col)
        )
        df_ranked = df_with_count.with_columns(
            CrossSection.rank(
                self.win_col, by=self.rank_over, name=self.rank_col, ascending=True
            )
        )
        # Apply uniform transformation
        df_uniform = df_ranked.with_columns(
            (
                (pl.col(self.rank_col) - self.alpha)
                / (pl.col(self.count_col) + 1 - 2 * self.alpha)
            )
            .clip(1e-10, 1 - 1e-10)
            .alias(self.uniform_col)
        )
        result = df_uniform.with_columns(
            pl.col(self.uniform_col)
            .map_elements(lambda u: stats.norm.ppf(u))
            .alias(self.label_col)
        )
        columns_to_drop = [
            self.win_col,
            self.rank_col,
            self.uniform_col,
            self.count_col,
        ]
        result = result.drop(
            [c for c in columns_to_drop if c in result.columns]
        ).filter(pl.col(self.label_col).is_not_null())

        return result

    @property
    def label_name(self) -> str:
        return self.label_col

    @property
    def rank_over_name(self) -> str:
        return self.rank_over


@dataclass
class LGBDataset:
    """
    Construct LightGBM Dataset objects with correct query groups.

    The "group" parameter is the single most important thing for
    LambdaRank to work correctly.  It tells LightGBM which rows
    belong to the same ranking task (= same date).

    Requirements:
        - df MUST be sorted by date before calling this.
        - All feature columns must already be normalised (_z suffix).
        - Label column must be the percentile rank.
    """

    def __init__(self, features: list[str], label_cls: LabelBuilder) -> None:
        self.label_builder = label_cls
        self.features = features

    def build(
        self, df: pl.DataFrame, ref: lgb.Dataset | None = None
    ) -> tuple[lgb.Dataset, list[str]]:
        df = self.label_builder.label(df)
        avail_feats = [f for f in self.features if f in df.columns]
        X = df.select(avail_feats).drop_nulls().to_numpy()
        y = df.select(self.label_builder.label_name).drop_nulls().to_series().to_numpy()
        group_name = self.label_builder.rank_over_name

        groups = df.group_by(group_name).len().sort(group_name)["len"].to_numpy()
        dataset = lgb.Dataset(
            X,
            y,
            group=groups,
            feature_name=avail_feats,
            reference=ref,
            free_raw_data=False,
        )

        return dataset, avail_feats


@dataclass
class LGBModelResult:
    """Minimal container for model results."""

    model: lgb.Booster
    feature_names: list[str]
    metric_val: float
    params: dict[str, Any]
    importance: dict[str, float]


@dataclass(frozen=True)
class LGBRankConfig:
    objective: str = "regression"
    metric: str = "rmse"
    ndcg_eval_at: tuple[int, ...] = (20,)
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    log_period: int = 100
    seed: int = 42

    default_params: dict[str, Any] = field(
        default_factory=lambda: {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "min_child_samples": 50,
        }
    )


@dataclass
class LGBTrainer:
    def __init__(
        self,
        dataset: "LGBDataset",
        config: LGBRankConfig,
    ) -> None:
        self.dataset = dataset
        self.config = config

    def _objective(
        self,
        trial: optuna.Trial,
        train_ds: lgb.Dataset,
        val_ds: lgb.Dataset,
    ) -> float:
        cfg = self.config

        params = {
            "objective": cfg.objective,
            "metric": cfg.metric,
            "ndcg_eval_at": list(cfg.ndcg_eval_at),
            "verbosity": -1,
            "seed": cfg.seed,
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        }

        model = lgb.train(
            params,
            train_set=train_ds,
            valid_sets=[val_ds],
            num_boost_round=cfg.num_boost_round,
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds),
            ],
        )

        return model.best_score["valid_0"][self.config.metric]

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        optimize: bool = True,
        n_trials: int = 50,
    ) -> "LGBModelResult":
        train_ds, features = self.dataset.build(train_df)
        val_ds, _ = self.dataset.build(val_df)

        cfg = self.config

        if optimize:
            sampler = optuna.samplers.TPESampler(seed=cfg.seed)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
            )
            study.optimize(
                lambda t: self._objective(t, train_ds, val_ds),
                n_trials=n_trials,
            )

            params = {**cfg.default_params, **study.best_params}
        else:
            params = dict(cfg.default_params)

        params.update(
            {
                "objective": cfg.objective,
                "metric": cfg.metric,
                "ndcg_eval_at": list(cfg.ndcg_eval_at),
                "verbosity": -1,
                "seed": cfg.seed,
            }
        )

        model = lgb.train(
            params,
            train_ds,
            valid_sets=[val_ds],
            num_boost_round=cfg.num_boost_round,
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds),
                lgb.log_evaluation(cfg.log_period),
            ],
        )

        metric_val = model.best_score["valid_0"][self.config.metric]
        importance = dict(zip(features, model.feature_importance()))
        return LGBModelResult(
            model=model,
            feature_names=features,
            metric_val=metric_val,
            params=params,
            importance=importance,
        )

    def train_batchwise(
        self,
        batch_iter: Generator[tuple[pl.DataFrame, pl.DataFrame]],
        optimize: bool = False,
    ) -> LGBModelResult:
        """
        Train a **single model** using batches for memory efficiency,
        while optionally performing purged CV for validation.
        """
        # Accumulate all training batches
        train_batches = []
        val_batches = []

        for train_df, val_df in batch_iter:
            train_batches.append(train_df)
            val_batches.append(val_df)

        # Concatenate all batches
        full_train = pl.concat(train_batches)
        full_val = pl.concat(val_batches)

        log.info(f"Total accumulated: Train={len(full_train):,}, Val={len(full_val):,}")

        # Train single model
        result = self.train(train_df=full_train, val_df=full_val, optimize=optimize)
        return result
