# LightGBM Flexible Objective/Metric Configuration Plan

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MetricConfig Factory                          │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  MetricConfig.ranking(label_builder, metric="ndcg@10")    │   │
│  │  MetricConfig.regression(label_builder, metric="rmse")    │   │
│  │  MetricConfig.binary(label_builder, metric="auc")        │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                    │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│         ┌─────────┐    ┌──────────┐    ┌─────────┐             │
│         │ Ranking │    │Regression│    │ Binary  │             │
│         └─────────┘    └──────────┘    └─────────┘             │
│              │               │               │                  │
│              └───────────────┼───────────────┘                  │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Processor[Target: LabelBuilder]             │   │
│  │         Takes MetricConfig - knows objective/metric      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Trainer                              │   │
│  │  trainer = Trainer(processor)                            │   │
│  │  result = trainer.train(common_config)                   │   │
│  │  (metric_config comes from processor)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Python 3.12+ Generic Design

### 1. LabelBuilder Protocol
```python
# src/quant_trade/model/process.py

class LabelBuilder(Protocol):
    """Protocol for label builders."""
    
    def label(self, df: pl.DataFrame) -> pl.DataFrame: ...
    
    @property
    def label_name(self) -> str: ...
    
    @property
    def rank_by_name(self) -> str: ...


@dataclass
class GaussianLabelBuilder:
    """For ranking."""
    factor: str
    rank_by: str = "date"
    winsor_limits: tuple[float, float] | None = (0.01, 0.99)


@dataclass  
class IdentityLabelBuilder:
    """For regression."""
    factor: str
    rank_by: str = "date"


@dataclass
class BinaryLabelBuilder:
    """For binary classification."""
    factor: str
    threshold: float = 0.0
    rank_by: str = "date"
```

### 2. CommonConfig
```python
# src/quant_trade/model/lgb.py

from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import optuna
import polars as pl

from quant_trade.config.logger import log
from quant_trade.model.process import LabelBuilder, GaussianLabelBuilder


@dataclass(frozen=True)
class CommonConfig:
    """
    Common training config - objective/metric agnostic.
    
    Contains: num_boost_round, seed, early_stopping, log_period, default_params.
    Passed to trainer.train(common_config).
    """
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
```

### 3. MetricConfig with Static Factory Methods
```python
# src/quant_trade/model/lgb.py

@dataclass(frozen=True)
class MetricConfig[Target: LabelBuilder]:
    """
    Metric-specific config - objective, metric, label_builder, ndcg_eval_at.
    
    Factory methods are static - pure functions.
    Processor takes MetricConfig for objective/metric knowledge.
    """
    Target: type[Target]
    objective: str
    metric: str
    ndcg_eval_at: tuple[int, ...] | None
    label_builder: Target
    
    @staticmethod
    def ranking(
        label_builder: GaussianLabelBuilder,
        *,
        metric: Literal["ndcg", "ndcg@10", "ndcg@20", "map", "map@10"] = "ndcg@10",
    ) -> "MetricConfig[GaussianLabelBuilder]":
        """Create ranking config."""
        if metric.startswith("ndcg@"):
            ndcg = (int(metric.split("@")[1]),)
        else:
            ndcg = (10,)
        
        return MetricConfig(
            Target=GaussianLabelBuilder,
            objective="lambdarank",
            metric=metric,
            ndcg_eval_at=ndcg,
            label_builder=label_builder,
        )
    
    @staticmethod
    def regression(
        label_builder: IdentityLabelBuilder,
        *,
        metric: Literal["rmse", "mae", "mape"] = "rmse",
    ) -> "MetricConfig[IdentityLabelBuilder]":
        """Create regression config."""
        return MetricConfig(
            Target=IdentityLabelBuilder,
            objective="regression",
            metric=metric,
            ndcg_eval_at=None,
            label_builder=label_builder,
        )
    
    @staticmethod
    def binary(
        label_builder: BinaryLabelBuilder,
        *,
        metric: Literal["binary_logloss", "binary_error", "auc"] = "binary_logloss",
    ) -> "MetricConfig[BinaryLabelBuilder]":
        """Create binary config."""
        return MetricConfig(
            Target=BinaryLabelBuilder,
            objective="binary",
            metric=metric,
            ndcg_eval_at=None,
            label_builder=label_builder,
        )
    
    def lgb_params(
        self,
        optimized_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build LGB params - metric-specific."""
        params = dict(optimized_params or {})
        params.update({
            "objective": self.objective,
            "metric": self.metric,
            "verbosity": -1,
        })
        
        if self.ndcg_eval_at is not None:
            params["ndcg_eval_at"] = list(self.ndcg_eval_at)
        
        return params
    
    def optimization_direction(self) -> str:
        """Get direction for Optuna."""
        if "ndcg" in self.metric or "auc" in self.metric:
            return "maximize"
        return "minimize"
```

### 4. Processor Takes MetricConfig
```python
# src/quant_trade/model/lgb.py

class Processor[Target: LabelBuilder]:
    """
    Processor takes MetricConfig - knows objective/metric.
    
    MetricConfig is stored here, accessible to Trainer.
    """
    
    def __init__(self, features: list[str], config: MetricConfig[Target]) -> None:
        self.features = features
        self.config = config
    
    def build(
        self,
        df: pl.DataFrame,
        *,
        ref: lgb.Dataset | None = None,
    ) -> tuple[lgb.Dataset, list[str]]:
        """Build lgb.Dataset using config's label_builder."""
        df = self.config.label_builder.label(df)
        # ... existing implementation
        return dataset, avail_feats
```

### 5. Trainer Instantiated by Processor
```python
# src/quant_trade/model/lgb.py

@dataclass
class Trainer:
    """
    Trainer instantiated by Processor.
    
    MetricConfig comes from processor.config.
    train() only takes CommonConfig.
    
    Usage:
        processor = Processor(features, metric_config)
        trainer = Trainer(processor)
        result = trainer.train(common_config)
    """
    
    processor: Processor[LabelBuilder]
    optimize: bool = True
    n_trials: int = 50
    
    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        common_config: CommonConfig,
    ) -> ModelResult:
        """
        Train - metric_config comes from processor.
        
        Only CommonConfig needed - metric_config already in processor.
        """
        config = self.processor.config
        common = common_config
        
        # Build datasets
        train_ds, features = self.processor.build(train_df)
        val_ds, _ = self.processor.build(val_df)
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective."""
            opt_params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            }
            
            lgb_params = config.lgb_params(optimized_params=opt_params)
            lgb_params.update({"seed": common.seed})
            
            model = lgb.train(
                lgb_params,
                train_set=train_ds,
                valid_sets=[val_ds],
                num_boost_round=common.num_boost_round,
                callbacks=[lgb.early_stopping(common.early_stopping_rounds)],
            )
            
            return model.best_score["valid_0"][config.metric]
        
        # Optimization
        if self.optimize:
            sampler = optuna.samplers.TPESampler(seed=common.seed)
            study = optuna.create_study(
                direction=config.optimization_direction(),
                sampler=sampler,
            )
            study.optimize(objective, n_trials=self.n_trials)
            optimized_params = {**common.default_params, **study.best_params}
        else:
            optimized_params = dict(common.default_params)
        
        # Final training
        lgb_params = config.lgb_params(optimized_params=optimized_params)
        lgb_params.update({"seed": common.seed})
        
        model = lgb.train(
            lgb_params,
            train_ds,
            valid_sets=[val_ds],
            num_boost_round=common.num_boost_round,
            callbacks=[
                lgb.early_stopping(common.early_stopping_rounds),
                lgb.log_evaluation(common.log_period),
            ],
        )
        
        metric_val = model.best_score["valid_0"][config.metric]
        importance = dict(zip(features, model.feature_importance()))
        
        return ModelResult(
            model=model,
            feature_names=features,
            metric_val=metric_val,
            params=lgb_params,
            importance=importance,
        )
```

---

## Usage

### Example 1: Ranking
```python
from quant_trade.model.process import GaussianLabelBuilder
from quant_trade.model.lgb import MetricConfig, CommonConfig, Processor, Trainer

# MetricConfig factory - static method
metric_config = MetricConfig.ranking(
    label_builder=GaussianLabelBuilder(factor="return_1m"),
    metric="ndcg@10",
)

# Processor takes MetricConfig
processor = Processor(features=features, config=metric_config)

# Trainer instantiated by processor
trainer = Trainer(processor=processor)

# train() only takes CommonConfig
result = trainer.train(
    train_df=train_df,
    val_df=val_df,
    common_config=CommonConfig(
        num_boost_round=500,
        seed=42,
    )
)
```

### Example 2: Regression
```python
from quant_trade.model.process import IdentityLabelBuilder
from quant_trade.model.lgb import MetricConfig, CommonConfig, Processor, Trainer

metric_config = MetricConfig.regression(
    label_builder=IdentityLabelBuilder(factor="return_1m"),
    metric="mae",
)

processor = Processor(features, config=metric_config)
trainer = Trainer(processor)

result = trainer.train(
    train_df=train_df,
    val_df=val_df,
    common_config=CommonConfig(num_boost_round=1000),
)
```

### Example 3: Binary
```python
from quant_trade.model.process import BinaryLabelBuilder
from quant_trade.model.lgb import MetricConfig, CommonConfig, Processor, Trainer

metric_config = MetricConfig.binary(
    label_builder=BinaryLabelBuilder(factor="signal", threshold=0.5),
    metric="auc",
)

processor = Processor(features, config=metric_config)
trainer = Trainer(processor)

result = trainer.train(
    train_df=train_df,
    val_df=val_df,
    common_config=CommonConfig(),
)
```

---

## Design Summary

| Component | Role |
|-----------|------|
| **MetricConfig** | Static factories (`ranking()`, `regression()`, `binary()`), objective, metric, ndcg_eval_at, label_builder |
| **CommonConfig** | num_boost_round, seed, early_stopping, log_period, default_params |
| **Processor** | Takes MetricConfig, stores config, used by Trainer |
| **Trainer** | Instantiated by Processor, train(common_config) |

**Flow:**
```
MetricConfig.ranking(...) → Processor(features, config) → Trainer(processor) → train(common_config)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/quant_trade/model/process.py` | Protocol + label builders |
| `src/quant_trade/model/lgb.py` | MetricConfig (staticmethod), CommonConfig, Processor, Trainer (instantiated by Processor) |

---

## Benefits

1. **Simple**: No _TrainerRunner, Trainer instantiated by Processor
2. **Separation**: MetricConfig ≠ CommonConfig
3. **Curried Feel**: `trainer.train(common_config)` - metric comes from processor
4. **Type Safe**: Static factories restrict metric choices
5. **Clean**: Each component has single responsibility
