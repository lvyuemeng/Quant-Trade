# AI-Driven Stock Selection System for China A-Shares

A library-level architecture for a production-grade boosting-based stock selection engine:

---

## **System Architecture Overview**

```
Environment Setup → Data Acquisition → Feature Engineering → 
Model Development → Portfolio Construction → Backtesting → 
Production Deployment → Monitoring & Iteration
```

---

## **Stage 1: Environment & Infrastructure**

### **1.1 Technology Stack**
```python
# Online book host
- marimo
- pyzmq

# Data processing
- polars
- narwhals
- numpy

# Plot
- plotly
- seaborn

# Data acquisition
- akshare

# Machine Learning
- scipy
- optuna  # Hyperparameter optimization
- shap   # Model interpretation
## Boost algorithm(until needed)
- lightgbm
- xgboost

# Backtesting & Portfolio
- backtrader==1.9.78
- empyrical==0.5.5
- cvxpy==1.4.0   # Portfolio optimization

# Data Storage
- clickhouse-driver==0.2.6  # Time-series DB
- redis==5.0.0              # Feature cache
- sqlite3 (built-in)        # Metadata storage
```

### **1.2 Directory Structure**

Below is a example rather strict routine.

```
project_root/
├── config/
│   ├── model.yaml
│   ├── data.yaml
│   └── backtest.yaml
├── data/
│   ├── raw/           # Original downloads
│   ├── interim/       # Partially processed
│   ├── processed/     # Final features
│   └── cache/         # Redis snapshots
├── src/
│   ├── data/
│   │   ├── acquisition.py
│   │   ├── cleaning.py
│   │   └── feature_store.py
│   ├── features/
│   │   ├── fundamental.py
│   │   ├── technical.py
│   │   ├── macro.py
│   │   └── normalization.py
│   ├── models/
│   │   ├── lgb_ranker.py
│   │   ├── ensemble.py
│   │   └── hyperopt.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── portfolio.py
│   │   └── metrics.py
│   └── utils/
│       ├── logger.py
│       └── validators.py
├── notebooks/         # Research & EDA
├── models/           # Saved model artifacts
├── results/          # Backtest outputs
└── tests/            # Unit tests
```

---

## **Stage 2: Data Acquisition Pipeline**

### **2.1 Data Sources (China A-Share Specific)**

```python
class DataAcquisition:
    """
    Priority data vendors for China A-shares:
    - AkShare (free alternative)
    - Tushare Pro (recommended for retail/academic)
    - Wind (institutional standard)
    - JoinQuant (quant-focused)
    """
    
    def __init__(self, provider='akshare'):
        self.provider = provider
		# cache...
        self.cache = RedisCache()
        
    def fetch_stock_universe(self, date):
        """
        Get tradeable A-share universe
        
        Filters:
        - Remove ST/\*ST stocks
        - Remove suspended stocks (停牌)
        - Remove stocks < 60 days from IPO
        - Only CSI All-Share constituents (optional)
        """
        universe = self._query_tradeable_stocks(date)
        
        # Critical: Check tradeability
        universe = universe[
            (~universe['is_suspended']) &
            (universe['days_since_ipo'] >= 60) &
            (~universe['name'].str.contains('ST'))
        ]
        
        return universe
    
    def fetch_fundamental_data(self, start_date, end_date):
        """
        Financial statement data with point-in-time alignment
        
        CRITICAL: Use announcement_date, NOT report_date
        """
        # Income statement
        income = self._get_income_statement(
            start_date, 
            end_date,
            date_type='announcement'  # Avoid look-ahead bias
        )
        
        # Balance sheet
        balance = self._get_balance_sheet(
            start_date,
            end_date,
            date_type='announcement'
        )
        
        # Cash flow
        cashflow = self._get_cashflow_statement(
            start_date,
            end_date,
            date_type='announcement'
        )
        
        return self._merge_financial_data(income, balance, cashflow)
    
    def fetch_market_data(self, start_date, end_date):
        """
        Daily OHLCV + adjustment factors
        """
        data = self._get_daily_prices(start_date, end_date)
        
        # Apply forward adjustment for stock splits/dividends
        data = self._apply_adj_factor(data)
        
        return data
    
    def fetch_macro_indicators(self, start_date, end_date):
        """
        Market-wide context features
        """
        macro = {
            'northbound_flow': self._get_northbound_flow(start_date, end_date),
            'margin_balance': self._get_margin_trading(start_date, end_date),
            'shibor': self._get_shibor_rates(start_date, end_date),
            'csi300_pe': self._get_index_valuation('000300', start_date, end_date),
            'market_turnover': self._get_market_turnover(start_date, end_date)
        }
        
        return pd.DataFrame(macro)
```

### **2.2 Data Storage Schema**

`config/config.yaml`:

```yml
# Database Configuration
database:
  clickhouse:
    host: "localhost"
    port: 9000
    database: "ashare_quant"
    user: "default"
    password: ""
  
  redis:
    host: "localhost"
    port: 6379
    db: 0
    decode_responses: true

# Data Sources
data_sources:
  primary: "tushare"  # tushare, akshare, wind
  tushare:
    token: "${TUSHARE_TOKEN}"  # Set in .env file
  
# Market Parameters
market:
  universe: "all_ashare"  # all_ashare, csi300, csi500
  min_days_since_ipo: 60
  exclude_st: true
  exclude_suspended: true

# Feature Engineering
features:
  fundamental:
    - "roe"
    - "roa"
    - "debt_to_asset"
    - "gross_margin"
    - "revenue_growth_yoy"
    - "profit_growth_yoy"
    - "cf_to_profit"
  
  valuation:
    - "pe_ttm"
    - "pb"
    - "ps"
    - "dividend_yield"
    - "earnings_yield"
  
  behavioral:
    - "turnover_rate"
    - "turnover_ratio"
    - "excess_return_20d"
    - "excess_return_60d"
    - "excess_return_120d"
    - "volatility_20d"
    - "abnormal_volume"
  
  macro:
    - "northbound_flow"
    - "shibor_3m"
    - "csi300_pe"
    - "market_turnover"
    - "margin_balance"

# Normalization
normalization:
  method: "cross_sectional"  # cross_sectional, time_series
  industry_level: "sw_l1"  # sw_l1, sw_l2, citic_l1
  winsorize_limits: [0.025, 0.975]
  handle_missing: "median"  # median, forward_fill, drop

# Model Configuration
model:
  horizons: [20, 60, 252]  # Trading days: ~1m, ~3m, ~1y
  
  lgb_params:
    objective: "lambdarank"
    metric: "ndcg"
    ndcg_eval_at: [5, 10, 20]
    boosting_type: "gbdt"
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    min_child_samples: 20
    lambda_l1: 0.1
    lambda_l2: 0.1
    verbose: -1
    seed: 42
  
  training:
    num_boost_round: 1000
    early_stopping_rounds: 50
    train_test_split: 0.8  # Time-based split
  
  ensemble:
    weights:
      "20d": 0.3
      "60d": 0.4
      "252d": 0.3

# Backtesting
backtest:
  initial_capital: 10000000  # 10M CNY
  commission: 0.0003  # 0.03% each way
  slippage: 0.0005  # 0.05%
  rebalance_freq: "M"  # M (monthly), W (weekly)
  top_k: 50
  max_weight_per_stock: 0.05  # 5%
  weight_method: "equal"  # equal, risk_parity, factor_weighted
  benchmark: "000300.SH"  # CSI 300

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
  rotation: "100 MB"
  retention: "30 days"
  path: "logs/ashare_quant_{time}.log"
```

```sql
-- ClickHouse schema for time-series data

CREATE TABLE stock_daily (
    date Date,
    ts_code String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    turnover_rate Float64,  -- 換手率
    adj_factor Float64,
    is_limit_up UInt8,      -- 漲停
    is_limit_down UInt8,
    is_suspended UInt8
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, ts_code);

CREATE TABLE fundamentals (
    announcement_date Date,  -- NOT report_date
    ts_code String,
    roe Float64,
    net_profit_growth Float64,
    debt_to_asset Float64,
    operating_cf Float64,
    -- ... other fields
    PRIMARY KEY (announcement_date, ts_code)
) ENGINE = MergeTree()
ORDER BY (announcement_date, ts_code);

CREATE TABLE industry_classification (
    date Date,
    ts_code String,
    sw_l1 String,  -- Shenwan Level 1
    sw_l2 String,  -- Shenwan Level 2
    PRIMARY KEY (date, ts_code)
) ENGINE = ReplacingMergeTree()
ORDER BY (date, ts_code);
```

---

## **Stage 3: Feature Engineering Pipeline**

### **3.1 Normalization Engine (Critical Component)**

```python
class CrossSectionalNormalizer:
    """
    Implements point-in-time, industry-neutral Z-score normalization
    
    This is THE most critical component to avoid look-ahead bias
    """
    
    def __init__(self, industry_column='sw_l1'):
        self.industry_column = industry_column
        
    def normalize(self, df, feature_cols, date_column='date'):
        """
        Apply cross-sectional Z-score within industries
        
        Parameters:
        -----------
        df : DataFrame with columns [date, ts_code, industry, features...]
        feature_cols : List of column names to normalize
        
        Returns:
        --------
        DataFrame with normalized features (suffix '_z')
        """
        normalized = []
        
        for date, group in df.groupby(date_column):
            # Step 1: Winsorize per industry per date
            for industry, ind_group in group.groupby(self.industry_column):
                ind_group = self._winsorize(
                    ind_group, 
                    feature_cols,
                    limits=(0.025, 0.975)  # 2.5th to 97.5th percentile
                )
                
                # Step 2: Calculate industry-specific mean/std
                for col in feature_cols:
                    mean = ind_group[col].mean()
                    std = ind_group[col].std()
                    
                    # Handle edge case: zero std
                    if std < 1e-8:
                        ind_group[f'{col}_z'] = 0.0
                    else:
                        ind_group[f'{col}_z'] = (ind_group[col] - mean) / std
                
                normalized.append(ind_group)
        
        return pd.concat(normalized, ignore_index=True)
    
    def _winsorize(self, df, cols, limits=(0.01, 0.99)):
        """
        Cap extreme values to percentile boundaries
        
        Why: A P/E of 5000 will destroy the mean/std calculation
        """
        from scipy.stats.mstats import winsorize
        
        df_copy = df.copy()
        for col in cols:
            if df_copy[col].notna().sum() > 10:  # Need minimum samples
                df_copy[col] = winsorize(
                    df_copy[col].fillna(df_copy[col].median()),
                    limits=limits
                )
        
        return df_copy
```

### **3.2 Feature Library**

```python
class FeatureEngineer:
    """
    Generate all features following the categorization from the document
    """
    
    def __init__(self, data_engine):
        self.data = data_engine
        self.normalizer = CrossSectionalNormalizer()
        
    def build_fundamental_features(self, df):
        """
        Financial statement-based features (1-year anchor)
        """
        features = pd.DataFrame(index=df.index)
        
        # Quality metrics
        features['roe'] = df['net_profit'] / df['total_equity']
        features['roa'] = df['net_profit'] / df['total_assets']
        features['gross_margin'] = df['gross_profit'] / df['revenue']
        features['debt_to_asset'] = df['total_debt'] / df['total_assets']
        
        # Growth metrics (YoY)
        features['revenue_growth_yoy'] = df.groupby('ts_code')['revenue'].pct_change(4)
        features['profit_growth_yoy'] = df.groupby('ts_code')['net_profit'].pct_change(4)
        
        # Cash flow quality
        features['cf_to_profit'] = df['operating_cf'] / (df['net_profit'] + 1e-6)
        
        return features
    
    def build_valuation_features(self, df):
        """
        Valuation ratios (selection filter)
        """
        features = pd.DataFrame(index=df.index)
        
        # Price multiples
        features['pe_ttm'] = df['market_cap'] / df['net_profit_ttm']
        features['pb'] = df['market_cap'] / df['total_equity']
        features['ps'] = df['market_cap'] / df['revenue_ttm']
        features['dividend_yield'] = df['dividend'] / df['close']
        
        # Earnings yield (inverse P/E)
        features['earnings_yield'] = 1 / (features['pe_ttm'] + 1e-6)
        
        return features
    
    def build_behavioral_features(self, df):
        """
        Volume/momentum features (1m-3m trigger)
        """
        features = pd.DataFrame(index=df.index)
        
        # Turnover analysis (critical for A-shares)
        features['turnover_rate'] = df['turnover_rate']
        features['turnover_ma20'] = df.groupby('ts_code')['turnover_rate'].transform(
            lambda x: x.rolling(20).mean()
        )
        features['turnover_ratio'] = features['turnover_rate'] / (features['turnover_ma20'] + 1e-6)
        
        # Momentum (relative to CSI 300)
        csi300_return = self._get_benchmark_return(df, '000300')
        
        for period in [20, 60, 120]:
            stock_return = df.groupby('ts_code')['close'].pct_change(period)
            features[f'excess_return_{period}d'] = stock_return - csi300_return
        
        # Volatility
        features['volatility_20d'] = df.groupby('ts_code')['close'].pct_change().rolling(20).std()
        
        # Volume abnormality
        features['volume_ma20'] = df.groupby('ts_code')['volume'].transform(
            lambda x: x.rolling(20).mean()
        )
        features['abnormal_volume'] = df['volume'] / (features['volume_ma20'] + 1e-6)
        
        return features
    
    def build_macro_features(self, df, macro_df):
        """
        Market-wide context (same value for all stocks on given date)
        """
        # Merge macro data onto stock data by date
        df = df.merge(macro_df, on='date', how='left')
        
        features = pd.DataFrame(index=df.index)
        
        # Liquidity
        features['northbound_flow'] = df['northbound_flow']
        features['northbound_flow_ma5'] = df['northbound_flow'].rolling(5).mean()
        features['shibor_3m'] = df['shibor_3m']
        
        # Market state
        features['csi300_pe'] = df['csi300_pe']
        features['csi300_volatility'] = df['csi300_volatility']
        features['market_turnover'] = df['market_turnover']
        
        # Sentiment
        features['margin_balance'] = df['margin_balance']
        features['margin_balance_chg'] = df['margin_balance'].pct_change(5)
        
        return features
    
    def build_all_features(self, start_date, end_date):
        """
        Master pipeline: combines all feature types
        """
        # Load base data
        market_data = self.data.fetch_market_data(start_date, end_date)
        fundamental_data = self.data.fetch_fundamental_data(start_date, end_date)
        macro_data = self.data.fetch_macro_indicators(start_date, end_date)
        industry_data = self.data.fetch_industry_classification(start_date, end_date)
        
        # Merge all sources
        df = market_data.merge(fundamental_data, on=['date', 'ts_code'], how='left')
        df = df.merge(industry_data, on=['date', 'ts_code'], how='left')
        
        # Generate features
        fund_features = self.build_fundamental_features(df)
        val_features = self.build_valuation_features(df)
        behav_features = self.build_behavioral_features(df)
        macro_features = self.build_macro_features(df, macro_data)
        
        all_features = pd.concat([
            df[['date', 'ts_code', 'sw_l1', 'close']],
            fund_features,
            val_features,
            behav_features,
            macro_features
        ], axis=1)
        
        # CRITICAL: Cross-sectional normalization
        feature_cols = [col for col in all_features.columns 
                       if col not in ['date', 'ts_code', 'sw_l1', 'close']]
        
        normalized_features = self.normalizer.normalize(
            all_features,
            feature_cols,
            date_column='date'
        )
        
        return normalized_features
```

---

## **Stage 4: Label Engineering**

```python
class LabelGenerator:
    """
    Generate forward-looking returns as prediction targets
    
    CRITICAL: Labels must also be cross-sectionally ranked
    """
    
    def __init__(self, periods=[20, 60, 252]):  # 1m, 3m, 1y
        self.periods = periods
        
    def generate_returns(self, df):
        """
        Calculate forward returns for multiple horizons
        """
        labels = pd.DataFrame(index=df.index)
        
        for period in self.periods:
            # Forward return
            labels[f'return_{period}d'] = df.groupby('ts_code')['close'].pct_change(period).shift(-period)
            
            # Winsorize returns (handle extreme outliers)
            labels[f'return_{period}d'] = self._winsorize_returns(
                labels[f'return_{period}d'],
                limits=(0.01, 0.99)
            )
        
        return labels
    
    def generate_cross_sectional_ranks(self, df, return_cols):
        """
        Convert raw returns to percentile ranks within each date
        
        Why: This makes the model learn RELATIVE performance
        """
        ranked = df.copy()
        
        for col in return_cols:
            ranked[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
        
        return ranked
    
    def _winsorize_returns(self, returns, limits=(0.01, 0.99)):
        """
        Cap extreme returns to avoid overfitting to black swans
        """
        from scipy.stats.mstats import winsorize
        return winsorize(returns.dropna(), limits=limits)
```

---

## **Stage 5: Model Development**

### **5.1 Multi-Horizon Ranking Models**

```python
class MultiHorizonRanker:
    """
    Train separate LightGBM models for each time horizon
    Following the "Stacked Boosting" approach from the document
    """
    
    def __init__(self, horizons=[20, 60, 252]):
        self.horizons = horizons
        self.models = {}
        self.feature_importance = {}
        
    def train_single_horizon(self, X_train, y_train, horizon, params=None):
        """
        Train LightGBM ranker for one specific horizon
        """
        if params is None:
            params = {
                'objective': 'lambdarank',  # Ranking objective
                'metric': 'ndcg',
                'ndcg_eval_at': [5, 10, 20],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
        
        # Create query groups (one per date)
        # LightGBM needs to know which samples belong to the same ranking task
        train_data = X_train.copy()
        train_data['label'] = y_train
        
        query_groups = train_data.groupby('date').size().values
        
        # Prepare training data
        feature_cols = [col for col in X_train.columns if col.endswith('_z')]
        
        lgb_train = lgb.Dataset(
            train_data[feature_cols],
            label=train_data['label'],
            group=query_groups
        )
        
        # Train model
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.models[f'model_{horizon}d'] = model
        self.feature_importance[f'model_{horizon}d'] = dict(zip(
            feature_cols,
            model.feature_importance(importance_type='gain')
        ))
        
        return model
    
    def train_all_horizons(self, features_df, labels_df):
        """
        Train models for all time horizons
        """
        for horizon in self.horizons:
            print(f"\n{'='*50}")
            print(f"Training model for {horizon}-day horizon")
            print(f"{'='*50}")
            
            # Prepare data
            df = features_df.merge(
                labels_df[[f'return_{horizon}d_rank']],
                left_index=True,
                right_index=True
            )
            
            # Remove rows with missing labels
            df = df.dropna(subset=[f'return_{horizon}d_rank'])
            
            # Time-based split (critical: no random shuffle!)
            train_end_date = df['date'].quantile(0.8)
            
            train_df = df[df['date'] <= train_end_date]
            val_df = df[df['date'] > train_end_date]
            
            # Train model
            model = self.train_single_horizon(
                X_train=train_df,
                y_train=train_df[f'return_{horizon}d_rank'],
                horizon=horizon
            )
            
            # Validation performance
            val_score = self.evaluate(model, val_df, f'return_{horizon}d_rank')
            print(f"Validation NDCG@20: {val_score:.4f}")
    
    def predict(self, X, horizon):
        """
        Generate ranking scores for a specific horizon
        """
        model = self.models[f'model_{horizon}d']
        feature_cols = [col for col in X.columns if col.endswith('_z')]
        
        scores = model.predict(X[feature_cols])
        return scores
    
    def evaluate(self, model, df, label_col):
        """
        Calculate NDCG@20 on validation set
        """
        from sklearn.metrics import ndcg_score
        
        feature_cols = [col for col in df.columns if col.endswith('_z')]
        predictions = model.predict(df[feature_cols])
        
        # Group by date and calculate NDCG for each date
        ndcg_scores = []
        for date, group in df.groupby('date'):
            if len(group) > 20:  # Need enough stocks for meaningful ranking
                y_true = group[label_col].values.reshape(1, -1)
                y_pred = predictions[group.index].reshape(1, -1)
                
                ndcg = ndcg_score(y_true, y_pred, k=20)
                ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)
```

### **5.2 Meta-Ensemble Model**

```python
class MetaEnsemble:
    """
    Combine predictions from multiple horizon models
    Identifies "High Conviction" stocks that rank well across horizons
    """
    
    def __init__(self, base_models, weights=None):
        self.base_models = base_models
        self.weights = weights or {'20d': 0.3, '60d': 0.4, '252d': 0.3}
        
    def predict(self, X):
        """
        Weighted ensemble of base model predictions
        """
        ensemble_scores = np.zeros(len(X))
        
        for horizon, weight in self.weights.items():
            scores = self.base_models.predict(X, int(horizon.replace('d', '')))
            ensemble_scores += weight * self._normalize_scores(scores)
        
        return ensemble_scores
    
    def _normalize_scores(self, scores):
        """
        Normalize scores to [0, 1] range for comparable weighting
        """
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def select_top_k(self, X, k=50):
        """
        Select top K stocks based on ensemble scores
        """
        scores = self.predict(X)
        X['ensemble_score'] = scores
        
        return X.nlargest(k, 'ensemble_score')
```

---

## **Stage 6: Backtesting Engine**

```python
class AShareBacktester:
    """
    Realistic backtesting with A-share specific constraints
    """
    
    def __init__(self, initial_capital=10_000_000, commission=0.0003):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.03% each way
        self.positions = {}
        self.cash = initial_capital
        self.equity_curve = []
        
    def run_backtest(self, signals_df, price_df, 
                    rebalance_freq='M', top_k=50):
        """
        Execute backtest with monthly rebalancing
        
        Parameters:
        -----------
        signals_df : DataFrame with [date, ts_code, ensemble_score]
        price_df : DataFrame with [date, ts_code, close]
        rebalance_freq : 'M' (monthly) or 'W' (weekly)
        top_k : Number of stocks to hold
        """
        # Group by rebalance periods
        signals_df['rebalance_period'] = pd.to_datetime(signals_df['date']).dt.to_period(rebalance_freq)
        
        for period, period_data in signals_df.groupby('rebalance_period'):
            rebalance_date = period_data['date'].min()
            
            # Get current prices
            current_prices = price_df[price_df['date'] == rebalance_date]
            
            # Filter tradeable stocks (not limit-up, not suspended)
            tradeable = current_prices[
                (current_prices['is_limit_up'] == 0) &
                (current_prices['is_suspended'] == 0)
            ]['ts_code'].tolist()
            
            signals_tradeable = period_data[period_data['ts_code'].isin(tradeable)]
            
            # Select top K stocks
            selected = signals_tradeable.nlargest(top_k, 'ensemble_score')
            
            # Rebalance portfolio
            self._rebalance(
                selected['ts_code'].tolist(),
                current_prices,
                weight_method='equal'  # or 'risk_parity'
            )
            
            # Track equity
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.equity_curve.append({
                'date': rebalance_date,
                'value': portfolio_value,
                'positions': len(self.positions)
            })
    
    def _rebalance(self, target_stocks, prices, weight_method='equal'):
        """
        Rebalance portfolio to target stocks
        
        Handles:
        - Transaction costs
        - Position sizing
        - Max weight constraints (5% per stock for A-shares)
        """
        current_stocks = set(self.positions.keys())
        target_stocks = set(target_stocks)
        
        # Sell positions not in target
        for stock in current_stocks - target_stocks:
            self._sell(stock, prices)
        
        # Calculate target weights
        if weight_method == 'equal':
            target_weight = 1.0 / len(target_stocks)
            target_weights = {stock: target_weight for stock in target_stocks}
        else:
            # Implement risk parity or other weighting schemes
            target_weights = self._calculate_risk_parity_weights(target_stocks, prices)
        
        # Buy/adjust positions
        portfolio_value = self.cash + sum([
            self.positions.get(s, {}).get('value', 0) for s in self.positions
        ])
        
        for stock, weight in target_weights.items():
            target_value = portfolio_value * min(weight, 0.05)  # Max 5% per stock
            self._adjust_position(stock, target_value, prices)
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics
        """
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['value'].pct_change()
        
        metrics = {
            'total_return': (equity_df['value'].iloc[-1] / self.initial_capital - 1) * 100,
            'annual_return': self._annualized_return(equity_df),
            'sharpe_ratio': self._sharpe_ratio(equity_df),
            'max_drawdown': self._max_drawdown(equity_df),
            'win_rate': (equity_df['returns'] > 0).mean() * 100
        }
        
        return metrics
    
    def _max_drawdown(self, equity_df):
        """
        Calculate maximum drawdown
        """
        cumulative = (1 + equity_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min() * 100
```

---

## **Stage 7: Hyperparameter Optimization**

```python
import optuna

class HyperparameterOptimizer:
    """
    Use Optuna for Bayesian optimization of model parameters
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def objective(self, trial):
        """
        Optuna objective function
        """
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'verbose': -1
        }
        
        # Train model
        query_groups = self.X_train.groupby('date').size().values
        feature_cols = [col for col in self.X_train.columns if col.endswith('_z')]
        
        train_data = lgb.Dataset(
            self.X_train[feature_cols],
            label=self.y_train,
            group=query_groups
        )
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data]
        )
        
        # Evaluate on validation set
        val_preds = model.predict(self.X_val[feature_cols])
        
        # Calculate NDCG@20
        from sklearn.metrics import ndcg_score
        ndcg_scores = []
        
        for date, group in self.X_val.groupby('date'):
            if len(group) > 20:
                y_true = self.y_val.loc[group.index].values.reshape(1, -1)
                y_pred = val_preds[group.index].reshape
(1, -1)
                ndcg = ndcg_score(y_true, y_pred, k=20)
                ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)
    
    def optimize(self, n_trials=100):
        """
        Run optimization
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"Best NDCG@20: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params
```

---

## **Stage 8: Production Deployment & Monitoring**

```python
class ProductionPipeline:
    """
    End-to-end pipeline for production deployment
    """
    
    def __init__(self):
        self.data_engine = DataAcquisitionEngine()
        self.feature_engineer = FeatureEngineer(self.data_engine)
        self.models = MultiHorizonRanker()
        self.ensemble = None
        
    def daily_update(self, target_date):
        """
        Daily workflow: fetch new data, generate signals
        """
        # 1. Fetch latest data
        universe = self.data_engine.fetch_stock_universe(target_date)
        
        # 2. Generate features
        features = self.feature_engineer.build_all_features(
            start_date=target_date - pd.Timedelta(days=1),
            end_date=target_date
        )
        
        # 3. Generate predictions
        signals = self.ensemble.predict(features)
        
        # 4. Select top stocks
        selected = self.ensemble.select_top_k(features, k=50)
        
        # 5. Output to portfolio management system
        self._export_signals(selected, target_date)
        
        # 6. Log performance
        self._log_performance(target_date)
        
        return selected
    
    def monthly_retraining(self):
        """
        Retrain models with latest data
        """
        # Fetch expanded dataset
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365*3)  # 3 years
        
        features = self.feature_engineer.build_all_features(start_date, end_date)
        labels = LabelGenerator().generate_returns(features)
        
        # Retrain models
        self.models.train_all_horizons(features, labels)
        
        # Update ensemble
        self.ensemble = MetaEnsemble(self.models)
        
        # Save models
        self._save_models()
```

---

## **Key Takeaways & Critical Design Decisions**

### **1. Data Integrity (Most Critical)**
- **Point-in-time alignment**: Use `announcement_date`, never `report_date`
- **Cross-sectional normalization**: Industry-neutral Z-scores prevent sector bias
- **Winsorization**: Cap at 2.5%/97.5% to avoid outlier contamination

### **2. A-Share Specifics**
- **Limit-up/down handling**: Exclude from portfolio on rebalance day
- **Suspension handling**: Forward-fill or exclude from universe
- **ST stock filtering**: Avoid delisting risk

### **3. Model Architecture**
- **Multi-horizon approach**: Separate models for 1m, 3m, 1y capture different alpha sources
- **Ranking objective**: Use `lambdarank` not regression
- **Ensemble method**: Weighted combination identifies high-conviction picks

### **4. Backtest Realism**
- **Monthly rebalancing**: Avoids alpha decay and transaction costs
- **5% max weight**: Matches typical A-share long-only constraints
- **Transaction costs**: 0.03% each way minimum

This architecture provides a production-ready foundation. The next step would be implementing each module with comprehensive unit tests and gradual rollout using paper trading before live deployment.