# Intraday Trading Strategy with Machine Learning

A Python-based quantitative trading system that uses machine learning to predict price movements and execute trades with proper risk management and transaction cost accounting.

## Overview

This implementation provides an end-to-end algorithmic trading pipeline that:
- Engineers causal features from market microstructure data
- Trains ensemble ML models (LightGBM, XGBoost, Ridge Regression) with automatic fallback
- Generates trading signals based on price predictions
- Executes trades with realistic transaction costs (0.01%)
- Supports expanding window backtesting with periodic model retraining

## Key Features

### 1. **Causal Feature Engineering**
- Price-based momentum and volatility indicators
- Cross-price spreads and ratios
- Current vs. Future feature interactions
- Market microstructure features (m0/m1 aggregations)
- Order book imbalance metrics
- **All features use only historical data** (no look-ahead bias)

### 2. **Multi-Model Ensemble**
- **Primary**: LightGBM (fast, robust gradient boosting)
- **Fallback 1**: XGBoost (if LightGBM fails)
- **Fallback 2**: Ridge Regression (guaranteed to work)
- Automatic model selection based on availability

### 3. **Realistic Execution**
- Transaction cost: 0.01% per trade
- Position management: Long (+1), Flat (0), Short (-1)
- Entry/exit price tracking
- Realized P&L and mark-to-market P&L
- End-of-day position closure

### 4. **Expanding Window Training**
- Initial training on 50 days (configurable)
- Periodic retraining every 5 days (configurable)
- Uses all historical data up to current point
- Continuously adapts to market conditions

## Requirements

```bash
numpy
pandas
lightgbm
xgboost
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bookish-computing-machine-main
```

2. Install dependencies:
```bash
pip install numpy pandas lightgbm xgboost scikit-learn
```

## Usage

### Command Line Interface

#### Full Backtest (Directory Mode)
Run backtest on all CSV files in a directory:

```bash
python quant.py --input C:\path\to\train --output all_trades.csv
```

**Arguments:**
- `--input`: Path to directory containing CSV files (one file per day)
- `--output`: Path for output CSV file (default: `all_trades.csv`)
- `--horizon`: Prediction horizon in bars (default: 30, minimum: 30)
- `--train-days`: Initial training days (default: 50)
- `--retrain-freq`: Retrain frequency in days (default: 5)

**Example:**
```bash
python quant.py --input ./train_data --output results.csv --train-days 50 --retrain-freq 5 --horizon 30
```

#### Single Day Execution
Execute strategy on a single day:

```bash
python quant.py --input C:\path\to\train\123.csv --output day_123_trades.csv
```

### Python API

```python
from quant import TradingStrategy, run_backtest

# Initialize strategy
strategy = TradingStrategy(
    horizon=30,              # Prediction horizon (bars)
    lookback_window=10,      # Feature engineering window
    signal_threshold=0.0001  # 0.01% threshold for signals
)

# Train on historical data
train_files = ['day1.csv', 'day2.csv', ..., 'day50.csv']
strategy.train_on_historical_days(train_files)

# Execute on new day
import pandas as pd
df = pd.read_csv('new_day.csv')
trades = strategy.execute_day(df)
print(trades)

# Or run full backtest
results = run_backtest(
    train_dir='./train',
    output_dir='./outputs',
    initial_train_days=50,
    retrain_frequency=5,
    horizon=30
)
```

## Input Data Format

CSV files should contain market microstructure data with the following columns:

### Required Columns
- `P3`: Primary price (used for trading and target calculation)
- `ts` or `ts_ns`: Timestamp (optional, for tracking)

### Optional Feature Columns
- **Price columns**: `P1`, `P2`, `P4` (additional price levels)
- **Current features**: `C_*` (current state indicators)
- **Future features**: `F_*` (forward-looking indicators)
- **Delta features**: `D_*` (change indicators)
- **Bid features**: `B_*` (bid-side data)
- **Market features**: `m0_*`, `m1_*` (market microstructure)
- **Holding features**: `*_H_*` (position-related)
- **Order features**: `*_O_*` (order flow)
- **Book features**: `BK_*` (order book)
- **Implied features**: `IC*` (implied calculations)

**Example:**
```csv
ts,P1,P2,P3,P4,C_A,C_B,F_A,F_B,m0_vol,m1_vol,BK_ask,BK_bid
1000,100.1,100.2,100.3,100.4,0.5,0.3,0.6,0.4,1000,1100,500,450
1001,100.2,100.3,100.4,100.5,0.6,0.4,0.7,0.5,1050,1150,520,480
...
```

## Output Format

The strategy outputs a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Trade timestamp |
| `price` | Execution price (P3) |
| `signal` | Trading signal: +1 (long), 0 (flat), -1 (short) |
| `prediction` | Model prediction (future return) |
| `position_before` | Position before trade |
| `position` | Position after trade |
| `entry_price` | Entry price for current position |
| `realized_pnl` | Realized profit/loss for closed trades |
| `mtm_pnl` | Mark-to-market P&L for open positions |
| `transaction_costs` | Transaction costs incurred |
| `cumulative_pnl` | Cumulative P&L (realized - costs) |
| `day` | Day number |
| `file` | Source file name |
| `training_days_used` | Number of days used for training |

## Architecture

### Class Structure

```
TradingStrategy (Main Orchestrator)
├── FeatureEngineer
│   ├── parse_feature_hierarchy()
│   ├── create_causal_features()
│   └── engineer_target()
├── ModelEnsemble
│   ├── train()
│   └── predict()
└── ExecutionEngine
    ├── generate_signal()
    ├── execute_trade()
    ├── close_all_positions()
    └── get_cumulative_pnl()
```

### Workflow

1. **Feature Engineering**: Extract causal features from raw market data
2. **Target Creation**: Calculate future returns at specified horizon
3. **Model Training**: Train ML models on historical data
4. **Signal Generation**: Convert predictions to trading signals (+1/0/-1)
5. **Trade Execution**: Execute trades with cost accounting
6. **Position Management**: Track P&L and close positions at day end
7. **Periodic Retraining**: Update model with expanding data window

## Key Parameters

### Strategy Parameters
- **horizon**: Prediction horizon in bars (minimum: 30)
  - Higher values = longer-term predictions
  - Lower values = more frequent trading signals
- **lookback_window**: Bars used for feature engineering (default: 10)
  - Determines momentum/volatility calculation window
- **signal_threshold**: Minimum prediction magnitude to trade (default: 0.0001)
  - Higher threshold = fewer trades, stronger signals only
  - Lower threshold = more trades, responds to smaller predictions

### Backtest Parameters
- **initial_train_days**: Days for initial model training (default: 50)
  - More days = better initial model, slower startup
- **retrain_frequency**: Days between model updates (default: 5)
  - Lower frequency = more adaptive, higher compute cost
  - Higher frequency = more stable, less compute

### Model Hyperparameters

**LightGBM:**
- `num_leaves`: 31 (model complexity)
- `learning_rate`: 0.05 (training speed vs. accuracy)
- `max_depth`: 7 (tree depth limit)
- `feature_fraction`: 0.8 (random feature sampling)
- `bagging_fraction`: 0.8 (random row sampling)
- `lambda_l1/l2`: 0.1 (regularization)

**Transaction Costs:**
- Rate: 0.01% per trade (0.0001)
- Applied on both entry and exit

## Algorithm Details

### Signal Generation
```python
if prediction > threshold:
    signal = +1  # Long
elif prediction < -threshold:
    signal = -1  # Short
else:
    signal = 0   # Flat
```

### Position Management
- Only one position at a time: -1, 0, or +1
- Position changes trigger trade execution
- All positions closed at end of day

### P&L Calculation
```python
realized_pnl = position * (exit_price - entry_price)
transaction_costs = |price * 0.0001| * 2  # entry + exit
net_pnl = realized_pnl - transaction_costs
```

## Assumptions

1. **Data Quality**: CSV files contain valid market data with required columns
2. **Temporal Order**: Files are numbered sequentially (earlier = older)
3. **Bar Frequency**: All bars have consistent time intervals
4. **Liquidity**: Trades execute at observed P3 price (no slippage modeling)
5. **Single Asset**: Strategy trades only one instrument (P3)
6. **Minimum Horizon**: 30 bars required for predictions
7. **Transaction Costs**: Fixed at 0.01% regardless of trade size
8. **Market Hours**: Each CSV represents a complete trading session

## Performance Metrics

The backtest summary provides:
- **Total Trading Days**: Number of days traded
- **Total Trades**: Number of position changes
- **Winning Trades**: Trades with positive realized P&L
- **Win Rate**: Percentage of winning trades
- **Total P&L**: Cumulative profit/loss after costs
- **Average P&L per Trade**: Total P&L / Number of trades
- **Average P&L per Day**: Total P&L / Trading days

## Example Output

```
=== BACKTEST SUMMARY (EXPANDING WINDOW) ===
Model Type: lightgbm
Total Trading Days: 150
Initial Training: 50 days
Final Training Size: 199 days (all historical data)
Retrain Frequency: Every 5 days
Total Trades: 487
Winning Trades: 263 (54.0%)
Total PnL: 0.1234
Average PnL per Trade: 0.000253
Average PnL per Day: 0.0008
```

## Troubleshooting

### Model Training Fails
- **Issue**: All models fail to train
- **Solution**: Check data quality, ensure sufficient samples (>100 rows)

### No Trades Generated
- **Issue**: Empty trades DataFrame
- **Solution**: 
  - Reduce `signal_threshold`
  - Check prediction magnitudes
  - Ensure sufficient data (> lookback_window + horizon)

### Memory Errors
- **Issue**: Out of memory on large datasets
- **Solution**: 
  - Reduce `initial_train_days`
  - Increase `retrain_frequency`
  - Process days in batches

### Poor Performance
- **Issue**: Negative total P&L
- **Possible causes**:
  - Overfitting: Reduce model complexity or increase regularization
  - High transaction costs: Increase `signal_threshold` to reduce trades
  - Insufficient training data: Increase `initial_train_days`

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]

## Acknowledgments

- LightGBM: Microsoft
- XGBoost: DMLC
- scikit-learn: scikit-learn developers
