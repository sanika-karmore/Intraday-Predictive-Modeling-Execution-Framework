import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        
    def parse_feature_hierarchy(self, columns):
        # Extract feature families based on underscore structure
        # Example: F_H_B -> family 'F', subfamily 'H', type 'B'
        hierarchy = {
            'price': [],      # P1, P2, P3, P4
            'current': [],    # C_* features
            'future': [],     # F_* features
            'implied': [],    # IC related
            'delta': [],      # D_* features
            'bid': [],        # B_* features
            'market': [],     # m0_*, m1_* features
            'holding': [],    # H_* features
            'order': [],      # O_* features
            'book': []        # BK_* features
        }
        
        for col in columns:
            if col.startswith('P') and col[1:].replace('_','').isdigit():
                hierarchy['price'].append(col)
            elif col.startswith('C_'):
                hierarchy['current'].append(col)
            elif col.startswith('F_'):
                hierarchy['future'].append(col)
            elif col.startswith('IC') or 'IC' in col:
                hierarchy['implied'].append(col)
            elif col.startswith('D_'):
                hierarchy['delta'].append(col)
            elif col.startswith('B_'):
                hierarchy['bid'].append(col)
            elif col.startswith('m0_') or col.startswith('m1_'):
                hierarchy['market'].append(col)
            elif '_H_' in col:
                hierarchy['holding'].append(col)
            elif '_O_' in col or col.startswith('O_'):
                hierarchy['order'].append(col)
            elif 'BK_' in col:
                hierarchy['book'].append(col)
                
        return hierarchy
    
    def create_causal_features(self, df, lookback=10):
        # Generate causal derived features using only past information
        # Assumption: lookback window of 10 bars is sufficient for momentum/volatility
        df = df.copy()
        
        # Price-based features (momentum, volatility)
        if 'P3' in df.columns:
            # Returns over multiple horizons
            df['P3_ret_1'] = df['P3'].pct_change(1)
            df['P3_ret_3'] = df['P3'].pct_change(3)
            df['P3_ret_5'] = df['P3'].pct_change(5)
            df['P3_ret_10'] = df['P3'].pct_change(10)
            
            # Rolling volatility (expanding to maintain causality)
            df['P3_vol_10'] = df['P3_ret_1'].rolling(window=10, min_periods=1).std()
            
            # Momentum indicators
            df['P3_mom_10'] = df['P3'] / df['P3'].shift(10) - 1
            
        # Cross-price spreads (if multiple prices available)
        price_cols = [c for c in df.columns if c.startswith('P') and c[1:].isdigit()]
        if len(price_cols) > 1:
            for i in range(len(price_cols)-1):
                p1, p2 = price_cols[i], price_cols[i+1]
                df[f'{p1}_{p2}_spread'] = df[p1] - df[p2]
                df[f'{p1}_{p2}_ratio'] = df[p1] / (df[p2] + 1e-10)
        
        # Feature family aggregations
        hierarchy = self.parse_feature_hierarchy(df.columns)
        
        # Current vs Future feature interactions
        if hierarchy['current'] and hierarchy['future']:
            # Mean of current features
            df['C_mean'] = df[hierarchy['current']].mean(axis=1)
            df['F_mean'] = df[hierarchy['future']].mean(axis=1)
            df['CF_spread'] = df['C_mean'] - df['F_mean']
            
        # Market microstructure features (m0 vs m1)
        m0_cols = [c for c in df.columns if c.startswith('m0_')]
        m1_cols = [c for c in df.columns if c.startswith('m1_')]
        
        if m0_cols and m1_cols:
            # Find matching pairs
            m0_base = [c.replace('m0_', '') for c in m0_cols]
            m1_base = [c.replace('m1_', '') for c in m1_cols]
            common = set(m0_base) & set(m1_base)
            
            for base in common:
                m0_col = f'm0_{base}'
                m1_col = f'm1_{base}'
                if m0_col in df.columns and m1_col in df.columns:
                    df[f'm0m1_{base}_diff'] = df[m0_col] - df[m1_col]
        
        # Book imbalance features
        book_cols = [c for c in df.columns if 'BK_' in c]
        if book_cols:
            df['BK_aggregate'] = df[book_cols].mean(axis=1)
            
        return df
    
    def engineer_target(self, df, horizon=30):
        # Create target: future return of P3 at specified horizon
        # Target = (P3[t+horizon] - P3[t]) / P3[t]
        # Assumption: horizon >= 30 bars as per problem requirement
        if horizon < 30:
            raise ValueError(f"Horizon must be >= 30 bars, got {horizon}")
            
        # Forward return calculation
        target = (df['P3'].shift(-horizon) - df['P3']) / df['P3']
        
        return target


class ModelEnsemble:
    # Multi-model system with automatic fallback
    # Priority: LightGBM -> XGBoost -> Ridge Regression
    
    def __init__(self, horizon=30):
        self.horizon = horizon
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X, y):
        # Train models with fallback mechanism
        # Returns: model type used
        # Remove NaN from target
        valid_idx = ~(y.isna() | np.isinf(y))
        X_clean = X[valid_idx].copy()
        y_clean = y[valid_idx].copy()
        
        if len(y_clean) < 100:
            raise ValueError(f"Insufficient training data: {len(y_clean)} samples")
        
        # Handle infinite/NaN in features
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(method='ffill').fillna(0)
        
        self.feature_names = X_clean.columns.tolist()
        
        # Attempt LightGBM first (fastest, most robust)
        try:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 7,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
            
            train_data = lgb.Dataset(X_clean, label=y_clean)
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            self.model_type = 'lightgbm'
            return 'lightgbm'
            
        except Exception as e:
            print(f"LightGBM failed: {e}, trying XGBoost")
            
            # Fallback to XGBoost
            try:
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'verbosity': 0
                }
                
                dtrain = xgb.DMatrix(X_clean, label=y_clean)
                self.model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=200,
                    evals=[(dtrain, 'train')],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
                self.model_type = 'xgboost'
                return 'xgboost'
                
            except Exception as e2:
                print(f"XGBoost failed: {e2}, falling back to Ridge")
                
                # Final fallback: Ridge regression (always works)
                X_scaled = self.scaler.fit_transform(X_clean)
                self.model = Ridge(alpha=1.0)
                self.model.fit(X_scaled, y_clean)
                self.model_type = 'ridge'
                return 'ridge'
    
    def predict(self, X):
        # Generate predictions using trained model
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Ensure feature alignment
        X_aligned = X[self.feature_names].copy()
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if self.model_type == 'lightgbm':
            return self.model.predict(X_aligned)
        elif self.model_type == 'xgboost':
            dmat = xgb.DMatrix(X_aligned)
            return self.model.predict(dmat)
        else:  # ridge
            X_scaled = self.scaler.transform(X_aligned)
            return self.model.predict(X_scaled)


class ExecutionEngine:
    # Iterative trading execution with proper cost accounting
    # Operates strictly on P3 with 0.01% transaction costs
    
    def __init__(self, tc_rate=0.0001):
        self.tc_rate = tc_rate
        self.reset()
        
    def reset(self):
        # Reset execution state
        self.position = 0  # -1, 0, +1
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_costs = 0.0
        self.trades = []
        
    def generate_signal(self, prediction, threshold=0.0001):
        # Convert model prediction to trading signal
        # Assumption: threshold of 0.01% (0.0001) filters noise
        # prediction > threshold: long (+1)
        # prediction < -threshold: short (-1)
        # otherwise: flat (0)
        if prediction > threshold:
            return 1
        elif prediction < -threshold:
            return -1
        else:
            return 0
    
    def execute_trade(self, timestamp, current_price, signal, prediction):
        # Execute trade with cost accounting
        # Returns trade record with PnL accounting
        trade_record = {
            'timestamp': timestamp,
            'price': current_price,
            'signal': signal,
            'prediction': prediction,
            'position_before': self.position,
            'position_after': signal,
            'entry_price': self.entry_price,  # Track current entry price
            'trade_cost': 0.0,
            'realized_pnl': 0.0,
            'mtm_pnl': 0.0
        }
        
        # Check if position change occurs
        if signal != self.position:
            # Close existing position if any
            if self.position != 0:
                # Realize PnL
                pnl = self.position * (current_price - self.entry_price)
                self.realized_pnl += pnl
                trade_record['realized_pnl'] = pnl
                
                # Transaction cost for exit
                cost = abs(current_price * self.tc_rate)
                self.total_costs += cost
                trade_record['trade_cost'] += cost
            
            # Open new position if signal != 0
            if signal != 0:
                self.entry_price = current_price
                trade_record['entry_price'] = current_price  # Update with new entry
                # Transaction cost for entry
                cost = abs(current_price * self.tc_rate)
                self.total_costs += cost
                trade_record['trade_cost'] += cost
            else:
                trade_record['entry_price'] = 0.0  # No position
            
            self.position = signal
        
        # Calculate MTM PnL for open positions
        if self.position != 0:
            mtm_pnl = self.position * (current_price - self.entry_price)
            trade_record['mtm_pnl'] = mtm_pnl
        
        self.trades.append(trade_record)
        return trade_record
    
    def get_cumulative_pnl(self):
        # Get total PnL including costs
        return self.realized_pnl - self.total_costs
    
    def close_all_positions(self, final_price, timestamp):
        # Force close all positions at end of day
        if self.position != 0:
            pnl = self.position * (final_price - self.entry_price)
            self.realized_pnl += pnl
            cost = abs(final_price * self.tc_rate)
            self.total_costs += cost
            
            self.trades.append({
                'timestamp': timestamp,
                'price': final_price,
                'signal': 0,
                'prediction': 0,
                'position_before': self.position,
                'position_after': 0,
                'entry_price': self.entry_price,
                'trade_cost': cost,
                'realized_pnl': pnl,
                'mtm_pnl': 0.0
            })
            
            self.position = 0
            self.entry_price = 0.0


class TradingStrategy:
    # Main strategy orchestrator
    # Combines feature engineering, modeling, and execution
    
    def __init__(self, horizon=30, lookback_window=10, signal_threshold=0.0001):
        # Assumptions:
        # - horizon: 30 bars minimum (per problem statement)
        # - lookback_window: 10 bars for feature engineering
        # - signal_threshold: 0.01% to filter noise
        if horizon < 30:
            raise ValueError("Horizon must be >= 30 bars")
            
        self.horizon = horizon
        self.lookback_window = lookback_window
        self.signal_threshold = signal_threshold
        
        self.feature_engineer = FeatureEngineer()
        self.model = ModelEnsemble(horizon=horizon)
        self.execution = ExecutionEngine()
        
    def prepare_training_data(self, df):
        # Prepare features and target for model training
        # All operations are causal
        
        # Engineer features
        df_features = self.feature_engineer.create_causal_features(
            df, lookback=self.lookback_window
        )
        
        # Create target
        target = self.feature_engineer.engineer_target(df_features, horizon=self.horizon)
        
        # Remove target-related timestamps (last horizon rows)
        valid_idx = ~target.isna()
        X = df_features[valid_idx].copy()
        y = target[valid_idx].copy()
        
        # Remove non-feature columns
        feature_cols = [c for c in X.columns if c not in ['ts', 'ts_ns']]
        X = X[feature_cols]
        
        return X, y
    
    def train_on_historical_days(self, file_paths, max_days=None):
        # Train model on multiple historical days (expanding window)
        # Assumption: earlier files are earlier trading days
        all_X = []
        all_y = []
        
        files_to_use = file_paths[:max_days] if max_days else file_paths
        
        for fp in files_to_use:
            try:
                df = pd.read_csv(fp)
                X, y = self.prepare_training_data(df)
                all_X.append(X)
                all_y.append(y)
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No valid training data loaded")
        
        # Combine all data
        X_combined = pd.concat(all_X, axis=0, ignore_index=True)
        y_combined = pd.concat(all_y, axis=0, ignore_index=True)
        
        # Train model
        model_type = self.model.train(X_combined, y_combined)
        print(f"Trained {model_type} on {len(files_to_use)} days, {len(y_combined)} samples")
        
        return model_type
    
    def execute_day(self, df):
        # Execute trading strategy for a single day iteratively
        # Returns DataFrame with required columns per problem statement:
        # - timestamp, price, position, entry_price
        # - realized_pnl, mtm_pnl, transaction_costs, cumulative_pnl
        # - signal, prediction
        self.execution.reset()
        
        # Engineer features for entire day (still causal - only uses past)
        df_features = self.feature_engineer.create_causal_features(
            df, lookback=self.lookback_window
        )
        
        # Remove non-feature columns
        feature_cols = [c for c in df_features.columns if c not in ['ts', 'ts_ns', 'P3']]
        X = df_features[feature_cols].copy()
        
        # Iterative execution
        for idx in range(len(df)):
            # Only trade if we have enough history and enough future bars
            if idx < self.lookback_window or idx >= len(df) - self.horizon:
                continue
            
            # Use only data up to current timestamp
            X_current = X.iloc[:idx+1].tail(1)
            
            try:
                # Generate prediction
                prediction = self.model.predict(X_current)[0]
                
                # Convert to signal
                signal = self.execution.generate_signal(
                    prediction, threshold=self.signal_threshold
                )
                
                # Execute trade
                current_price = df.iloc[idx]['P3']
                timestamp = df.iloc[idx].get('ts', idx)
                
                self.execution.execute_trade(
                    timestamp, current_price, signal, prediction
                )
                
            except Exception as e:
                # On error, maintain current position (do nothing)
                continue
        
        # Close all positions at end of day
        final_price = df.iloc[-1]['P3']
        final_timestamp = df.iloc[-1].get('ts', len(df)-1)
        self.execution.close_all_positions(final_price, final_timestamp)
        
        # Convert trades to DataFrame with standardized column names
        trades_df = pd.DataFrame(self.execution.trades)
        
        if len(trades_df) > 0:
            # Rename columns to match problem statement specification
            trades_df = trades_df.rename(columns={
                'position_after': 'position',
                'trade_cost': 'transaction_costs'
            })
            
            # Calculate cumulative PnL
            trades_df['cumulative_pnl'] = (
                trades_df['realized_pnl'].cumsum() - trades_df['transaction_costs'].cumsum()
            )
            
            # Reorder columns for clarity
            column_order = [
                'timestamp', 'price', 'signal', 'prediction',
                'position_before', 'position', 'entry_price',
                'realized_pnl', 'mtm_pnl', 'transaction_costs', 'cumulative_pnl'
            ]
            
            # Only reorder if all columns exist
            existing_cols = [c for c in column_order if c in trades_df.columns]
            other_cols = [c for c in trades_df.columns if c not in column_order]
            trades_df = trades_df[existing_cols + other_cols]
        
        return trades_df


def run_backtest(train_dir=r'C:\Users\arjun\Downloads\train',
                 output_dir='./outputs',
                 initial_train_days=50,
                 retrain_frequency=5,
                 horizon=30):
    # Main execution function with EXPANDING WINDOW training
    # Assumptions:
    # - initial_train_days: 50 days for initial model (minimum stable training set)
    # - retrain_frequency: retrain every 5 days to balance compute vs adaptability
    # - horizon: 30 bars (minimum per problem statement)
    # Process:
    # - Days 1-50: train model
    # - Day 51: execute with model trained on days 1-50
    # - Day 55: retrain on days 1-55, execute day 55
    # - Day 60: retrain on days 1-60, execute day 60
    # - etc. (expanding window continuously learns from ALL past data)
    
    import os
    import glob
    
    # Get all CSV files sorted numerically
    pattern = os.path.join(train_dir, '*.csv')
    files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {train_dir}")
    
    print(f"Found {len(files)} trading days")
    print(f"Using EXPANDING WINDOW with retrain every {retrain_frequency} days")
    
    # Initialize strategy
    strategy = TradingStrategy(
        horizon=horizon,
        lookback_window=10,
        signal_threshold=0.0001
    )
    
    # Initial training on first N days
    print(f"\n[INITIAL TRAINING] Using first {initial_train_days} days...")
    train_files = files[:initial_train_days]
    model_type = strategy.train_on_historical_days(train_files)
    print(f"Initial model trained: {model_type}")
    
    # Execute on remaining days with periodic retraining
    all_results = []
    
    for i in range(initial_train_days, len(files)):
        day_num = int(os.path.basename(files[i]).split('.')[0])
        
        # EXPANDING WINDOW: Retrain on all data up to current day
        if (i - initial_train_days) % retrain_frequency == 0 and i > initial_train_days:
            print(f"\n[RETRAIN] Day {day_num}: Training on days 1-{i} ({i} days total)")
            retrain_files = files[:i]
            try:
                model_type = strategy.train_on_historical_days(retrain_files)
                print(f"Model updated: {model_type}")
            except Exception as e:
                print(f"Retraining failed: {e}, continuing with previous model")
        
        # Execute current day
        try:
            df = pd.read_csv(files[i])
            trades_df = strategy.execute_day(df)
            
            # Add metadata
            trades_df['day'] = day_num
            trades_df['file'] = os.path.basename(files[i])
            trades_df['training_days_used'] = i  # Track how many days model was trained on
            
            all_results.append(trades_df)
            
            # Progress update
            if len(trades_df) > 0:
                day_pnl = strategy.execution.get_cumulative_pnl()
                print(f"Day {day_num}: {len(trades_df)} trades, PnL: {day_pnl:.4f}")
            
        except Exception as e:
            print(f"Error processing day {day_num}: {e}")
            continue
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, axis=0, ignore_index=True)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'all_trades.csv')
        final_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Summary statistics
        total_pnl = final_results['realized_pnl'].sum() - final_results['trade_cost'].sum()
        total_trades = len(final_results)
        winning_trades = len(final_results[final_results['realized_pnl'] > 0])
        days_traded = final_results['day'].nunique()
        
        print("\n=== BACKTEST SUMMARY (EXPANDING WINDOW) ===")
        print(f"Model Type: {model_type}")
        print(f"Total Trading Days: {days_traded}")
        print(f"Initial Training: {initial_train_days} days")
        print(f"Final Training Size: {len(files)-1} days (all historical data)")
        print(f"Retrain Frequency: Every {retrain_frequency} days")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({100*winning_trades/max(total_trades,1):.1f}%)")
        print(f"Total PnL: {total_pnl:.4f}")
        print(f"Average PnL per Trade: {total_pnl/max(total_trades,1):.6f}")
        print(f"Average PnL per Day: {total_pnl/max(days_traded,1):.4f}")
        
        return final_results
    else:
        print("No results generated")
        return pd.DataFrame()


# Command-line execution
if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Intraday Trading Strategy Execution')
    parser.add_argument('--input', type=str, 
                      default=r'C:\Users\arjun\Downloads\train',
                      help='Path to input CSV file or directory')
    parser.add_argument('--output', type=str, 
                      default='all_trades.csv',
                      help='Path to output trades CSV file')
    parser.add_argument('--horizon', type=int, default=30,
                      help='Prediction horizon in bars (default: 30)')
    parser.add_argument('--train-days', type=int, default=50,
                      help='Initial training days for expanding window (default: 50)')
    parser.add_argument('--retrain-freq', type=int, default=5,
                      help='Retrain frequency in days (default: 5)')
    
    args = parser.parse_args()
    
    import os
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        # Single day execution
        print(f"Executing single day: {args.input}")
        
        # Need a trained model - check if we can find training data
        input_dir = os.path.dirname(args.input)
        all_files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith('.csv')],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0
        )
        
        if len(all_files) < args.train_days:
            print(f"WARNING: Need at least {args.train_days} files for training, found {len(all_files)}")
            print("Using all available files for training")
            train_files = [os.path.join(input_dir, f) for f in all_files[:-1]]
        else:
            train_files = [os.path.join(input_dir, f) for f in all_files[:args.train_days]]
        
        # Initialize and train
        strategy = TradingStrategy(
            horizon=args.horizon,
            lookback_window=10,
            signal_threshold=0.0001
        )
        
        print(f"Training on {len(train_files)} files...")
        model_type = strategy.train_on_historical_days(train_files)
        print(f"Model trained: {model_type}")
        
        # Execute on target day
        df = pd.read_csv(args.input)
        trades_df = strategy.execute_day(df)
        
        # Add metadata
        trades_df['day'] = os.path.basename(args.input).split('.')[0]
        trades_df['file'] = os.path.basename(args.input)
        
        # Save output
        trades_df.to_csv(args.output, index=False)
        
        # Summary
        total_pnl = strategy.execution.get_cumulative_pnl()
        print(f"\nResults saved to: {args.output}")
        print(f"Total trades: {len(trades_df)}")
        print(f"Day PnL: {total_pnl:.4f}")
        
    elif os.path.isdir(args.input):
        # Directory mode - run full backtest
        results = run_backtest(
            train_dir=args.input,
            output_dir=os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
            initial_train_days=args.train_days,
            retrain_frequency=args.retrain_freq,
            horizon=args.horizon
        )
        # Save to specified output path
        results.to_csv(args.output, index=False)
        print(f"\nAll results saved to: {args.output}")
    else:
        raise ValueError(f"Input path not found: {args.input}")
