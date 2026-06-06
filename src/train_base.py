import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure config can be imported even if executed from another path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PROCESSED_DIR, MODEL_DIR, TARGET_COL, EXPECTED_RETURN_COL,
    FEE_RATE, SLIPPAGE, RESULTS_DIR
)

def calculate_financial_metrics(df_test, y_pred):
    """
    Calculates the financial metrics (Sharpe Ratio, Max Drawdown) for given predictions,
    taking into account trading fees and slippage.
    
    df_test: Test DataFrame containing at least the 'Close' column
    y_pred: Model predictions (1 = Buy/Hold, 0 = Sell/Out)
    """
    # Copy to avoid modifying the original dataframe
    df = df_test.copy()
    
    # The position we hold on day t+1 is the prediction made at the end of day t
    df['Position'] = y_pred
    df['Position_Prev'] = df['Position'].shift(1).fillna(0)
    
    # Buy & Hold Asset Return
    df['Asset_Return'] = df['Close'].pct_change()
    
    # Transaction costs (paid when changing position)
    # Position changes: 0 to 1 (buy) or 1 to 0 (sell)
    df['Trade_Cost'] = abs(df['Position'] - df['Position_Prev']) * (FEE_RATE + SLIPPAGE)
    
    # Strategy Return
    # The return at time t is achieved with the position decided at t-1
    df['Strategy_Return'] = df['Position_Prev'] * df['Asset_Return'] - df['Trade_Cost']
    
    # Clean initial NaNs
    df = df.dropna()
    
    if len(df) == 0:
        return 0.0, 0.0, 0.0
        
    # Calculate metrics
    # 1. Cumulative Return
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    total_return = df['Cum_Strategy_Return'].iloc[-1] - 1.0 if not df.empty else 0.0
    
    # 2. Sharpe Ratio (annualized, assuming daily data)
    mean_return = df['Strategy_Return'].mean()
    std_return = df['Strategy_Return'].std()
    
    if std_return > 0:
        sharpe_ratio = np.sqrt(365) * (mean_return / std_return)
    else:
        sharpe_ratio = 0.0
        
    # 3. Maximum Drawdown
    cum_returns = df['Cum_Strategy_Return']
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return total_return, sharpe_ratio, max_drawdown

def run_training():
    """
    Executes Phase 3: Training the baseline model (Random Forest) using 
    Walk-Forward Validation (TimeSeriesSplit).
    """
    print("Starting Phase 3: Baseline Training with Walk-Forward Validation")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load processed data
    data_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run features.py first.")
        
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # 2. Separate Features (X) and Target (y)
    # Ensure no future variables or strings are included
    cols_to_drop = [TARGET_COL, EXPECTED_RETURN_COL]
    
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]
    
    print(f"Total data points: {len(df)}")
    print(f"Features used: {list(X.columns)}")
    
    # 3. Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    sharpe_scores = []
    mdd_scores = []
    
    # Create pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        df_test = df.iloc[test_index]
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate ML Metrics (Secondary)
        acc = accuracy_score(y_test, y_pred)
        
        # Evaluate Financial Metrics (Primary)
        total_ret, sharpe, mdd = calculate_financial_metrics(df_test, y_pred)
        
        print(f"--- Fold {fold} ---")
        print(f"Train: {X_train.index[0].date()} to {X_train.index[-1].date()} | Test: {X_test.index[0].date()} to {X_test.index[-1].date()}")
        print(f"Accuracy: {acc:.4f} | Total Return: {total_ret*100:.2f}% | Sharpe Ratio: {sharpe:.4f} | Max Drawdown: {mdd*100:.2f}%")
        
        sharpe_scores.append(sharpe)
        mdd_scores.append(mdd)
        fold += 1
        
    print("\n=== Final Walk-Forward Results ===")
    avg_sharpe = np.mean(sharpe_scores)
    avg_mdd = np.mean(mdd_scores) * 100
    
    print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
    print(f"Average Max Drawdown: {avg_mdd:.2f}%")
    
    # Save results to file
    results_path = os.path.join(RESULTS_DIR, "baseline_metrics.txt")
    with open(results_path, "w") as f:
        f.write("=== Baseline Random Forest Results ===\n")
        f.write(f"Average Sharpe Ratio: {avg_sharpe:.4f}\n")
        f.write(f"Average Max Drawdown: {avg_mdd:.2f}%\n")
    print(f"Metrics saved to: {results_path}")
    
    # 4. Train final model with all historical data and save
    print("\nTraining final model with all historical data...")
    pipeline.fit(X, y)
    
    model_path = os.path.join(MODEL_DIR, "baseline_rf.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved at: {model_path}")

def run_xgboost_training():
    """
    Executes Phase 3: Training the baseline model (XGBoost) using 
    Walk-Forward Validation (TimeSeriesSplit).
    """
    print("\nStarting Phase 3: Baseline Training (XGBoost) with Walk-Forward Validation")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load processed data
    data_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run features.py first.")
        
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # 2. Separate Features (X) and Target (y)
    cols_to_drop = [TARGET_COL, EXPECTED_RETURN_COL]
    
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]
    
    # 3. Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    sharpe_scores = []
    mdd_scores = []
    
    # Create pipeline with scaler and model
    # scale_pos_weight is XGBoost's equivalent for class_weight='balanced'
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            scale_pos_weight=spw,
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        df_test = df.iloc[test_index]
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate Financial Metrics (Primary)
        total_ret, sharpe, mdd = calculate_financial_metrics(df_test, y_pred)
        
        print(f"--- XGB Fold {fold} ---")
        print(f"Total Return: {total_ret*100:.2f}% | Sharpe Ratio: {sharpe:.4f} | Max Drawdown: {mdd*100:.2f}%")
        
        sharpe_scores.append(sharpe)
        mdd_scores.append(mdd)
        fold += 1
        
    print("\n=== Final XGBoost Walk-Forward Results ===")
    avg_sharpe = np.mean(sharpe_scores)
    avg_mdd = np.mean(mdd_scores) * 100
    
    print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
    print(f"Average Max Drawdown: {avg_mdd:.2f}%")
    
    # Save results to file
    results_path = os.path.join(RESULTS_DIR, "baseline_xgboost_metrics.txt")
    with open(results_path, "w") as f:
        f.write("=== Baseline XGBoost Results ===\n")
        f.write(f"Average Sharpe Ratio: {avg_sharpe:.4f}\n")
        f.write(f"Average Max Drawdown: {avg_mdd:.2f}%\n")
    print(f"Metrics saved to: {results_path}")
    
    # 4. Train final model
    pipeline.fit(X, y)
    model_path = os.path.join(MODEL_DIR, "baseline_xgb.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved at: {model_path}")

if __name__ == "__main__":
    run_training()
    run_xgboost_training()
