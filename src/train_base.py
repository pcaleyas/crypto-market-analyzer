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
import mlflow
import shap
import matplotlib.pyplot as plt

# Ensure config can be imported even if executed from another path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
logger = get_logger(__name__)

mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
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
    logger.info("Starting Phase 3: Baseline Training with Walk-Forward Validation")
    mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Crypto_Baseline")
    
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
    
    logger.info(f"Total data points: {len(df)}")
    logger.info(f"Features used: {list(X.columns)}")
    
    # 3. Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    sharpe_scores = []
    mdd_scores = []
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    with mlflow.start_run(run_name="RandomForest_CV"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("class_weight", "balanced")
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            df_test = df.iloc[test_index]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            total_ret, sharpe, mdd = calculate_financial_metrics(df_test, y_pred)
            
            logger.info(f"--- Fold {fold} ---")
            logger.info(f"Train: {X_train.index[0].date()} to {X_train.index[-1].date()} | Test: {X_test.index[0].date()} to {X_test.index[-1].date()}")
            logger.info(f"Accuracy: {acc:.4f} | Total Return: {total_ret*100:.2f}% | Sharpe Ratio: {sharpe:.4f} | Max Drawdown: {mdd*100:.2f}%")
            
            mlflow.log_metric(f"fold_{fold}_sharpe", sharpe)
            mlflow.log_metric(f"fold_{fold}_mdd", mdd)
            
            sharpe_scores.append(sharpe)
            mdd_scores.append(mdd)
            fold += 1
            
        avg_sharpe = np.mean(sharpe_scores)
        avg_mdd = np.mean(mdd_scores) * 100
        
        logger.info("\n=== Final Walk-Forward Results ===")
        logger.info(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
        logger.info(f"Average Max Drawdown: {avg_mdd:.2f}%")
        
        mlflow.log_metric("avg_cv_sharpe", avg_sharpe)
        mlflow.log_metric("avg_cv_mdd", avg_mdd)
        
        results_path = os.path.join(RESULTS_DIR, "baseline_metrics.txt")
        with open(results_path, "w") as f:
            f.write("=== Baseline Random Forest Results ===\n")
            f.write(f"Average Sharpe Ratio: {avg_sharpe:.4f}\n")
            f.write(f"Average Max Drawdown: {avg_mdd:.2f}%\n")
        logger.info(f"Metrics saved to: {results_path}")
        
        logger.info("Training final model with all historical data...")
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, "model")
        
        model_path = os.path.join(MODEL_DIR, "baseline_rf.joblib")
        joblib.dump(pipeline, model_path)
        logger.info(f"Model successfully saved at: {model_path}")
        
        # 5. SHAP Analysis
        logger.info("Generating SHAP importance plot for Random Forest...")
        rf_model = pipeline.named_steps['rf']
        scaler = pipeline.named_steps['scaler']
        
        # We take a sample to speed up SHAP calculation for RF if needed, 
        # but 3000 rows is small enough.
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        explainer = shap.TreeExplainer(rf_model)
        
        # For RF classifier, shap_values is a list. We take the positive class [1].
        # In newer shap versions it might return an array of shape (n_samples, n_features, n_classes)
        shap_values_raw = explainer.shap_values(X_scaled)
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        elif len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:,:,1]
        else:
            shap_values = shap_values_raw
            
        plt.figure()
        shap.summary_plot(shap_values, X_scaled, show=False)
        shap_plot_path = os.path.join(RESULTS_DIR, "shap_summary_rf.png")
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary plot saved to {shap_plot_path}")
        mlflow.log_artifact(shap_plot_path)

def run_xgboost_training():
    """
    Executes Phase 3: Training the baseline model (XGBoost) using 
    Walk-Forward Validation (TimeSeriesSplit).
    """
    logger.info("Starting Phase 3: Baseline Training (XGBoost) with Walk-Forward Validation")
    mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Crypto_Baseline")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    data_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run features.py first.")
        
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    cols_to_drop = [TARGET_COL, EXPECTED_RETURN_COL]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1
    sharpe_scores = []
    mdd_scores = []
    
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
    
    with mlflow.start_run(run_name="XGBoost_CV"):
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("scale_pos_weight", spw)
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            df_test = df.iloc[test_index]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            total_ret, sharpe, mdd = calculate_financial_metrics(df_test, y_pred)
            
            logger.info(f"--- XGB Fold {fold} ---")
            logger.info(f"Total Return: {total_ret*100:.2f}% | Sharpe Ratio: {sharpe:.4f} | Max Drawdown: {mdd*100:.2f}%")
            
            mlflow.log_metric(f"fold_{fold}_sharpe", sharpe)
            mlflow.log_metric(f"fold_{fold}_mdd", mdd)
            
            sharpe_scores.append(sharpe)
            mdd_scores.append(mdd)
            fold += 1
            
        avg_sharpe = np.mean(sharpe_scores)
        avg_mdd = np.mean(mdd_scores) * 100
        
        logger.info("\n=== Final XGBoost Walk-Forward Results ===")
        logger.info(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
        logger.info(f"Average Max Drawdown: {avg_mdd:.2f}%")
        
        mlflow.log_metric("avg_cv_sharpe", avg_sharpe)
        mlflow.log_metric("avg_cv_mdd", avg_mdd)
        
        results_path = os.path.join(RESULTS_DIR, "baseline_xgboost_metrics.txt")
        with open(results_path, "w") as f:
            f.write("=== Baseline XGBoost Results ===\n")
            f.write(f"Average Sharpe Ratio: {avg_sharpe:.4f}\n")
            f.write(f"Average Max Drawdown: {avg_mdd:.2f}%\n")
        logger.info(f"Metrics saved to: {results_path}")
        
        logger.info("Training final model...")
        pipeline.fit(X, y)
        mlflow.xgboost.log_model(pipeline.named_steps['xgb'], "xgb_model")
        
        model_path = os.path.join(MODEL_DIR, "baseline_xgb.joblib")
        joblib.dump(pipeline, model_path)
        logger.info(f"Model successfully saved at: {model_path}")

if __name__ == "__main__":
    run_training()
    run_xgboost_training()
