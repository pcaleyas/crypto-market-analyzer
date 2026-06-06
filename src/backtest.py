import os
import sys
import pandas as pd
import numpy as np
import vectorbt as vbt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Suppress tensorflow output for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, TARGET_COL, EXPECTED_RETURN_COL,
    FEE_RATE, SLIPPAGE, OOS_START_DATE
)

def create_sequences(X, time_steps=14):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

def run_backtest():
    print("Starting Phase 5: Backtesting Engine")
    
    # 1. Load Data
    data_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    cols_to_drop = [TARGET_COL, EXPECTED_RETURN_COL]
    X_df = df.drop(columns=cols_to_drop)
    
    # 2. Select Out-of-Sample Period
    # We take the data from OOS_START_DATE to give a robust out-of-sample period
    # ensuring the model hasn't learned these specific movements
    df_oos = df.loc[OOS_START_DATE:].copy()
    X_oos = X_df.loc[OOS_START_DATE:].copy()
    
    if len(df_oos) < 20:
        print("Not enough data for out-of-sample backtest.")
        return

    # 3. Load Model and Scaler
    model_path = os.path.join(MODEL_DIR, "dl_lstm_dyn_threshold.keras")
    scaler_path = os.path.join(MODEL_DIR, "dl_lstm_scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Run train_dl.py first.")
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # 4. Preprocess
    time_steps = 14
    X_oos_scaled = scaler.transform(X_oos)
    X_seq = create_sequences(X_oos_scaled, time_steps)
    
    # Generate predictions
    y_pred_prob = model.predict(X_seq, verbose=0).flatten()
    
    # We apply the threshold. 0.35 was consistently a good dynamic threshold in experiments
    THRESHOLD = 0.35 
    signals = (y_pred_prob > THRESHOLD).astype(int)
    
    # Align the signals with the dataframe
    df_oos_aligned = df_oos.iloc[time_steps:].copy()
    
    entries = signals == 1
    exits = signals == 0
    
    # 5. Run Vectorbt Simulation
    price = df_oos_aligned['Close']
    
    portfolio = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=10000,
        fees=FEE_RATE,
        slippage=SLIPPAGE,
        freq='1d' # Daily data
    )
    
    # 6. Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "backtest_report.txt")
    stats = portfolio.stats()
    
    with open(report_path, "w") as f:
        f.write("=== VECTORBT BACKTEST REPORT ===\n")
        f.write(f"Model: LSTM Dynamic Threshold (>{THRESHOLD})\n")
        f.write(f"Period: {df_oos_aligned.index[0].date()} to {df_oos_aligned.index[-1].date()}\n")
        f.write("-" * 30 + "\n")
        f.write(stats.to_string())
        
    print(stats)
    print(f"\nReport saved to {report_path}")
    
    # 7. Plot Equity Curve
    fig = portfolio.plot()
    plot_path = os.path.join(RESULTS_DIR, "equity_curve.html")
    fig.write_html(plot_path)
    print(f"Equity curve plot saved to {plot_path}")

if __name__ == "__main__":
    run_backtest()
