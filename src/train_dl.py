import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, TARGET_COL, EXPECTED_RETURN_COL,
    FEE_RATE, SLIPPAGE
)
from src.train_base import calculate_financial_metrics

def create_sequences(X, y, time_steps=14):
    """
    Transforms tabular data into 3D tensors for LSTM.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_classifier(input_shape):
    """LSTM model for binary classification."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_regressor(input_shape):
    """LSTM model for continuous return regression."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def run_experiment(approach_name, df, X_df, y_series, time_steps):
    """
    Runs walk-forward validation for a given DL approach.
    """
    print(f"\n{'='*50}\nRunning approach: {approach_name}\n{'='*50}")
    tscv = TimeSeriesSplit(n_splits=5)
    
    sharpe_scores = []
    mdd_scores = []
    fold = 1
    
    for train_index, test_index in tscv.split(X_df):
        X_train_df, X_test_df = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train_series, y_test_series = y_series.iloc[train_index], y_series.iloc[test_index]
        df_test = df.iloc[test_index]
        
        # Validation split chronologically
        val_size = int(len(X_train_df) * 0.2)
        train_size = len(X_train_df) - val_size
        
        X_train_sub = X_train_df.iloc[:train_size]
        y_train_sub = y_train_series.iloc[:train_size]
        X_val_sub = X_train_df.iloc[train_size:]
        y_val_sub = y_train_series.iloc[train_size:]
        
        # Scale
        scaler = StandardScaler()
        X_train_sub_scaled = scaler.fit_transform(X_train_sub)
        X_val_sub_scaled = scaler.transform(X_val_sub)
        X_test_scaled = scaler.transform(X_test_df)
        
        # Sequences
        X_train_seq, y_train_seq = create_sequences(X_train_sub_scaled, y_train_sub.values, time_steps)
        X_val_seq, y_val_seq = create_sequences(X_val_sub_scaled, y_val_sub.values, time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_series.values, time_steps)
        
        # Align test dataframe
        df_test_aligned = df_test.iloc[time_steps:]
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        
        if approach_name == "dl_regression":
            model = build_lstm_regressor((X_train_seq.shape[1], X_train_seq.shape[2]))
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping], verbose=0)
            
            y_pred_return = model.predict(X_test_seq, verbose=0).flatten()
            threshold = (FEE_RATE * 2) + SLIPPAGE
            y_pred = (y_pred_return > threshold).astype(int)
            
        elif approach_name == "dl_class_weights":
            model = build_lstm_classifier((X_train_seq.shape[1], X_train_seq.shape[2]))
            
            # Compute balanced weights
            classes = np.unique(y_train_seq)
            if len(classes) > 1:
                weights = compute_class_weight('balanced', classes=classes, y=y_train_seq)
                class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
            else:
                class_weight_dict = None
                
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_val_seq, y_val_seq), class_weight=class_weight_dict, callbacks=[early_stopping], verbose=0)
            
            y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
        elif approach_name == "dl_dyn_threshold":
            model = build_lstm_classifier((X_train_seq.shape[1], X_train_seq.shape[2]))
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping], verbose=0)
            
            y_val_prob = model.predict(X_val_seq, verbose=0).flatten()
            
            # Align validation dataframe for Sharpe calculation
            df_val_aligned = df.iloc[train_index].iloc[train_size:].iloc[time_steps:]
            
            best_thresh = 0.5
            best_sharpe = -999.0
            
            for thresh in np.arange(0.30, 0.75, 0.05):
                y_val_pred_temp = (y_val_prob > thresh).astype(int)
                _, sharpe, _ = calculate_financial_metrics(df_val_aligned, y_val_pred_temp)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_thresh = thresh
                    
            print(f"Fold {fold} - Best threshold from Val: {best_thresh:.2f}")
            y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()
            y_pred = (y_pred_prob > best_thresh).astype(int)
            
        total_ret, sharpe, mdd = calculate_financial_metrics(df_test_aligned, y_pred)
        print(f"Fold {fold} | Total Return: {total_ret*100:.2f}% | Sharpe: {sharpe:.4f} | MDD: {mdd*100:.2f}%")
        
        sharpe_scores.append(sharpe)
        mdd_scores.append(mdd)
        fold += 1
        
    avg_sharpe = np.mean(sharpe_scores)
    avg_mdd = np.mean(mdd_scores) * 100
    print(f"\n{approach_name} -> Avg Sharpe: {avg_sharpe:.4f} | Avg MDD: {avg_mdd:.2f}%")
    
    # Save metrics
    res_file = os.path.join(RESULTS_DIR, f"{approach_name}_metrics.txt")
    with open(res_file, "w") as f:
        f.write(f"=== {approach_name} Results ===\n")
        f.write(f"Average Sharpe Ratio: {avg_sharpe:.4f}\n")
        f.write(f"Average Max Drawdown: {avg_mdd:.2f}%\n")
        
    return avg_sharpe, avg_mdd, model, scaler

def run_dl_training():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    data_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run features.py first.")
        
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    cols_to_drop = [TARGET_COL, EXPECTED_RETURN_COL]
    X_df = df.drop(columns=cols_to_drop)
    
    # y for Classification approaches
    y_class = df[TARGET_COL]
    
    # y for Regression approaches
    y_reg = df[EXPECTED_RETURN_COL]
    
    time_steps = 14
    
    # 1. Classification with Class Weights
    run_experiment("dl_class_weights", df, X_df, y_class, time_steps)
    
    # 2. Classification with Dynamic Threshold (The Best Performer)
    _, _, final_model, final_scaler = run_experiment("dl_dyn_threshold", df, X_df, y_class, time_steps)
    
    # 3. Regression 
    run_experiment("dl_regression", df, X_df, y_reg, time_steps)
    
    # Save the Dynamic Threshold model as the final chosen model based on experiments
    model_path = os.path.join(MODEL_DIR, "dl_lstm_dyn_threshold.keras")
    final_model.save(model_path)
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(MODEL_DIR, "dl_lstm_scaler.joblib")
    joblib.dump(final_scaler, scaler_path)
    
    print(f"\nDynamic Threshold model and scaler successfully saved at: {MODEL_DIR}")

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    run_dl_training()
