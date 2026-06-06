import os
import sys
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

# Adjust path to handle direct script execution vs module execution
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, TARGET_COL, EXPECTED_RETURN_COL,
    FEE_RATE, SLIPPAGE
)

def add_technical_indicators(df):
    """
    Adds RSI, MACD, Moving Averages and Bollinger Bands to the dataframe.
    """
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Moving Averages
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    return df

def run_feature_engineering():
    """
    Executes Phase 2: Feature Engineering and Realistic Target creation.
    """
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # 1. Load Primary Asset
    btc_path = os.path.join(DATA_RAW_DIR, "BTC_raw.csv")
    if not os.path.exists(btc_path):
        raise FileNotFoundError(f"{btc_path} not found. Run data_loader.py first.")
    
    df_btc = pd.read_csv(btc_path, index_col='Date', parse_dates=True)
    
    # Add Technical Indicators
    df_btc = add_technical_indicators(df_btc)
    
    # 2. Merge Exogenous Variables
    # Load VIX
    vix_path = os.path.join(DATA_RAW_DIR, "VIX_raw.csv")
    if os.path.exists(vix_path):
        df_vix = pd.read_csv(vix_path, index_col='Date', parse_dates=True)
        # We only keep Close to avoid column collision
        df_btc['VIX_Close'] = df_vix['Close']
    else:
        print("Warning: VIX data not found.")

    # Load DXY
    dxy_path = os.path.join(DATA_RAW_DIR, "DX_Y_NYB_raw.csv")
    if os.path.exists(dxy_path):
        df_dxy = pd.read_csv(dxy_path, index_col='Date', parse_dates=True)
        df_btc['DXY_Close'] = df_dxy['Close']
    else:
        print("Warning: DXY data not found.")

    # Load Fear & Greed
    fng_path = os.path.join(DATA_RAW_DIR, "fear_and_greed.csv")
    if os.path.exists(fng_path):
        df_fng = pd.read_csv(fng_path, index_col='Date', parse_dates=True)
        df_btc['Fear_Greed_Value'] = df_fng['value']
    else:
        print("Warning: Fear & Greed data not found.")
        
    # Forward fill exogen variables since traditional markets might close on weekends/holidays 
    # while Crypto is 24/7. We ffill to propagate the last known value.
    # Fear and greed might also miss some days.
    cols_to_ffill = [col for col in ['VIX_Close', 'DXY_Close', 'Fear_Greed_Value'] if col in df_btc.columns]
    df_btc[cols_to_ffill] = df_btc[cols_to_ffill].ffill()

    # 3. Create Realistic Target
    # We shift close by -1 to get "Tomorrow's Close"
    df_btc['Close_Tomorrow'] = df_btc['Close'].shift(-1)
    
    # Calculate expected return
    expected_return = (df_btc['Close_Tomorrow'] / df_btc['Close']) - 1.0
    
    # Threshold = (FEE_RATE * 2) + SLIPPAGE (Entry fee, Exit fee, Slippage)
    threshold = (FEE_RATE * 2) + SLIPPAGE
    
    # Realistic Target: 1 if return > threshold, else 0
    df_btc[TARGET_COL] = (expected_return > threshold).astype(int)
    
    # Save the continuous return for Regression approach
    df_btc[EXPECTED_RETURN_COL] = expected_return
    
    # 4. Clean up - CRITICAL: Data Leakage Prevention!
    # Drop the shifted column 'Close_Tomorrow' so the model cannot see the future
    df_btc = df_btc.drop(columns=['Close_Tomorrow'])
    
    # Drop rows with NaN (due to indicators lag, e.g. SMA_50 generates 50 NaNs at the beginning,
    # and shifting Target generates 1 NaN at the end)
    df_btc = df_btc.dropna()
    
    # Save processed dataframe
    output_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    df_btc.to_csv(output_path)
    
    print(f"Phase 2: Feature Engineering completed successfully.")
    print(f"Dataset shape: {df_btc.shape}")
    print(f"Target distribution:\n{df_btc[TARGET_COL].value_counts(normalize=True)}")
    print(f"Saved processed features to {output_path}")

if __name__ == "__main__":
    run_feature_engineering()
