import os
from datetime import datetime

# 1. Asset Configuration and Context
TICKER = "BTC-USD"
EXOGENOUS_TICKERS = ["^VIX", "DX-Y.NYB"] # S&P500 VIX and Dollar Index (DXY)
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d') # Automated current date
OOS_START_DATE = "2024-01-01" # Out of sample start date for backtesting
INTERVAL = "1d"

# 2. Real-World Frictions
FEE_RATE = 0.0015  # Exchange fee (e.g., 0.15% per trade)
SLIPPAGE = 0.0005  # Expected price slippage

# 3. Dynamic Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 4. Model Variables
TARGET_COL = "Realistic_Target"
EXPECTED_RETURN_COL = "Expected_Return"
