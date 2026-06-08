import os
import yaml
from datetime import datetime

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If running as a package, BASE_DIR is src/. We want the project root.
if os.path.basename(BASE_DIR) == "src":
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

# Load configuration from YAML
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# 1. Asset Configuration and Context
TICKER = config["assets"]["primary_ticker"]
EXOGENOUS_TICKERS = config["assets"]["exogenous_tickers"]
START_DATE = config["timeframe"]["start_date"]
END_DATE = datetime.today().strftime('%Y-%m-%d') # Automated current date
OOS_START_DATE = config["timeframe"]["oos_start_date"]
INTERVAL = config["timeframe"]["interval"]

# 2. Real-World Frictions
FEE_RATE = config["trading"]["fee_rate"]
SLIPPAGE = config["trading"]["slippage"]

# 3. Dynamic Paths
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 4. Model Variables
TARGET_COL = config["modeling"]["target_col"]
EXPECTED_RETURN_COL = config["modeling"]["expected_return_col"]

# 5. Data Infrastructure
DUCKDB_PATH = os.path.join(PROJECT_ROOT, config.get("infrastructure", {}).get("duckdb_path", "data/market_data.duckdb"))
PARQUET_DIR = os.path.join(PROJECT_ROOT, config.get("infrastructure", {}).get("parquet_dir", "data/processed/parquet"))
BINANCE_VISION_BASE_URL = config.get("infrastructure", {}).get("binance_vision_base_url", "https://data.binance.vision/data/futures/um/monthly")

os.makedirs(PARQUET_DIR, exist_ok=True)
