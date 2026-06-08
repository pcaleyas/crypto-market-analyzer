import os
import requests
import pandas as pd
import yfinance as yf
import duckdb

# Use absolute import so it works from anywhere
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import (
    TICKER, EXOGENOUS_TICKERS, START_DATE, END_DATE, 
    INTERVAL, DATA_RAW_DIR, DUCKDB_PATH, PARQUET_DIR
)
from src.logger import get_logger
from src.binance_vision import sync_historical_data

logger = get_logger(__name__)

def download_yfinance_data(ticker_symbol, file_name):
    """
    Downloads historical data from Yahoo Finance and saves it as a CSV.
    """
    logger.info(f"Downloading {ticker_symbol} data...")
    ticker_data = yf.download(
        ticker_symbol, 
        start=START_DATE, 
        end=END_DATE, 
        interval=INTERVAL,
        progress=False
    )
    
    if ticker_data.empty:
        logger.warning(f"No data found for {ticker_symbol}")
        return

    if isinstance(ticker_data.columns, pd.MultiIndex):
        ticker_data.columns = ticker_data.columns.droplevel(1)

    file_path = os.path.join(DATA_RAW_DIR, f"{file_name}.csv")
    ticker_data.to_csv(file_path)
    logger.info(f"Saved {ticker_symbol} to {file_path}")

def download_fear_and_greed_index():
    """
    Downloads the Crypto Fear & Greed Index from Alternative.me.
    """
    logger.info("Downloading Crypto Fear & Greed Index...")
    url = "https://api.alternative.me/fng/?limit=0&date_format=kr"
    response = requests.get(url)
    
    if response.status_code != 200:
        logger.error(f"Error fetching Fear & Greed Index: HTTP {response.status_code}")
        return

    data = response.json()
    if 'data' not in data:
        logger.warning("Unexpected structure in Fear & Greed API response")
        return
        
    df = pd.DataFrame(data['data'])
    df['value'] = df['value'].astype(int)
    df['Date'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values(by='Date').set_index('Date')
    df = df[['value', 'value_classification']]
    
    mask = (df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))
    df = df.loc[mask]

    file_path = os.path.join(DATA_RAW_DIR, "fear_and_greed.csv")
    df.to_csv(file_path)
    logger.info(f"Saved Fear & Greed Index to {file_path}")

def setup_duckdb_views():
    """
    Connects to DuckDB and sets up views over the parquet files.
    """
    logger.info("Setting up DuckDB views over Parquet files...")
    
    # Connect to DuckDB (read_write mode)
    con = duckdb.connect(database=DUCKDB_PATH, read_only=False)
    
    # Klines View
    klines_path = os.path.join(PARQUET_DIR, "klines", "*.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW historical_klines AS 
        SELECT Date, open, high, low, close, volume 
        FROM '{klines_path}'
    """)
    
    # Funding Rate View
    funding_path = os.path.join(PARQUET_DIR, "fundingRate", "*.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW historical_funding AS 
        SELECT Date, funding_rate 
        FROM '{funding_path}'
    """)
    
    # Open Interest View
    oi_path = os.path.join(PARQUET_DIR, "openInterestHist", "*.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW historical_oi AS 
        SELECT Date, sum_open_interest, sum_open_interest_value 
        FROM '{oi_path}'
    """)
    
    con.close()
    logger.info("DuckDB views created successfully.")

def run_data_ingestion():
    """
    Main execution function for data ingestion.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    # 1. Download Binance Vision historical data and save as Parquet
    logger.info("Starting Binance Vision historical data sync...")
    sync_historical_data()
    
    # 2. Setup DuckDB Views
    setup_duckdb_views()
    
    # 3. Download exogenous variables
    for ex_ticker in EXOGENOUS_TICKERS:
        safe_name = ex_ticker.replace("^", "").replace(".", "_").replace("-", "_") + "_raw"
        download_yfinance_data(ex_ticker, safe_name)
        
    # 4. Download Sentiment Data
    download_fear_and_greed_index()
    
    logger.info("Phase 1: Data Ingestion completed successfully.")

if __name__ == "__main__":
    run_data_ingestion()
