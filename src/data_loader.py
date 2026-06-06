import os
import requests
import pandas as pd
import yfinance as yf
from config import (
    TICKER, EXOGENOUS_TICKERS, START_DATE, END_DATE, 
    INTERVAL, DATA_RAW_DIR
)

def download_yfinance_data(ticker_symbol, file_name):
    """
    Downloads historical data from Yahoo Finance and saves it as a CSV.
    """
    print(f"Downloading {ticker_symbol} data...")
    ticker_data = yf.download(
        ticker_symbol, 
        start=START_DATE, 
        end=END_DATE, 
        interval=INTERVAL,
        progress=False
    )
    
    if ticker_data.empty:
        print(f"Warning: No data found for {ticker_symbol}")
        return

    # Ensure the columns don't have a multi-index if yfinance returns one
    if isinstance(ticker_data.columns, pd.MultiIndex):
        ticker_data.columns = ticker_data.columns.droplevel(1)

    file_path = os.path.join(DATA_RAW_DIR, f"{file_name}.csv")
    ticker_data.to_csv(file_path)
    print(f"Saved {ticker_symbol} to {file_path}")

def download_fear_and_greed_index():
    """
    Downloads the Crypto Fear & Greed Index from Alternative.me.
    """
    print("Downloading Crypto Fear & Greed Index...")
    url = "https://api.alternative.me/fng/?limit=0&date_format=kr"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching Fear & Greed Index: HTTP {response.status_code}")
        return

    data = response.json()
    if 'data' not in data:
        print("Warning: Unexpected structure in Fear & Greed API response")
        return
        
    df = pd.DataFrame(data['data'])
    
    # The API returns 'timestamp' which is a formatted date string because of date_format=kr
    df['value'] = df['value'].astype(int)
    df['Date'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically and set Date as index
    df = df.sort_values(by='Date').set_index('Date')
    
    # We keep 'value' and 'value_classification'
    df = df[['value', 'value_classification']]
    
    # Filter by the configured date range
    mask = (df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))
    df = df.loc[mask]

    file_path = os.path.join(DATA_RAW_DIR, "fear_and_greed.csv")
    df.to_csv(file_path)
    print(f"Saved Fear & Greed Index to {file_path}")

def run_data_ingestion():
    """
    Main execution function for data ingestion.
    """
    # Create raw directory if it doesn't exist
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    # 1. Download primary asset (BTC)
    download_yfinance_data(TICKER, "BTC_raw")
    
    # 2. Download exogenous variables
    for ex_ticker in EXOGENOUS_TICKERS:
        # Sanitize ticker name for filename (e.g., ^VIX -> VIX)
        safe_name = ex_ticker.replace("^", "").replace(".", "_").replace("-", "_") + "_raw"
        download_yfinance_data(ex_ticker, safe_name)
        
    # 3. Download Sentiment Data
    download_fear_and_greed_index()
    
    print("Phase 1: Data Ingestion completed successfully.")

if __name__ == "__main__":
    run_data_ingestion()
