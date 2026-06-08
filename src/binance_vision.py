import os
import io
import time
import zipfile
import requests
import polars as pl
from datetime import datetime, timedelta
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import TICKER, PARQUET_DIR, BINANCE_VISION_BASE_URL, START_DATE, END_DATE
from src.logger import get_logger

logger = get_logger(__name__)

# Constants for Binance Vision
INTERVAL = "1d"
BINANCE_VISION_DAILY_URL = "https://data.binance.vision/data/futures/um/daily"

def download_and_extract_zip(url):
    """Downloads a zip file from memory and extracts the first CSV found."""
    response = requests.get(url)
    if response.status_code == 200:
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_name in z.namelist():
                    if file_name.endswith('.csv'):
                        return z.read(file_name)
        except zipfile.BadZipFile:
            logger.error(f"Bad zip file at {url}")
    return None

def sync_klines(start_year, end_year, end_month):
    """Sync monthly klines data."""
    data_type = "klines"
    type_dir = os.path.join(PARQUET_DIR, data_type)
    os.makedirs(type_dir, exist_ok=True)
    
    csv_columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]

    for year in range(start_year, end_year + 1):
        for m in range(1, 13):
            if year == end_year and m >= end_month:
                break
                
            month_str = f"{m:02d}"
            url = f"{BINANCE_VISION_BASE_URL}/klines/{TICKER}/{INTERVAL}/{TICKER}-{INTERVAL}-{year}-{month_str}.zip"
            parquet_path = os.path.join(type_dir, f"{TICKER}_{year}_{month_str}.parquet")
            
            if os.path.exists(parquet_path):
                continue
                
            logger.info(f"Downloading {data_type} for {year}-{month_str}...")
            csv_bytes = download_and_extract_zip(url)
            if csv_bytes:
                df = pl.read_csv(csv_bytes, has_header=False, new_columns=csv_columns, truncate_ragged_lines=True, infer_schema_length=0)
                # Filter out the header row if it exists (some months have it, some don't)
                df = df.filter(pl.col("open_time") != "open_time")
                
                # Cast string columns to appropriate types before doing math
                df = df.with_columns(
                    pl.col("open_time").cast(pl.Int64),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64)
                )
                
                df = df.with_columns(pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("Date"))
                df.write_parquet(parquet_path)
                logger.info(f"Saved {data_type} {year}-{month_str} to Parquet.")
            else:
                logger.debug(f"Data not found for {url}")
            time.sleep(0.3)

def sync_funding_rate(start_year, end_year, end_month):
    """Sync monthly funding rate data."""
    data_type = "fundingRate"
    type_dir = os.path.join(PARQUET_DIR, data_type)
    os.makedirs(type_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        for m in range(1, 13):
            if year == end_year and m >= end_month:
                break
                
            month_str = f"{m:02d}"
            url = f"{BINANCE_VISION_BASE_URL}/fundingRate/{TICKER}/{TICKER}-fundingRate-{year}-{month_str}.zip"
            parquet_path = os.path.join(type_dir, f"{TICKER}_{year}_{month_str}.parquet")
            
            if os.path.exists(parquet_path):
                continue
                
            logger.info(f"Downloading {data_type} for {year}-{month_str}...")
            csv_bytes = download_and_extract_zip(url)
            if csv_bytes:
                # Funding rate HAS headers: calc_time, funding_interval_hours, last_funding_rate
                df = pl.read_csv(csv_bytes, has_header=True)
                # Ensure columns are renamed appropriately to match expected format
                if "last_funding_rate" in df.columns:
                    df = df.rename({"last_funding_rate": "funding_rate"})
                df = df.with_columns(pl.from_epoch(pl.col("calc_time"), time_unit="ms").alias("Date"))
                df.write_parquet(parquet_path)
                logger.info(f"Saved {data_type} {year}-{month_str} to Parquet.")
            else:
                logger.debug(f"Data not found for {url}")
            time.sleep(0.3)

def sync_metrics():
    """Sync daily metrics data (which contains open interest)."""
    data_type = "openInterestHist"  # Keep dir name the same for duckdb views
    type_dir = os.path.join(PARQUET_DIR, data_type)
    os.makedirs(type_dir, exist_ok=True)

    # Start from 2020-01-01 or START_DATE
    current_date = datetime.strptime(max("2020-01-01", START_DATE), "%Y-%m-%d")
    end_date = datetime.now() - timedelta(days=1)  # Only up to yesterday

    # We will aggregate daily parquets into monthly to avoid too many small files
    # We collect all days in a month, then save
    
    current_month_df = None
    current_month_str = ""
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        month_str = current_date.strftime("%Y_%m")
        
        # If we crossed into a new month, save the previous month's data
        if current_month_str and month_str != current_month_str:
            if current_month_df is not None and len(current_month_df) > 0:
                parquet_path = os.path.join(type_dir, f"{TICKER}_{current_month_str}.parquet")
                current_month_df.write_parquet(parquet_path)
                logger.info(f"Saved aggregated metrics {current_month_str} to Parquet.")
            current_month_df = None
            
        current_month_str = month_str
        parquet_path = os.path.join(type_dir, f"{TICKER}_{current_month_str}.parquet")
        
        # If the full monthly parquet already exists, skip to the next month
        if os.path.exists(parquet_path) and current_date.month != end_date.month:
            # Fast forward to next month
            next_month = current_date.replace(day=28) + timedelta(days=4)
            current_date = next_month.replace(day=1)
            continue

        url = f"{BINANCE_VISION_DAILY_URL}/metrics/{TICKER}/{TICKER}-metrics-{date_str}.zip"
        
        # logger.info(f"Downloading metrics for {date_str}...") # Muted to avoid spam
        csv_bytes = download_and_extract_zip(url)
        if csv_bytes:
            # metrics HAS headers: create_time, symbol, sum_open_interest, sum_open_interest_value...
            df = pl.read_csv(csv_bytes, has_header=True, infer_schema_length=0)
            
            # Parse create_time which is string "YYYY-MM-DD HH:MM:SS"
            # and cast only the columns we need to Float64, avoiding schema issues with unused columns
            df = df.select([
                pl.col("create_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("Date"),
                pl.col("sum_open_interest").cast(pl.Float64, strict=False),
                pl.col("sum_open_interest_value").cast(pl.Float64, strict=False)
            ])
            
            if current_month_df is None:
                current_month_df = df
            else:
                current_month_df = pl.concat([current_month_df, df], how="vertical")
                
        time.sleep(0.1)
        current_date += timedelta(days=1)

    # Save any remaining data for the last month
    if current_month_df is not None and len(current_month_df) > 0:
        parquet_path = os.path.join(type_dir, f"{TICKER}_{current_month_str}.parquet")
        current_month_df.write_parquet(parquet_path)
        logger.info(f"Saved aggregated metrics {current_month_str} to Parquet.")

def sync_historical_data():
    """Main function to sync all historical data."""
    start_year = int(START_DATE.split("-")[0])
    current_date = datetime.now()
    end_year = current_date.year
    end_month = current_date.month

    logger.info("--- Starting sync for klines ---")
    sync_klines(start_year, end_year, end_month)
    
    logger.info("--- Starting sync for fundingRate ---")
    sync_funding_rate(start_year, end_year, end_month)
    
    logger.info("--- Starting sync for daily metrics (Open Interest) ---")
    sync_metrics()

if __name__ == "__main__":
    sync_historical_data()
