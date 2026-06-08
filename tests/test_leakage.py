import os
import sys
import pandas as pd
import pytest
from src.config import DATA_PROCESSED_DIR, TARGET_COL, EXPECTED_RETURN_COL

def test_no_data_leakage():
    """
    Ensures that the 'Close_Tomorrow' or any shifted column used to calculate
    the target is NOT present in the final features file.
    """
    file_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    if not os.path.exists(file_path):
        pytest.skip(f"Processed file {file_path} not found. Skipping test.")
        
    df = pd.read_csv(file_path)
    
    # Check if 'Close_Tomorrow' is accidentally left in the dataframe
    assert 'Close_Tomorrow' not in df.columns, "Data Leakage Alert: 'Close_Tomorrow' is present in the features dataset!"
    
    # Check that there are no NaNs which might indicate improper shifting
    assert df.isnull().sum().sum() == 0, "Dataset contains NaN values, check the shifting or indicator logic."

def test_target_calculation():
    """
    Verifies that the TARGET_COL only contains 0 or 1.
    """
    file_path = os.path.join(DATA_PROCESSED_DIR, "final_features.csv")
    if not os.path.exists(file_path):
        pytest.skip(f"Processed file {file_path} not found. Skipping test.")
        
    df = pd.read_csv(file_path)
    
    assert set(df[TARGET_COL].unique()).issubset({0, 1}), f"{TARGET_COL} must be strictly binary (0 or 1)."
