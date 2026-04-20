import pandas as pd
import pytest
from src.data.cleaner import clean

def test_loss_ratio_calculation():
    # Create a tiny "fake" dataset
    test_data = pd.DataFrame({
        "IncurredLosses": [100.0],
        "EarnedPremDIR": [200.0],
        "CumPaidLoss": [50.0]
    })
    
    # Run your cleaner on it
    result = clean(test_data)
    
    # Assert: 100 incurred / 200 premium should be exactly 0.5
    assert result["loss_ratio"].iloc[0] == 0.5

def test_remove_negatives():
    # Create data with one valid row and one negative (invalid) row
    test_data = pd.DataFrame({
        "IncurredLosses": [50.0, -10.0],
        "EarnedPremDIR": [100.0, 100.0],
        "CumPaidLoss": [20.0, 0.0]
    })
    
    result = clean(test_data)
    
    # Assert: The negative row should have been filtered out
    assert len(result) == 1
    assert result["incurred_loss"].iloc[0] == 50.0