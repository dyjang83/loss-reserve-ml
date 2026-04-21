import pandas as pd

RENAME_MAP = {
    "GRNAME": "company",
    "AccidentYear": "accident_year",
    "DevelopmentLag": "dev_lag",
    "IncurredLosses": "incurred_loss",
    "CumPaidLoss": "paid_loss",
    "EarnedPremDIR": "earned_premium",
}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    
    # Drop rows missing vital info
    df = df.dropna(subset=["incurred_loss", "paid_loss", "earned_premium"])
    
    # Remove negatives
    df = df[(df["incurred_loss"] >= 0) & (df["paid_loss"] >= 0)]
    
    # Actuarial Math (Engineering new columns)
    df["loss_ratio"] = df["incurred_loss"] / df["earned_premium"].replace(0, float("nan"))
    df["paid_to_incurred"] = df["paid_loss"] / df["incurred_loss"].replace(0, float("nan"))
    
    return df.reset_index(drop=True)