import pandas as pd
import numpy as np


def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms actuarial snapshots into a supervised learning format.

    Each row represents a claim snapshot at a given development lag (1-9).
    The target is the ultimate incurred loss at lag 10 for that
    company / accident_year / line combination.
    """
    # ------------------------------------------------------------------ #
    # 1. Identify the target: incurred loss at full development (lag 10)  #
    # ------------------------------------------------------------------ #
    ultimates = (
        df[df["dev_lag"] == 10][["company", "accident_year", "line", "incurred_loss"]]
        .rename(columns={"incurred_loss": "target_ultimate"})
    )

    # ------------------------------------------------------------------ #
    # 2. Merge target back onto early-development rows                    #
    # ------------------------------------------------------------------ #
    ml_df = df.merge(ultimates, on=["company", "accident_year", "line"])

    # ------------------------------------------------------------------ #
    # 3. Core actuarial features                                          #
    # ------------------------------------------------------------------ #

    # Case reserve: insurer's estimate of remaining unpaid losses
    ml_df["case_reserve"] = ml_df["incurred_loss"] - ml_df["paid_loss"]

    # Clip negatives — paid occasionally exceeds incurred due to recoveries
    # or data artifacts; clamp to zero so the feature stays interpretable
    ml_df["case_reserve"] = ml_df["case_reserve"].clip(lower=0)

    # Paid ratio: maturity signal — how much of incurred has been paid out?
    # Replace zero incurred with 1 to avoid NaN; paid_ratio correctly becomes 0
    ml_df["paid_ratio"] = ml_df["paid_loss"] / ml_df["incurred_loss"].replace(0, 1)

    # Cap at 2.0 — ratios above this are data artifacts (145 rows in full dataset)
    ml_df["paid_ratio"] = ml_df["paid_ratio"].clip(0, 2.0)

    # ------------------------------------------------------------------ #
    # 4. Development maturity feature                                     #
    # ------------------------------------------------------------------ #

    # Normalised lag: 0.11 at lag 1, 1.0 at lag 9
    # Gives the model a continuous maturity signal independent of loss size
    ml_df["maturity_pct"] = ml_df["dev_lag"] / 9.0

    # ------------------------------------------------------------------ #
    # 5. Log transformations                                              #
    # ------------------------------------------------------------------ #

    # Insurance losses are heavy-tailed; log scale helps the model focus
    # on relative patterns rather than being dominated by large outliers
    ml_df["log_incurred"] = np.log1p(ml_df["incurred_loss"])
    ml_df["log_target"] = np.log1p(ml_df["target_ultimate"])

    # ------------------------------------------------------------------ #
    # 6. Line-of-business dummies                                         #
    # ------------------------------------------------------------------ #

    # One-hot encode the six lines; drop_first=False keeps all dummies so
    # each line has an explicit coefficient — easier to interpret per-line
    ml_df = pd.get_dummies(ml_df, columns=["line"], drop_first=False)

    # ------------------------------------------------------------------ #
    # 7. Filter to training lags only                                     #
    # ------------------------------------------------------------------ #

    # We train on lags 1-9 (observable snapshots) and predict lag 10
    # (ultimate). Keeping lag 10 rows would be data leakage.
    return ml_df[ml_df["dev_lag"] < 10].reset_index(drop=True)


# Feature columns used by the model — import this list in your notebook
# so the feature set stays in sync between training and evaluation
X_COLS = [
    "dev_lag",
    "maturity_pct",
    "incurred_loss",
    "paid_loss",
    "case_reserve",
    "paid_ratio",
    "log_incurred",
    "line_comauto",
    "line_medmal",
    "line_othliab",
    "line_ppauto",
    "line_prodliab",
    "line_wkcomp",
]
