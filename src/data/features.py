import pandas as pd
import numpy as np


def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms actuarial snapshots into a supervised learning format.

    Each row is a claim snapshot at development lag 1-9. The target is
    ultimate paid loss at lag 10 for that company / accident_year / line.

    Why paid basis:
    The CAS database records net incurred losses inclusive of reserve
    releases, causing incurred losses to *decrease* from lag 1 to lag 10
    in ~70% of cases. Paid losses develop upward in ~74% of cases and
    show large development factors, making paid the correct and consistent
    basis for both the ML model and the chain-ladder benchmark.
    """

    # ------------------------------------------------------------------ #
    # 1. Identify target: paid loss at full development (lag 10)         #
    # ------------------------------------------------------------------ #

    ultimates = (
        df[df["dev_lag"] == 10][["company", "accident_year", "line", "paid_loss"]]
        .rename(columns={"paid_loss": "target_ultimate"})
    )

    # ------------------------------------------------------------------ #
    # 2. Merge target onto early-development rows                        #
    # ------------------------------------------------------------------ #

    ml_df = df.merge(ultimates, on=["company", "accident_year", "line"])

    # ------------------------------------------------------------------ #
    # 3. Core actuarial features                                         #
    # ------------------------------------------------------------------ #

    # Case reserve: insurer's estimate of remaining unpaid losses
    ml_df["case_reserve"] = ml_df["incurred_loss"] - ml_df["paid_loss"]

    # Clip negatives — paid occasionally exceeds incurred due to recoveries
    ml_df["case_reserve"] = ml_df["case_reserve"].clip(lower=0)

    # Paid ratio: maturity signal — how much has been paid relative to incurred?
    # Replace zero incurred with 1 to avoid NaN; paid_ratio correctly becomes 0
    ml_df["paid_ratio"] = ml_df["paid_loss"] / ml_df["incurred_loss"].replace(0, 1)

    # Cap at 2.0 — ratios above this are data artifacts
    ml_df["paid_ratio"] = ml_df["paid_ratio"].clip(0, 2.0)

    # ------------------------------------------------------------------ #
    # 4. Development maturity                                            #
    # ------------------------------------------------------------------ #

    # Normalised lag: 0.11 at lag 1, 1.0 at lag 9
    ml_df["maturity_pct"] = ml_df["dev_lag"] / 9.0

    # ------------------------------------------------------------------ #
    # 5. Log transformations                                             #
    # ------------------------------------------------------------------ #

    # Paid losses are heavy-tailed; log scale stabilises training
    ml_df["log_paid"] = np.log1p(ml_df["paid_loss"])
    ml_df["log_incurred"] = np.log1p(ml_df["incurred_loss"])
    ml_df["log_target"] = np.log1p(ml_df["target_ultimate"])

    # ------------------------------------------------------------------ #
    # 6. Line-of-business dummies                                        #
    # ------------------------------------------------------------------ #

    ml_df = pd.get_dummies(ml_df, columns=["line"], drop_first=False)

    # ------------------------------------------------------------------ #
    # 7. Filter to training lags only (no leakage)                      #
    # ------------------------------------------------------------------ #

    return ml_df[ml_df["dev_lag"] < 10].reset_index(drop=True)


# Feature columns used by the ML model
X_COLS = [
    "dev_lag",
    "maturity_pct",
    "incurred_loss",
    "paid_loss",
    "case_reserve",
    "paid_ratio",
    "log_paid",
    "log_incurred",
    "line_comauto",
    "line_medmal",
    "line_othliab",
    "line_ppauto",
    "line_prodliab",
    "line_wkcomp",
]
