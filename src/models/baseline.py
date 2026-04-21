import pandas as pd
import numpy as np


class ActuarialBaseline:
    """
    Chain-ladder reserve estimator using median age-to-age (ATA) factors.

    Factors are pooled across all lines of business — the simplest valid
    chain-ladder implementation. This is intentional: the ML model gets
    explicit line-of-business features and can learn line-specific patterns,
    while the baseline uses pooled factors. That asymmetry makes the
    comparison more meaningful, not less.
    """

    def __init__(self):
        self.factors_ = None       # DataFrame: dev_lag, ata
        self._factor_lookup = {}   # dict: {lag: factor} for fast predict

    def fit(self, train_df: pd.DataFrame) -> "ActuarialBaseline":
        """
        Compute median ATA factors from training data.

        Parameters
        ----------
        train_df : DataFrame
            Cleaned, feature-engineered training rows (lags 1-9).
            Must contain: company, accident_year, dev_lag, incurred_loss.
        """
        temp = train_df.sort_values(["company", "accident_year", "dev_lag"]).copy()

        # ATA(lag) = incurred_loss(lag+1) / incurred_loss(lag)
        # Group by company + accident_year only — no 'line' column after get_dummies
        temp["next_loss"] = (
            temp.groupby(["company", "accident_year"])["incurred_loss"]
            .shift(-1)
        )

        # Drop lag 9 rows (no lag 10 in training since dev_lag < 10 filter)
        temp = temp.dropna(subset=["next_loss"])
        temp = temp[temp["incurred_loss"] > 0]

        temp["ata"] = temp["next_loss"] / temp["incurred_loss"]

        # Pool factors across all lines — median is robust to outlier companies
        self.factors_ = (
            temp.groupby("dev_lag")["ata"]
            .median()
            .reset_index()
        )

        # Build fast lookup: {lag: factor}
        self._factor_lookup = dict(
            zip(self.factors_["dev_lag"], self.factors_["ata"])
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Project each snapshot to ultimate by chaining ATA factors.

        For a row at dev_lag k, multiplies incurred_loss by:
            ATA(k) * ATA(k+1) * ... * ATA(9)

        Parameters
        ----------
        df : DataFrame
            Rows to predict. Must contain: dev_lag, incurred_loss.

        Returns
        -------
        pd.Series of predicted ultimate losses, aligned to df.index.
        """
        if self.factors_ is None:
            raise RuntimeError("Call fit() before predict().")

        def _chain(row):
            loss = row["incurred_loss"]
            for lag in range(int(row["dev_lag"]), 10):
                loss *= self._factor_lookup.get(lag, 1.0)
            return loss

        return df.apply(_chain, axis=1).rename("baseline_pred")

    def get_factors_table(self) -> pd.DataFrame:
        """
        Return fitted ATA factors as a tidy Series indexed by dev_lag.
        Useful for README methodology section.
        """
        if self.factors_ is None:
            raise RuntimeError("Call fit() before get_factors_table().")
        return self.factors_.set_index("dev_lag")["ata"].round(4)
