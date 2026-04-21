import pandas as pd
import numpy as np


class ActuarialBaseline:
    """
    Chain-ladder reserve estimator using median age-to-age (ATA) factors
    computed on cumulative paid losses.

    Paid losses are used because the CAS database records net incurred losses
    inclusive of reserve releases, causing incurred to decrease over time in
    ~70% of cases. Paid losses develop upward in ~74% of cases and are the
    correct basis for a chain-ladder projection in this dataset.

    Factors are pooled across all lines. The ML model receives line dummies
    and learns line-specific patterns; the baseline uses pooled factors.
    """

    def __init__(self):
        self.factors_ = None      # DataFrame: dev_lag, ata
        self._factor_lookup = {}  # dict: {lag: factor}

    def fit(self, train_df: pd.DataFrame) -> "ActuarialBaseline":
        """
        Compute median ATA factors from paid losses in training data.

        Parameters
        ----------
        train_df : DataFrame
            Feature-engineered training rows (lags 1-9).
            Must contain: company, accident_year, dev_lag, paid_loss.
        """
        temp = train_df.sort_values(["company", "accident_year", "dev_lag"]).copy()

        temp["next_paid"] = (
            temp.groupby(["company", "accident_year"])["paid_loss"]
            .shift(-1)
        )

        temp = temp.dropna(subset=["next_paid"])
        temp = temp[temp["paid_loss"] > 0]
        temp["ata"] = temp["next_paid"] / temp["paid_loss"]

        self.factors_ = (
            temp.groupby("dev_lag")["ata"]
            .median()
            .reset_index()
        )

        self._factor_lookup = dict(
            zip(self.factors_["dev_lag"], self.factors_["ata"])
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Project each paid loss snapshot to ultimate by chaining ATA factors.

        For a row at dev_lag k, multiplies paid_loss by:
            ATA(k) * ATA(k+1) * ... * ATA(9)

        Parameters
        ----------
        df : DataFrame
            Must contain: dev_lag, paid_loss.

        Returns
        -------
        pd.Series of predicted ultimate paid losses, aligned to df.index.
        """
        if self.factors_ is None:
            raise RuntimeError("Call fit() before predict().")

        def _chain(row):
            loss = row["paid_loss"]
            for lag in range(int(row["dev_lag"]), 10):
                loss *= self._factor_lookup.get(lag, 1.0)
            return loss

        return df.apply(_chain, axis=1).rename("baseline_pred")

    def get_factors_table(self) -> pd.Series:
        """Return fitted ATA factors as a Series indexed by dev_lag."""
        if self.factors_ is None:
            raise RuntimeError("Call fit() before get_factors_table().")
        return self.factors_.set_index("dev_lag")["ata"].round(4)
