import pandas as pd
import numpy as np
import pytest
from src.models.baseline import ActuarialBaseline


def make_train():
    """
    Minimal synthetic training set: one company, one accident year,
    two lines, lags 1-10 with perfectly predictable development.

    Lags go to 10 so fit() can compute ATA(9) = loss(10)/loss(9).
    Without lag 10, there is no ATA(9) factor and the chain defaults
    to 1.0 at the last step, making lag 1 and lag 9 predictions equal.
    """
    rows = []
    for line in ["ppauto", "medmal"]:
        for lag in range(1, 11):  # 1 through 10 inclusive
            rows.append({
                "company": "Test Co",
                "accident_year": 2000,
                "line": line,
                "dev_lag": lag,
                "incurred_loss": float(lag * 1000),
            })
    return pd.DataFrame(rows)


def make_test(lag: int, line: str = "ppauto") -> pd.DataFrame:
    """
    Single test row at a given lag.

    incurred_loss is fixed at 1000 for all lags so tests compare
    predictions from the same starting base — not from different
    incurred amounts that would confound the assertion.
    """
    return pd.DataFrame([{
        "company": "Test Co",
        "accident_year": 2001,
        "line": line,
        "dev_lag": lag,
        "incurred_loss": 1000.0,
    }])


class TestActuarialBaseline:

    def test_fit_produces_factors(self):
        model = ActuarialBaseline().fit(make_train())
        assert model.factors_ is not None
        assert set(model.factors_.columns) == {"line", "dev_lag", "ata"}

    def test_ata_factors_correct(self):
        """
        With perfectly linear development (lag * 1000), ATA factor at
        every lag should be exactly (lag+1)/lag.
        ATA(1) = 2000/1000 = 2.0
        ATA(8) = 9000/8000 = 1.125
        ATA(9) = 10000/9000 ≈ 1.111
        """
        model = ActuarialBaseline().fit(make_train())
        factors = model.get_factors_table()
        assert abs(factors.loc["ppauto", 1] - 2.0) < 1e-4
        assert abs(factors.loc["ppauto", 8] - 9000 / 8000) < 1e-4
        assert abs(factors.loc["ppauto", 9] - 10000 / 9000) < 1e-4

    def test_lag1_prediction_greater_than_lag9(self):
        """
        Core correctness test: starting from the same incurred_loss,
        a lag 1 snapshot should predict a much higher ultimate than
        a lag 9 snapshot. Lag 1 chains 9 factors; lag 9 chains 1.
        """
        model = ActuarialBaseline().fit(make_train())
        pred_lag1 = model.predict(make_test(lag=1)).iloc[0]
        pred_lag9 = model.predict(make_test(lag=9)).iloc[0]
        assert pred_lag1 > pred_lag9, (
            f"Lag 1 pred ({pred_lag1:.1f}) should exceed lag 9 pred ({pred_lag9:.1f})"
        )

    def test_lag1_chains_to_correct_ultimate(self):
        """
        With linear development, chaining all factors from lag 1 should
        yield exactly 10 * incurred_loss (since ultimate = lag 10 loss
        = 10 * lag 1 loss in our synthetic data).
        """
        model = ActuarialBaseline().fit(make_train())
        pred = model.predict(make_test(lag=1)).iloc[0]
        # 1000 * ATA(1)*ATA(2)*...*ATA(9) = 1000 * (10000/1000) = 10000
        assert abs(pred - 10000.0) < 1e-3, f"Expected 10000.0, got {pred:.3f}"

    def test_lag9_prediction_close_to_incurred(self):
        """
        At lag 9, only ATA(9) = 10/9 ≈ 1.111 remains.
        Prediction should be incurred * 1.111, not a large multiple.
        """
        model = ActuarialBaseline().fit(make_train())
        pred = model.predict(make_test(lag=9)).iloc[0]
        expected = 1000.0 * (10000 / 9000)
        assert abs(pred - expected) < 1e-3, f"Expected {expected:.3f}, got {pred:.3f}"

    def test_predict_before_fit_raises(self):
        model = ActuarialBaseline()
        with pytest.raises(RuntimeError):
            model.predict(make_test(lag=5))

    def test_missing_line_defaults_to_no_development(self):
        """
        If a line wasn't in training, factors default to 1.0 and
        prediction equals incurred_loss unchanged.
        """
        model = ActuarialBaseline().fit(make_train())
        test_row = pd.DataFrame([{
            "company": "Unknown Co",
            "accident_year": 2001,
            "line": "unknown_line",
            "dev_lag": 5,
            "incurred_loss": 5000.0,
        }])
        pred = model.predict(test_row).iloc[0]
        assert pred == 5000.0

    def test_output_aligned_to_index(self):
        """predict() must return a Series aligned to the input DataFrame index."""
        model = ActuarialBaseline().fit(make_train())
        test_df = make_test(lag=3)
        test_df.index = [42]
        result = model.predict(test_df)
        assert result.index.tolist() == [42]

    def test_both_lines_produce_same_factors(self):
        """Both lines use identical synthetic data, so factors should match."""
        model = ActuarialBaseline().fit(make_train())
        factors = model.get_factors_table()
        for lag in range(1, 10):
            assert abs(factors.loc["ppauto", lag] - factors.loc["medmal", lag]) < 1e-6
