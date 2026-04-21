# ML-Powered Loss Reserve Estimator

Comparing machine learning against the chain-ladder method for insurance loss reserving — a core actuarial problem that has relied on the same deterministic approach for decades.

**Result:** XGBoost reduced reserve estimation error by 48.9% vs chain-ladder ($198,304 vs $387,770 RMSE) on held-out accident years 2003–2007, evaluated on a paid loss basis.

---

## The problem

Property & casualty insurers must estimate their ultimate claim liabilities — losses that have been incurred but not yet fully paid. The industry standard is the **chain-ladder method**: compute historical age-to-age (ATA) development factors and project each accident year's losses to ultimate by chaining those factors forward.

Chain-ladder is transparent and auditable, but it has known weaknesses. It relies on median factors that can be skewed by outlier years, assumes development patterns are stable over time, and cannot incorporate signals like paid-to-incurred ratios or reserve maturity that an experienced actuary would use qualitatively.

This project asks: can a gradient boosted tree model learn those signals and produce more accurate reserve estimates?

---

## Data

**Source:** [CAS Loss Reserve Database](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p) — publicly available from the Casualty Actuarial Society.

Six lines of business: private passenger auto, commercial auto, workers compensation, medical malpractice, other liability, and product liability. Each contains schedule P triangle data for accident years 1998–2007 across hundreds of US insurers.

- 70,894 rows after cleaning
- 768 unique company / accident year combinations
- Train: accident years 1998–2002 (33,567 rows)
- Test: accident years 2003–2007 (30,719 rows) — never seen during training

**Why paid basis:** The CAS database records net incurred losses inclusive of reserve releases, causing incurred losses to decrease from lag 1 to lag 10 in ~70% of cases. Paid losses develop upward in ~74% of cases and show large development factors (e.g. medmal: 68× from lag 1 to ultimate). Both the ML model and chain-ladder baseline are therefore evaluated on a paid loss basis for a consistent, meaningful comparison.

---

## Methodology

### Chain-ladder baseline

For each development lag, the median ATA factor is computed across all companies in the training set on a paid loss basis. To predict ultimate paid losses for a snapshot at lag *k*, those factors are chained from lag *k* through lag 9:

```
ultimate_paid = paid_loss × ATA(k) × ATA(k+1) × ... × ATA(9)
```

Median factors are used rather than volume-weighted averages to be robust against outlier companies. Factors are pooled across lines of business — the simplest valid implementation and the correct baseline to benchmark against.

**A dataset limitation worth noting:** the CAS data covers only 10 years of development (1998–2007). For companies in the training set (1998–2002), paid losses between consecutive lags are largely stable by the time the data was recorded — producing ATA factors near 1.0 for most lags. This limits the chain-ladder's predictive power on this particular dataset. In a real reserving context, the baseline would be fit on a richer historical triangle. The comparison remains valid: both models see identical training data and are evaluated on the same held-out test set.

### Feature engineering

Each row represents a claim snapshot at a given development lag (1–9). The target is the ultimate paid loss at lag 10 for that company / accident year / line combination.

| Feature | Description |
|---|---|
| `dev_lag` | Development year (1–9) |
| `maturity_pct` | `dev_lag / 9.0` — normalized maturity signal |
| `incurred_loss` | Cumulative incurred loss at snapshot date |
| `paid_loss` | Cumulative paid loss at snapshot date |
| `case_reserve` | `incurred - paid`, clipped to 0 |
| `paid_ratio` | `paid / incurred`, clipped to 2.0 |
| `log_paid` | Log-transformed paid loss — stabilizes heavy tail |
| `log_incurred` | Log-transformed incurred loss |
| `line_*` | One-hot encoded line of business (6 dummies) |

The target (`log_target = log1p(target_ultimate)`) is log-transformed during training to prevent large losses from dominating the loss function. Predictions are exponentiated back to dollars for evaluation.

### XGBoost model

```python
XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

Depth-4 trees prevent memorizing company-specific noise. Validation loss stabilized around round 150–200 with no meaningful overfitting through round 300.

---

## Results

### Overall

| Model | RMSE | vs baseline |
|---|---|---|
| Chain-ladder | $387,770 | — |
| XGBoost | $198,304 | −48.9% |

### By line of business

| Line | Chain-ladder RMSE | XGBoost RMSE | Improvement |
|---|---|---|---|
| comauto | $28,079 | $4,219 | −85.0% |
| medmal | $42,214 | $15,993 | −62.1% |
| othliab | $23,772 | $33,648 | +41.5% ⚠ |
| ppauto | $905,262 | $461,364 | −49.0% |
| prodliab | $8,432 | $2,200 | −73.9% |
| wkcomp | $40,740 | $22,361 | −45.1% |

**Notable finding:** `othliab` (other liability) is the one line where chain-ladder outperforms XGBoost. Other liability has the most heterogeneous claim mix — spanning general, environmental, and professional liability — which may require line-specific feature engineering or a longer training window. This is an honest limitation and a direction for future work.

`ppauto`'s high absolute RMSE reflects its large exposure base and individual claim sizes; the 49.0% percentage improvement is consistent with the other lines.

### Feature importance

`incurred_loss` and `log_incurred` are the dominant signals (combined ~70% of importance), followed by `paid_loss` and `log_paid`. Actuarial ratio features (`paid_ratio`, `case_reserve`, `maturity_pct`) contribute less than raw loss amounts, suggesting the model primarily learns from loss magnitude and scale rather than maturity signals — a direction for future feature engineering.

---

## Project structure

```
loss-reserve-ml/
├── data/
│   ├── raw/          # CAS CSVs — download from casact.org (gitignored)
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   └── 02_training.ipynb  # Model training and benchmarking
├── src/
│   ├── data/
│   │   ├── loader.py    # Ingestion
│   │   ├── cleaner.py   # Cleaning and derived columns
│   │   └── features.py  # Feature engineering + X_COLS definition
│   └── models/
│       └── baseline.py  # Chain-ladder ActuarialBaseline class
├── tests/
│   ├── test_cleaner.py
│   └── test_baseline.py
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/yourusername/loss-reserve-ml.git
cd loss-reserve-ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download the six CAS schedule P CSVs from [casact.org/research/reserve-data](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p) and place them in `data/raw/`.

```bash
pytest tests/        # run tests
jupyter notebook     # open notebooks/
```

---

## Skills demonstrated

- **Data engineering:** multi-file ingestion pipeline, schema normalization, reproducible cleaning
- **Actuarial domain knowledge:** chain-ladder implementation, ATA factor analysis, reserve triangle interpretation, paid vs incurred basis selection
- **Machine learning:** gradient boosting, log-target transformation, chronological train/test split, overfitting diagnostics
- **Software engineering:** modular src layout, unit tests with pytest, typed function signatures, documented limitations
