# ML-Powered Loss Reserve Estimator

Comparing machine learning against the chain-ladder method for insurance loss reserving — a core actuarial problem that has relied on the same deterministic approach for decades.

**Result:** XGBoost reduced reserve estimation error by 49.4% vs chain-ladder ($197,402 vs $390,109 RMSE) on held-out accident years 2003–2007.

---

## The problem

Property & casualty insurers must estimate their ultimate claim liabilities — losses that have been incurred but not yet fully paid. The industry standard is the chain-ladder method: compute historical age-to-age development factors and project each accident year's losses to ultimate by chaining those factors forward.

Chain-ladder is transparent and auditable, but it has known weaknesses. It relies on average factors that get skewed by outlier years, assumes development patterns are stable across lines of business, and cannot incorporate signals like paid-to-incurred ratios or reserve maturity that an experienced actuary would use qualitatively.

This project asks: can a gradient boosted tree model learn those signals and produce more accurate reserve estimates?

---

## Data

**Source:** [CAS Loss Reserve Database](https://www.casact.org/research/reserve-data) — publicly available from the Casualty Actuarial Society.

Six lines of business: private passenger auto, commercial auto, workers compensation, medical malpractice, other liability, and product liability. Each contains schedule P triangle data for accident years 1998–2007 across hundreds of US insurers.

- 70,894 rows after cleaning
- 768 unique company / accident year combinations
- Train: accident years 1998–2002 (33,567 rows)
- Test: accident years 2003–2007 (30,719 rows) — never seen during training

---

## Methodology

### Chain-ladder baseline

For each development lag, the median age-to-age (ATA) factor is computed across all companies in the training set. To predict ultimate losses for a snapshot at lag *k*, those factors are chained from lag *k* through lag 9:

```
ultimate = incurred_loss × ATA(k) × ATA(k+1) × ... × ATA(9)
```

Median factors are used rather than volume-weighted averages to be robust against outlier companies. Factors are pooled across lines of business — the simplest valid implementation and the correct baseline to benchmark against.

A key empirical finding: all ATA factors in this dataset are ≤ 1.0. This reflects that the CAS data uses *net incurred losses*, which include reserve releases. When reserves are released, incurred losses decrease — producing sub-1.0 factors. This is a real-world actuarial nuance that a naive positive-development assumption would miss.

### Feature engineering

Each row represents a claim snapshot at a given development lag (1–9). The target is the ultimate incurred loss at lag 10 for that company / accident year / line combination.

| Feature | Description |
|---|---|
| `dev_lag` | Development year (1–9) |
| `maturity_pct` | `dev_lag / 9.0` — normalized maturity signal |
| `incurred_loss` | Cumulative incurred loss at snapshot date |
| `paid_loss` | Cumulative paid loss at snapshot date |
| `case_reserve` | `incurred - paid`, clipped to 0 (reserve releases) |
| `paid_ratio` | `paid / incurred`, clipped to 2.0 (data artifacts) |
| `log_incurred` | Log-transformed incurred — stabilizes heavy tail |
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

Depth-4 trees prevent memorizing company-specific noise. Validation loss stabilized around round 150–200 with no overfitting through round 300.

---

## Results

### Overall

| Model | RMSE | vs baseline |
|---|---|---|
| Chain-ladder | $390,109 | — |
| XGBoost | $197,402 | −49.4% |

### By line of business

| Line | Chain-ladder RMSE | XGBoost RMSE | Improvement |
|---|---|---|---|
| comauto | $27,773 | $3,543 | −87.2% |
| medmal | $41,515 | $18,110 | −56.4% |
| othliab | $23,298 | $31,275 | +34.2% ⚠ |
| ppauto | $910,451 | $459,491 | −49.5% |
| prodliab | $10,000 | $2,198 | −78.0% |
| wkcomp | $48,113 | $22,837 | −52.6% |

**Notable finding:** `othliab` (other liability) is the one line where chain-ladder outperforms XGBoost. Other liability has the most heterogeneous claim mix of any line — general liability, environmental, professional, and more — which may require line-specific feature engineering or a longer training window to model reliably. This is an honest limitation and a direction for future work.

`ppauto`'s high absolute RMSE reflects its large exposure base and individual claim sizes; the 49.5% percentage improvement is consistent with the other lines.

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
- **Actuarial domain knowledge:** chain-ladder implementation, ATA factor analysis, reserve triangle interpretation
- **Machine learning:** gradient boosting, log-target transformation, chronological train/test split, overfitting diagnostics
- **Software engineering:** modular src layout, unit tests with pytest, typed function signatures, git history
