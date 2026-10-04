# 2026 Homework 3 Solution

## Run

From the repository root, install dependencies and run the script:

```bash
python -m pip install -r cohorts/2026/homework3-solution/requirements-homework3.txt
python cohorts/2026/homework3-solution/homework3_solution.py
```

The script downloads and caches the parquet dataset used by the 2026 Module 3 notebook on its first run. Use `--data PATH` to provide a local copy, or `--predictions-csv PATH` to export row-level predictions.

## Results

Using the dataset dated 2026-09-18:

- Q1: `October_w4`, absolute correlation `0.025`.
- Q2: `pred3_manual_dgs10_5`, precision `0.588`.
- Q3: the script reproduces `1,243` unique correct `pred5_clf_10` predictions on the test set. The provided choices (`271`, `571`, `1,171`, `1,571`) do not include this result; `1,171` is the likely intended choice, but it is not reproduced by the published conditions and dataset.
- Q4: `best_max_depth = 4`, test precision `0.629`.

The script computes these results when run; the values above are included for reference.

## Q5 Suggestion

Add earnings dates and earnings surprises, sector-relative returns for S&P, STOXX, and NIFTY sector indices, and local-market risk indicators such as VIX, India VIX, EUR/USD, and USD/INR. These capture event risk, industry effects, and regional drivers missing from the current technical and US macro features. Potential sources include company filings and earnings calendars, exchange or market-data providers for sector indices, CBOE or FRED for VIX, NSE for India VIX, the ECB Data Portal for EUR/USD, and RBI DBIE for USD/INR.

The depth search follows the homework's instruction to select by test precision. Choosing a model based on its test score leaks information from the test set; for an unbiased evaluation, choose the depth on validation data and report the test score once.