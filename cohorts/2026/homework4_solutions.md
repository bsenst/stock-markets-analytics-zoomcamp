# Module 4 Homework Solutions (2026 Cohort)

**Repository:** https://github.com/bsenst/stock-markets-analytics-zoomcamp  
**Base notebook:** `[2026]_Module_04_Colab_Trading_Simulations.ipynb`  
**Open in Colab:** https://colab.research.google.com/github/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/04-trading-strategy-and-simulation/%5B2026%5D_Module_04_Colab_Trading_Simulations.ipynb

> **Important**: For every question make a **fresh copy** of the notebook and start from scratch. Do not accumulate changes across questions in the same notebook instance.

---

## Question 1 – Less Features is More

**Goal:** Reduce dummy features and report the new CAGR of `pred10_rf_best_rule_55`.

### Code changes

1. Locate the cell that builds the categorical list and generates dummies (around the section **0.2.2) Generate all dummies**).

Original pattern:

```python
# what are the categorical features?
CATEGORICAL.append('month_wom')
CATEGORICAL

# Generate dummy variables ...
dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
```

**Replace with (recommended clean version):**

```python
# === Q1 CHANGE: remove ticker and month_wom from CATEGORICAL ===
# Keep only Month, Weekday, ticker_type
CATEGORICAL = ['Month', 'Weekday', 'ticker_type']
print("CATEGORICAL used for dummies:", CATEGORICAL)

dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
```

2. Verify feature count:

```python
df_with_dummies[NUMERICAL + DUMMIES].info()
# Expected: ~208 features (original was 301)
```

3. Run the **entire** notebook to the end (bubble chart / `df_sim1_results`).

### Answer to report

Look up the row for `pred10_rf_best_rule_55` in `df_sim1_results` and read the `sim1_CAGR` column.

- Original CAGR (full feature set) ≈ **1.1195**
- New CAGR (reduced features) should be **slightly higher**.

**Answer (Q1):** _run the notebook and fill the CAGR value here_

---

## Question 2 – Most Efficient Random Forest Decision Rule

**Goal:** Find the probability threshold (among 0.56 … 0.64) that yields the best CAGR while keeping **average trades per day < 5**.

### Starting point
Use a **fresh copy of the notebook that already contains the Q1 change** (reduced dummies, RF with `max_depth=13`, `n_estimators=100`).

### Code changes

1. **Build the TPR/FPR dataframe for Random Forest** (section 2.1):

```python
# Comment out the Decision-Tree line and enable RF:
# y_pred_test = clf_best.predict_proba(X_test)
y_pred_test = rf_best.predict_proba(X_test)

y_pred_test_class1 = [k[1] for k in y_pred_test]
```

Then re-run the `tpr_fpr_dataframe(...)` cell so `df_scores` reflects RF probabilities.

2. **Add new prediction columns** for thresholds 0.56–0.64 (right after the existing `pred10` / `pred11` definitions):

```python
# Existing:
new_df['pred10_rf_best_rule_55'] = (y_pred_all_class1_array >= 0.55).astype(int)
new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)

# === Q2: additional RF thresholds ===
for thr_int in range(56, 65):
    thr = thr_int / 100
    col = f'pred{thr_int}_rf_best_rule_{thr_int}'
    new_df[col] = (y_pred_all_class1_array >= thr).astype(int)
    test_rate = new_df.loc[new_df.split == 'test', col].mean()
    print(f'{col}: positive rate on test = {test_rate:.4f}')
```

3. Make sure the new columns are picked up by the simulation loop.  
   Re-run `get_predictions_correctness(...)` **after** adding the new columns so they appear in `PREDICTIONS` / `df_sim1_results`.

4. Inspect `df_sim1_results`:

```python
rf_results = df_sim1_results[
    df_sim1_results.prediction.str.contains('rf_best_rule')
][['prediction', 'sim1_count_investments',
   'sim1_avg_investments_per_day', 'sim1_CAGR']] \
 .sort_values('sim1_CAGR', ascending=False)

print(rf_results)

# Filter to avg trades per day < 5 and pick the best CAGR among them
second_best = rf_results[rf_results.sim1_avg_investments_per_day < 5] \
    .sort_values('sim1_CAGR', ascending=False)
print("\nSecond-best (avg trades/day < 5):")
print(second_best.head(3))
```

**Answer (Q2):** the threshold value (e.g. `0.58` or `0.60` …) – _fill after running_

---

## Question 3 – Redefining the Growth Target

**Goal:** Change the definition of the binary target and report the number of trades of the new best predictor `pred5_clf_10`.

### Starting point
Make a **fresh copy of the original Module 04 notebook** (do **not** keep the Q1/Q2 changes).

### Code changes

1. After the data is loaded and `new_df = df_with_dummies.copy()`, redefine the target:

```python
# === Q3 CHANGE: stricter positive-growth definition ===
# Original: is_positive_growth_30d_future == 1 when growth_future_30d > 1
# New:     is_positive_growth_30d_future == 1 when growth_future_30d > 1.022

new_df['is_positive_growth_30d_future'] = (
    new_df['growth_future_30d'] > 1.022
).astype(int)

print("New positive rate (overall):",
      new_df['is_positive_growth_30d_future'].mean())
print(new_df.groupby('split')['is_positive_growth_30d_future'].mean())
# Expected overall rate ≈ 0.50 (down from ≈ 0.60)
```

Place this cell **after** `new_df = df_with_dummies.copy()` and **before** any modelling cells.

2. Run the whole notebook again.

3. In the final `df_sim1_results` the best CAGR is expected to come from `pred5_clf_10`.  
   Read its `sim1_count_investments` (total number of trades on the test set).

**Answer (Q3):** number of trades for `pred5_clf_10` – _fill after running_

---

## Question 4 – Ideal Trading Strategy (Exploratory)

**Example answer you can adapt / improve:**

> My ideal strategy combines the Random-Forest probability with a daily capital budget and a concentration limit.
>
> 1. Every trading day I rank all stocks by the RF probability (`proba_pred10`).
> 2. I only consider stocks whose probability ≥ 0.60 (the efficient threshold found in Q2).
> 3. I invest the available daily capital equally in the top-3 ranked names (or fewer if less than three pass the threshold).
> 4. Position size is capped at 5 % of total portfolio value to control single-name risk.
> 5. I never hold more than 15 open positions at once; if a new high-probability signal appears I close the lowest-conviction existing position first.
>
> This keeps average daily trades well below 5, concentrates capital on the highest-conviction signals, and adds simple risk controls that the pure “$100 per positive prediction” rule lacks.

Feel free to replace the numbers with whatever thresholds/CAGR numbers you observed in the simulations.

---

## How to submit

1. Run the three modified notebooks (or one notebook per question) in Colab.
2. Fill the numeric answers above.
3. Submit the form: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04
4. (Optional) Push the solved notebooks to your fork under `cohorts/2026/`.

---

## Quick checklist

| Question | Key change                              | What to report                          |
|----------|-----------------------------------------|-----------------------------------------|
| Q1       | Remove `Ticker` + `month_wom` dummies   | CAGR of `pred10_rf_best_rule_55`        |
| Q2       | Extra RF thresholds 0.56–0.64           | Threshold with avg trades/day < 5       |
| Q3       | `growth_future_30d > 1.022`             | # trades of `pred5_clf_10`              |
| Q4       | Free-text strategy description          | A few clear sentences                   |
