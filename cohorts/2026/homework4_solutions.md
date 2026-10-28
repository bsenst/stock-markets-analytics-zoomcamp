# Module 4 Homework Solutions (2026 Cohort)

**Repository:** https://github.com/bsenst/stock-markets-analytics-zoomcamp  
**Base notebook:** `[2026]_Module_04_Colab_Trading_Simulations.ipynb`  
**Open original in Colab:** https://colab.research.google.com/github/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/04-trading-strategy-and-simulation/%5B2026%5D_Module_04_Colab_Trading_Simulations.ipynb

> **Important**: For every question make a **fresh copy** of the notebook and start from scratch. Do not accumulate changes across questions in the same notebook instance.

---

## Colab-ready notebook variants (recommended)

Three pre-modified notebooks are available (download from the conversation artifacts or the zip):

| Notebook | Purpose |
|----------|---------|
| `HW4_Q1_Less_Features.ipynb` | Q1 – reduced dummy features |
| `HW4_Q2_RF_Thresholds.ipynb` | Q2 – RF thresholds 0.56–0.64 (includes Q1 changes) |
| `HW4_Q3_Redefined_Target.ipynb` | Q3 – stricter growth target `> 1.022` |

Upload any of them to Google Colab (**File → Upload notebook**) and **Runtime → Run all**.

Alternatively use the code snippets below on a fresh copy of the original Module 04 notebook.

---

## Question 1 – Less Features is More

**Goal:** Reduce dummy features and report the new CAGR of `pred10_rf_best_rule_55`.

### Code changes

1. Locate the cell that builds the categorical list and generates dummies (around the section **0.2.2) Generate all dummies**).

**Replace with:**

```python
# === Q1 CHANGE: remove ticker and month_wom from CATEGORICAL ===
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
Use a **fresh copy of the notebook that already contains the Q1 change** (or use `HW4_Q2_RF_Thresholds.ipynb`).

### Code changes

1. Switch the probability source to Random Forest (section 2.1):

```python
# y_pred_test = clf_best.predict_proba(X_test)
y_pred_test = rf_best.predict_proba(X_test)
y_pred_test_class1 = [k[1] for k in y_pred_test]
```

2. Add new prediction columns for thresholds 0.56–0.64 (after the existing `pred10` / `pred11` definitions):

```python
for thr_int in range(56, 65):
    thr = thr_int / 100
    col = f'pred{thr_int}_rf_best_rule_{thr_int}'
    new_df[col] = (y_pred_all_class1_array >= thr).astype(int)
    test_rate = new_df.loc[new_df.split == 'test', col].mean()
    print(f'{col}: positive rate on test = {test_rate:.4f}')
```

3. Re-run `get_predictions_correctness(...)` so the new columns appear in the simulation.

4. Inspect results:

```python
rf_results = df_sim1_results[
    df_sim1_results.prediction.str.contains('rf_best_rule')
][['prediction', 'sim1_count_investments',
   'sim1_avg_investments_per_day', 'sim1_CAGR']] \
 .sort_values('sim1_CAGR', ascending=False)

second_best = rf_results[rf_results.sim1_avg_investments_per_day < 5] \
    .sort_values('sim1_CAGR', ascending=False)
print(second_best.head(3))
```

**Answer (Q2):** the threshold value – _fill after running_

---

## Question 3 – Redefining the Growth Target

**Goal:** Change the definition of the binary target and report the number of trades of the new best predictor `pred5_clf_10`.

### Starting point
Make a **fresh copy of the original Module 04 notebook** (or use `HW4_Q3_Redefined_Target.ipynb`).

### Code changes

After `new_df = df_with_dummies.copy()`:

```python
# === Q3 CHANGE: stricter positive-growth definition ===
new_df['is_positive_growth_30d_future'] = (
    new_df['growth_future_30d'] > 1.022
).astype(int)

print("New positive rate (overall):",
      new_df['is_positive_growth_30d_future'].mean())
print(new_df.groupby('split')['is_positive_growth_30d_future'].mean())
```

Run the whole notebook. Report `sim1_count_investments` for `pred5_clf_10`.

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

---

## How to submit

1. Run the three modified notebooks (or one notebook per question) in Colab.
2. Fill the numeric answers above.
3. Submit the form: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04

---

## Quick checklist

| Question | Key change                              | What to report                          |
|----------|-----------------------------------------|-----------------------------------------|
| Q1       | Remove `Ticker` + `month_wom` dummies   | CAGR of `pred10_rf_best_rule_55`        |
| Q2       | Extra RF thresholds 0.56–0.64           | Threshold with avg trades/day < 5       |
| Q3       | `growth_future_30d > 1.022`             | # trades of `pred5_clf_10`              |
| Q4       | Free-text strategy description          | A few clear sentences                   |
