"""
Homework 4 – ready-to-paste code snippets
Stock Markets Analytics Zoomcamp 2026

Copy each block into the corresponding place of a fresh copy of
[2026]_Module_04_Colab_Trading_Simulations.ipynb
"""

# ---------------------------------------------------------------------------
# QUESTION 1 – Less Features is More
# ---------------------------------------------------------------------------
# Location: section 0.2.2) Generate all dummies
# Replace the cell that does CATEGORICAL.append('month_wom') + get_dummies

# --- snippet start ---
# After the cell that creates month_wom and does:
#   CATEGORICAL.append('month_wom')
# OVERRIDE the list so that Ticker and month_wom are never turned into dummies:

CATEGORICAL = ['Month', 'Weekday', 'ticker_type']   # removed Ticker + month_wom
print("CATEGORICAL used for dummies:", CATEGORICAL)

dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
# --- snippet end ---

# Verification cell (run right after creating df_with_dummies):
# df_with_dummies[NUMERICAL + DUMMIES].info()
# Expected: ~208 features (original ≈ 301)


# ---------------------------------------------------------------------------
# QUESTION 2 – Most Efficient RF Decision Rule
# ---------------------------------------------------------------------------
# Start from a notebook that already contains the Q1 change.

# 2a. Switch the probability source to Random Forest
# Location: section 2.1 (the cell that currently does clf_best.predict_proba)

# --- snippet start ---
# y_pred_test = clf_best.predict_proba(X_test)          # ← comment out
y_pred_test = rf_best.predict_proba(X_test)              # ← enable this

y_pred_test_class1 = [k[1] for k in y_pred_test]
# --- snippet end ---

# Then re-run the tpr_fpr_dataframe(...) cell so df_scores is built from RF.

# 2b. Add extra threshold columns
# Location: right after the existing pred10 / pred11 definitions

# --- snippet start ---
# Existing lines (keep them):
# new_df['pred10_rf_best_rule_55'] = (y_pred_all_class1_array >= 0.55).astype(int)
# new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)

# New thresholds 0.56 … 0.64
for thr_int in range(56, 65):
    thr = thr_int / 100
    col = f'pred{thr_int}_rf_best_rule_{thr_int}'
    new_df[col] = (y_pred_all_class1_array >= thr).astype(int)
    test_rate = new_df.loc[new_df.split == 'test', col].mean()
    print(f'{col}: positive rate on test = {test_rate:.4f}')
# --- snippet end ---

# After adding the columns, re-run:
#   PREDICTIONS, IS_CORRECT = get_predictions_correctness(new_df, to_predict=...)
# so that the new columns appear in the simulation loop.

# 2c. Inspect results
# --- snippet start ---
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
# --- snippet end ---


# ---------------------------------------------------------------------------
# QUESTION 3 – Redefining the Growth Target
# ---------------------------------------------------------------------------
# Start from a FRESH copy of the original Module 04 notebook
# (do NOT keep Q1/Q2 changes).

# Location: after   new_df = df_with_dummies.copy()
#           and before any modelling cells.

# --- snippet start ---
# Stricter positive-growth definition
new_df['is_positive_growth_30d_future'] = (
    new_df['growth_future_30d'] > 1.022
).astype(int)

print("New positive rate (overall):",
      new_df['is_positive_growth_30d_future'].mean())
print(new_df.groupby('split')['is_positive_growth_30d_future'].mean())
# Expected overall rate ≈ 0.50 (down from ≈ 0.60)
# --- snippet end ---

# Run the whole notebook.
# In df_sim1_results the best CAGR should now belong to pred5_clf_10.
# Report its sim1_count_investments.


# ---------------------------------------------------------------------------
# QUESTION 4 – Ideal Trading Strategy (text answer)
# ---------------------------------------------------------------------------
"""
Example strategy description (adapt with your own numbers):

My ideal strategy uses the Random-Forest probability together with a daily
capital budget and simple risk controls.

1. Every day I rank all stocks by the RF probability (proba_pred10).
2. I only consider names whose probability ≥ 0.60 (the efficient threshold
   found in Question 2).
3. I allocate the available daily capital equally across the top-3 ranked
   names (or fewer if less than three pass the filter).
4. Single-name exposure is capped at 5 % of total portfolio value.
5. I never hold more than 15 open positions; if a stronger signal appears
   I close the lowest-conviction existing position first.

This keeps average daily trades well below 5, concentrates capital on the
highest-conviction signals, and adds risk controls that the pure
“$100 per positive prediction” rule lacks.
"""
