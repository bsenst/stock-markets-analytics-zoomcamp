"""Reproduce the Stock Markets Analytics Zoomcamp 2026 Homework 3 results.

Run from the repository root:
    python 03-modeling/homework3_solution.py

The script downloads the parquet dataset used by the 2026 Module 3 notebook on
the first run and caches it under ~/.cache/stock-markets-analytics-zoomcamp.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://drive.google.com/uc?id=1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5"
DATA_FILENAME = "stocks_df_combined_2026_09_18.parquet.brotli"
TARGET = "is_positive_growth_30d_future"

TECHNICAL_INDICATORS = [
    "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc", "bop",
    "cci", "cmo", "dx", "macd", "macdsignal", "macdhist", "macd_ext",
    "macdsignal_ext", "macdhist_ext", "macd_fix", "macdsignal_fix",
    "macdhist_fix", "mfi", "minus_di", "mom", "plus_di", "dm", "ppo",
    "roc", "rocp", "rocr", "rocr100", "rsi", "slowk", "slowd", "fastk",
    "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr", "ad",
    "adosc", "obv", "atr", "natr", "ht_dcperiod", "ht_dcphase",
    "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine",
    "ht_sine_leadsine", "ht_trendmod", "avgprice", "medprice", "typprice",
    "wclprice",
]
CUSTOM_NUMERICAL = [
    "SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative",
    "volatility", "ln_volume",
]
MACRO = [
    "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
    "FEDFUNDS", "DGS1", "DGS5", "DGS10",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading course dataset to {path}")
        downloaded_path = gdown.download(DATA_URL, str(path), quiet=False)
        if downloaded_path is None:
            raise RuntimeError(f"Could not download the course dataset: {DATA_URL}")
    return pd.read_parquet(path)


def prepare_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw_df.loc[raw_df["Date"] >= "2000-01-01"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = pd.to_datetime(df["Month"]).dt.month.astype(str)
    df["Weekday"] = df["Weekday"].astype(str)
    month_names = df["Date"].dt.month.map(lambda month: calendar.month_name[month])
    week_of_month = ((df["Date"].dt.day - 1) // 7 + 1).astype(str)
    df["month_wom"] = month_names + "_w" + week_of_month
    df["ln_volume"] = np.log(df["Volume"].replace(0, np.nan).fillna(1e-9))

    growth = [
        col for col in df.columns
        if col.startswith("growth_") and "future" not in col
    ]
    patterns = [col for col in df.columns if "cdl" in col]
    numerical = growth + TECHNICAL_INDICATORS + patterns + CUSTOM_NUMERICAL + MACRO
    categorical = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]
    required = [
        "Date", "Volume", "cci", "growth_30d", "growth_snp500_30d", "DGS10",
        "DGS5", "FEDFUNDS", TARGET, *numerical, *categorical,
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required notebook columns: {missing}")

    dummies = pd.get_dummies(df[categorical], dtype="int32")
    df = pd.concat([df, dummies], axis=1)

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_span = max_date - min_date
    train_end = min_date + date_span * 0.70
    validation_end = train_end + date_span * 0.15
    df["split"] = np.select(
        [df["Date"] <= train_end, df["Date"] <= validation_end],
        ["train", "validation"],
        default="test",
    )

    features = numerical + dummies.columns.tolist()
    return df, features


def positive_precision(
    y_true: pd.Series, y_pred: np.ndarray, test_mask: pd.Series
) -> float:
    return float(
        precision_score(
            y_true.loc[test_mask], y_pred[test_mask.to_numpy()], zero_division=0
        )
    )


def run(data_path: Path, predictions_path: Path | None) -> None:
    df, features = prepare_dataset(load_data(data_path))
    y = df[TARGET].astype(int)
    test_mask = df["split"].eq("test")
    train_validation_mask = df["split"].isin(["train", "validation"])

    # Q1: rank only the month/week-of-month dummy variables by absolute Pearson r.
    month_wom_dummies = [col for col in df.columns if col.startswith("month_wom_")]
    month_wom_corr = df[month_wom_dummies + [TARGET]].corr()[TARGET].drop(TARGET)
    best_month_wom = month_wom_corr.abs().idxmax()
    best_month_wom_corr = float(month_wom_corr[best_month_wom])

    # Q2: the three original hand rules and the two 2026 macro rules.
    df["pred0_manual_cci"] = (df["cci"] > 200).astype(int)
    df["pred1_manual_prev_g1"] = (df["growth_30d"] > 1).astype(int)
    df["pred2_manual_prev_g1_and_snp"] = (
        (df["growth_30d"] > 1) & (df["growth_snp500_30d"] > 1)
    ).astype(int)
    df["pred3_manual_dgs10_5"] = (
        (df["DGS10"] <= 4.5) & (df["DGS5"] <= 4)
    ).astype(int)
    df["pred4_manual_dgs10_fedfunds"] = (
        (df["DGS10"] > 4) & (df["FEDFUNDS"] <= 4.795)
    ).astype(int)
    prediction_columns = [
        "pred0_manual_cci", "pred1_manual_prev_g1",
        "pred2_manual_prev_g1_and_snp", "pred3_manual_dgs10_5",
        "pred4_manual_dgs10_fedfunds",
    ]
    hand_precision = {
        column: positive_precision(y, df[column].to_numpy(), test_mask)
        for column in prediction_columns
    }
    best_new_rule = max(prediction_columns[3:], key=hand_precision.get)

    # Match the notebook's feature cleaning: replace infinities and NaNs with 0.
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train_validation = X.loc[train_validation_mask]
    y_train_validation = y.loc[train_validation_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    # Q3: fit on train+validation, then predict across the full dataframe.
    clf10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf10.fit(X_train_validation, y_train_validation)
    df["pred5_clf_10"] = clf10.predict(X)
    hand_wrong = pd.concat(
        [df[column].ne(y) for column in prediction_columns], axis=1
    ).all(axis=1)
    unique_correct = test_mask & df["pred5_clf_10"].eq(y) & hand_wrong
    unique_correct_count = int(unique_correct.sum())

    # Q4: follow the assignment and rank depths by test precision (1..12).
    depth_precision: dict[int, float] = {}
    for depth in range(1, 13):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train_validation, y_train_validation)
        depth_precision[depth] = float(
            precision_score(y_test, model.predict(X_test), zero_division=0)
        )
    best_depth = max(depth_precision, key=depth_precision.get)
    best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_model.fit(X_train_validation, y_train_validation)
    df["pred6_clf_best"] = best_model.predict(X)

    print(f"Rows after 2000-01-01 filter: {len(df):,}")
    print(f"Features: {len(features):,}; test rows: {int(test_mask.sum()):,}")
    print("\nQ1: Highest absolute month_wom correlation")
    print(f"  {best_month_wom}: {abs(best_month_wom_corr):.3f} (r={best_month_wom_corr:.6f})")

    print("\nQ2: TEST precision for positive hand-rule predictions")
    for column, score in hand_precision.items():
        print(f"  {column}: {score:.3f}")
    print(f"  Best new rule ({best_new_rule}): {hand_precision[best_new_rule]:.3f}")

    print("\nQ3: Unique correct pred5_clf_10 predictions on TEST")
    print(f"  {unique_correct_count}")

    print("\nQ4: TEST precision by max_depth")
    for depth, score in depth_precision.items():
        print(f"  depth {depth:2d}: {score:.3f}")
    print(f"  best_max_depth: {best_depth}")

    if predictions_path is not None:
        output_columns = [
            "pred0_manual_cci", "pred1_manual_prev_g1",
            "pred2_manual_prev_g1_and_snp", "pred3_manual_dgs10_5",
            "pred4_manual_dgs10_fedfunds", "pred5_clf_10", "pred6_clf_best",
        ]
        output_columns = ["Date", "Ticker", "split", TARGET, *output_columns]
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        df[output_columns].to_csv(predictions_path, index=False)
        print(f"\nSaved row-level predictions to {predictions_path}")


def parse_args() -> argparse.Namespace:
    default_data_path = (
        Path.home() / ".cache" / "stock-markets-analytics-zoomcamp" / DATA_FILENAME
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=default_data_path,
        help=f"Parquet path (default: {default_data_path}; downloaded if absent)",
    )
    parser.add_argument(
        "--predictions-csv", type=Path,
        help="Optional path to save all row-level manual and model predictions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.predictions_csv)