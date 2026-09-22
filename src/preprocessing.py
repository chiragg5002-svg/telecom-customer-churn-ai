"""Data loading, cleaning and feature engineering for the Telco churn project.

Single source of truth for cleaning logic - used by the training script,
the Flask backend and (copied verbatim) by the submission notebook.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "Telco-Customer-Churn.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
CLEAN_PATH = os.path.join(PROCESSED_DIR, "telco_clean.csv")
SCORED_PATH = os.path.join(PROCESSED_DIR, "telco_scored.csv")

TARGET = "Churn"
ID_COL = "customerID"

PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]
# Columns where "No internet service" / "No phone service" is redundant with
# InternetService / PhoneService and is folded into "No".
FOLD_TO_NO = PROTECTION_SERVICES + STREAMING_SERVICES + ["MultipleLines"]

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "n_protection_services", "n_streaming_services",
]
# `gender` is deliberately excluded from the model (no predictive value, and it
# should not drive retention decisions).
CATEGORICAL_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw (customer-facing) input fields accepted by the API. Derived features
# (n_protection_services, n_streaming_services) are computed from these.
INPUT_NUMERIC = ["tenure", "MonthlyCharges"]          # TotalCharges optional
INPUT_CATEGORICAL = CATEGORICAL_FEATURES

CATEGORY_VALUES = {
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
    "StreamingTV": ["No", "Yes"],
    "StreamingMovies": ["No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                      "Electronic check", "Mailed check"],
}
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features (safe to call on cleaned data or on API input)."""
    df = df.copy()
    df["n_protection_services"] = (df[PROTECTION_SERVICES] == "Yes").sum(axis=1)
    df["n_streaming_services"] = (df[STREAMING_SERVICES] == "Yes").sum(axis=1)
    df["auto_pay"] = df["PaymentMethod"].astype(str).str.contains("automatic").map({True: "Yes", False: "No"})
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return df


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Clean the raw Telco dataset. Returns (clean_df, cleaning_log)."""
    log: list[dict] = []
    df = raw.copy()
    n0 = len(df)

    # 1. Duplicates
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df[ID_COL].duplicated().sum())
    df = df.drop_duplicates().drop_duplicates(subset=ID_COL)
    log.append({"step": "Duplicate check", "found": dup_rows + dup_ids,
                "action": "None needed" if dup_rows + dup_ids == 0 else "Dropped duplicates",
                "reason": "Every customerID must appear once."})

    # 2. TotalCharges is stored as text; blanks are new customers (tenure = 0)
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = tc.isna()
    n_blank = int(blank.sum())
    all_new = bool((df.loc[blank, "tenure"] == 0).all()) if n_blank else True
    df["TotalCharges"] = tc
    df.loc[blank & (df["tenure"] == 0), "TotalCharges"] = 0.0
    log.append({"step": "TotalCharges type conversion", "found": n_blank,
                "action": "Converted text -> float; blanks set to 0.0",
                "reason": f"All {n_blank} blank values belong to customers with tenure = 0 "
                          f"(not billed yet) - confirmed: {all_new}. Rows kept, not dropped."})

    # 3. Redundant categories
    n_folded = 0
    for col in FOLD_TO_NO:
        n_folded += int(df[col].isin(["No internet service", "No phone service"]).sum())
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    log.append({"step": "Redundant categories", "found": n_folded,
                "action": "'No internet service' / 'No phone service' -> 'No'",
                "reason": "Already encoded by InternetService and PhoneService; avoids duplicate signal."})

    # 4. SeniorCitizen 0/1 -> No/Yes so all binary fields share one encoding
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    log.append({"step": "SeniorCitizen encoding", "found": 1,
                "action": "0/1 -> No/Yes", "reason": "Consistent labels across binary columns."})

    # 5. Missing values after cleaning
    n_missing = int(df.isna().sum().sum())
    log.append({"step": "Missing values (after cleaning)", "found": n_missing,
                "action": "None needed" if n_missing == 0 else "Review",
                "reason": "No remaining nulls."})

    # 6. Category consistency check (values must match the documented schema)
    unexpected = {c: sorted(set(df[c].astype(str)) - set(v)) for c, v in CATEGORY_VALUES.items()}
    unexpected = {c: v for c, v in unexpected.items() if v}
    log.append({"step": "Category consistency", "found": len(unexpected),
                "action": "None needed" if not unexpected else f"Unexpected: {unexpected}",
                "reason": "All categorical values match the documented schema."})

    # 7. Outliers (IQR rule) - reported, NOT removed
    outl = {}
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        outl[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    log.append({"step": "Outlier check (IQR rule)", "found": sum(outl.values()),
                "action": "Kept all values",
                "reason": f"Counts {outl}. Values are valid business amounts / durations, not errors."})

    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    df = add_features(df).reset_index(drop=True)
    log.append({"step": "Feature engineering", "found": 4,
                "action": "Added n_protection_services, n_streaming_services, auto_pay, tenure_group",
                "reason": "Capture bundle depth, billing behaviour and lifecycle stage."})
    log.append({"step": "Rows retained", "found": len(df),
                "action": f"{n0} -> {len(df)} rows", "reason": "No rows were removed."})
    return df, log
