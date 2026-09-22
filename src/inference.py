"""Load the saved model, validate customer input and produce a prediction."""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (CATEGORY_VALUES, CLEAN_PATH, INPUT_CATEGORICAL, MODEL_FEATURES,  # noqa: E402
                               PROTECTION_SERVICES, ROOT, STREAMING_SERVICES, add_features)

MODEL_PATH = os.path.join(ROOT, "model", "churn_model.joblib")
META_PATH = os.path.join(ROOT, "model", "model_metrics.json")

TENURE_RANGE = (0, 120)
MONTHLY_RANGE = (0.0, 500.0)
TRAIN_TENURE = (0, 72)          # range seen in training data
TRAIN_MONTHLY = (18.25, 118.75)


class ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def risk_tier(p: float, cutoffs: dict) -> str:
    if p >= cutoffs["high"]:
        return "High"
    if p >= cutoffs["medium"]:
        return "Medium"
    return "Low"


def validate_payload(payload: dict) -> tuple[pd.DataFrame, list[str]]:
    """Validate raw customer JSON and return (one-row model frame, warnings)."""
    if not isinstance(payload, dict):
        raise ValidationError(["Request body must be a JSON object."])
    errors, warnings = [], []

    for col in ["tenure", "MonthlyCharges"]:
        if col not in payload:
            errors.append(f"Missing field: {col}")
    for col in INPUT_CATEGORICAL:
        if col not in payload:
            errors.append(f"Missing field: {col}")
    if errors:
        raise ValidationError(errors)

    def as_number(name, lo, hi):
        v = payload[name]
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            errors.append(f"{name} must be a number")
            return None
        try:
            x = float(v)
        except ValueError:
            errors.append(f"{name} must be a number")
            return None
        if not np.isfinite(x) or x < lo or x > hi:
            errors.append(f"{name} must be between {lo} and {hi}")
            return None
        return x

    tenure = as_number("tenure", *TENURE_RANGE)
    monthly = as_number("MonthlyCharges", *MONTHLY_RANGE)
    total = None
    if payload.get("TotalCharges") not in (None, ""):
        total = as_number("TotalCharges", 0.0, 100000.0)
    elif tenure is not None and monthly is not None:
        total = tenure * monthly
        warnings.append("TotalCharges not supplied; estimated as tenure x MonthlyCharges.")

    for col in INPUT_CATEGORICAL:
        if payload[col] not in CATEGORY_VALUES[col]:
            errors.append(f"{col} must be one of {CATEGORY_VALUES[col]}")

    if not errors:
        if payload["InternetService"] == "No":
            for col in PROTECTION_SERVICES + STREAMING_SERVICES:
                if payload[col] == "Yes":
                    errors.append(f"{col} cannot be 'Yes' when InternetService is 'No'")
        if payload["PhoneService"] == "No" and payload["MultipleLines"] == "Yes":
            errors.append("MultipleLines cannot be 'Yes' when PhoneService is 'No'")
    if errors:
        raise ValidationError(errors)

    if tenure is not None and not (TRAIN_TENURE[0] <= tenure <= TRAIN_TENURE[1]):
        warnings.append(f"tenure is outside the training range {TRAIN_TENURE}; prediction is an extrapolation.")
    if monthly is not None and not (TRAIN_MONTHLY[0] <= monthly <= TRAIN_MONTHLY[1]):
        warnings.append(f"MonthlyCharges is outside the training range {TRAIN_MONTHLY}; prediction is an extrapolation.")

    row = {c: payload[c] for c in INPUT_CATEGORICAL}
    row.update({"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total})
    frame = add_features(pd.DataFrame([row]))
    return frame[MODEL_FEATURES], warnings


def _group_name(transformed_name: str) -> tuple[str, str]:
    """'cat__Contract_One year' -> ('Contract', 'One year'); 'num__tenure' -> ('tenure', '')."""
    kind, rest = transformed_name.split("__", 1)
    if kind == "num":
        # TotalCharges is ~ tenure x MonthlyCharges; explain them together as one "tenure" effect
        return ("tenure" if rest == "TotalCharges" else rest), ""
    for feat in sorted(INPUT_CATEGORICAL, key=len, reverse=True):
        if rest.startswith(feat + "_"):
            return feat, rest[len(feat) + 1:]
    return rest, ""


def explain(model, X_row: pd.DataFrame, reference: pd.DataFrame, top_n: int = 3) -> dict | None:
    """Per-customer contributions (log-odds vs. the average customer) for linear models."""
    clf = model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        return None
    prep = model.named_steps["prep"]
    names = prep.get_feature_names_out()
    x = prep.transform(X_row)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    ref = prep.transform(reference)
    ref = ref.toarray() if hasattr(ref, "toarray") else np.asarray(ref)
    contrib = clf.coef_[0] * (x[0] - ref.mean(axis=0))
    grouped: dict[str, float] = {}
    for n, c in zip(names, contrib):
        g, _ = _group_name(n)
        grouped[g] = grouped.get(g, 0.0) + float(c)
    items = []
    for g, c in grouped.items():
        val = X_row.iloc[0][g]
        val = f"{val:.0f}" if g == "tenure" else (f"{val:.2f}" if isinstance(val, float) else str(val))
        items.append({"feature": g, "value": val, "impact": round(c, 3)})
    up = sorted([i for i in items if i["impact"] > 0], key=lambda i: -i["impact"])[:top_n]
    down = sorted([i for i in items if i["impact"] < 0], key=lambda i: i["impact"])[:top_n]
    return {"increases_risk": up, "decreases_risk": down,
            "note": "Impact = contribution to churn log-odds relative to the average customer."}


class Predictor:
    """Holds the model, metadata and a reference sample used for explanations."""

    def __init__(self):
        self.model, self.meta = load_artifacts()
        ref = pd.read_csv(CLEAN_PATH)
        self.reference = ref[MODEL_FEATURES]

    def predict(self, payload: dict) -> dict:
        X_row, warnings = validate_payload(payload)
        p = float(self.model.predict_proba(X_row)[0, 1])
        thr = self.meta["decision_threshold"]
        return {
            "churn_probability": round(p, 4),
            "risk_tier": risk_tier(p, self.meta["risk_cutoffs"]),
            "predicted_churn": bool(p >= thr),
            "decision_threshold": thr,
            "model": self.meta["selected_model"],
            "explanation": explain(self.model, X_row, self.reference),
            "warnings": warnings,
        }
