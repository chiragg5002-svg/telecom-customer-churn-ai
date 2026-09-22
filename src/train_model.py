"""Train, compare, evaluate and save the churn model.

Run from the project root:  python src/train_model.py
Produces: data/processed/*.csv, model/*.joblib|json|csv
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import best_f1_threshold, classification_metrics  # noqa: E402
from src.preprocessing import (CATEGORICAL_FEATURES, CLEAN_PATH, MODEL_FEATURES, NUMERIC_FEATURES,  # noqa: E402
                               PROCESSED_DIR, ROOT, SCORED_PATH, TARGET, clean_data, load_raw)

SEED = 42
MODEL_DIR = os.path.join(ROOT, "model")

# Risk tiers on predicted churn probability (cut-offs chosen from the observed
# churn rate per tier on out-of-fold scores - see notebook section 9).
RISK_CUTOFFS = {"medium": 0.30, "high": 0.60}


def risk_tier(p: float) -> str:
    if p >= RISK_CUTOFFS["high"]:
        return "High"
    if p >= RISK_CUTOFFS["medium"]:
        return "Medium"
    return "Low"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                        max_depth=3, random_state=SEED),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def main() -> dict:
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw = load_raw()
    df, cleaning_log = clean_data(raw)
    df.to_csv(CLEAN_PATH, index=False)

    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- baseline + model comparison (5-fold CV on the training split) ----
    comparison = {}
    base = DummyClassifier(strategy="prior").fit(X_train, y_train)
    base_proba = base.predict_proba(X_test)[:, 1]
    comparison["Baseline (majority class)"] = {
        "cv_roc_auc": 0.5, "cv_pr_auc": float(y_train.mean()),
        "test": classification_metrics(y_test, base_proba, 0.5)}
    fitted = {}
    for name, est in candidate_models().items():
        pipe = make_pipeline(est)
        cvr = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring=["roc_auc", "average_precision"], n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        comparison[name] = {
            "cv_roc_auc": float(cvr["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(cvr["test_roc_auc"].std()),
            "cv_pr_auc": float(cvr["test_average_precision"].mean()),
            "test": classification_metrics(y_test, pipe.predict_proba(X_test)[:, 1], 0.5),
        }

    # ---- model selection: best CV ROC-AUC; prefer the simplest model within 0.005 ----
    ranked = sorted(fitted, key=lambda n: comparison[n]["cv_roc_auc"], reverse=True)
    best = ranked[0]
    if comparison["Logistic Regression"]["cv_roc_auc"] >= comparison[best]["cv_roc_auc"] - 0.005:
        best = "Logistic Regression"
    final = fitted[best]

    # ---- threshold chosen on out-of-fold TRAIN predictions only (no test leakage) ----
    oof_train = cross_val_predict(clone(final), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof_train)
    proba_test = final.predict_proba(X_test)[:, 1]
    test_default = classification_metrics(y_test, proba_test, 0.5)
    test_tuned = classification_metrics(y_test, proba_test, threshold)

    # ---- permutation importance on the held-out test set (raw input columns) ----
    pi = permutation_importance(final, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": MODEL_FEATURES, "importance": pi.importances_mean,
                                "std": pi.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    importance.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

    # ---- out-of-fold scores for EVERY customer (each scored by a model that never saw them) ----
    oof_all = cross_val_predict(clone(final), X, y, cv=cv, method="predict_proba")[:, 1]
    scored = df.copy()
    scored["churn_probability"] = oof_all
    scored["risk_tier"] = [risk_tier(p) for p in oof_all]
    scored.to_csv(SCORED_PATH, index=False)

    meta = {
        "project": "Telecom Customer Churn - Decision Dashboard",
        "selected_model": best,
        "selection_rule": "Highest 5-fold CV ROC-AUC on the training split; the simpler Logistic "
                          "Regression is preferred when within 0.005 AUC of the best model.",
        "decision_threshold": threshold,
        "threshold_rule": "Maximises F1 on out-of-fold predictions of the training split.",
        "risk_cutoffs": RISK_CUTOFFS,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "features": MODEL_FEATURES, "random_state": SEED,
        "sklearn_version": sklearn.__version__,
        "trained_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "comparison": comparison,
        "test_metrics_default_threshold": test_default,
        "test_metrics_tuned_threshold": test_tuned,
        "cleaning_log": cleaning_log,
    }
    joblib.dump(final, os.path.join(MODEL_DIR, "churn_model.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Selected model: {best} | threshold={threshold:.2f} | {time.time()-t0:.0f}s")
    for n, r in comparison.items():
        print(f"  {n:28s} CV AUC={r['cv_roc_auc']:.4f} | test AUC={r['test']['roc_auc']:.4f}")
    print("Test @0.5     :", {k: round(v, 3) for k, v in test_default.items() if isinstance(v, float)})
    print("Test @tuned   :", {k: round(v, 3) for k, v in test_tuned.items() if isinstance(v, float)})
    return meta


if __name__ == "__main__":
    main()
