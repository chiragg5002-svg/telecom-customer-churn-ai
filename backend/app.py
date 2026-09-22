"""Flask API for the Telco churn decision-support system.

Run from the project root:  python backend/app.py      (http://127.0.0.1:5000)
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import insights as ins  # noqa: E402
from src.inference import Predictor, ValidationError  # noqa: E402
from src.preprocessing import (CATEGORY_VALUES, INPUT_CATEGORICAL, MODEL_FEATURES, SCORED_PATH,  # noqa: E402
                               TARGET)

app = Flask(__name__)
predictor = Predictor()
scored = pd.read_csv(SCORED_PATH)


def _clean(obj):
    """Make numpy/pandas values JSON friendly."""
    return jsonify(obj)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": predictor.model is not None,
                    "model": predictor.meta["selected_model"], "rows": int(len(scored))})


@app.get("/model_info")
def model_info():
    m = predictor.meta
    return jsonify({
        "selected_model": m["selected_model"], "selection_rule": m["selection_rule"],
        "decision_threshold": m["decision_threshold"], "threshold_rule": m["threshold_rule"],
        "risk_cutoffs": m["risk_cutoffs"], "features": m["features"],
        "train_rows": m["train_rows"], "test_rows": m["test_rows"], "trained_at": m["trained_at"],
        "sklearn_version": m["sklearn_version"],
        "test_metrics_default_threshold": m["test_metrics_default_threshold"],
        "test_metrics_tuned_threshold": m["test_metrics_tuned_threshold"],
        "comparison": m["comparison"],
        "input_schema": {"numeric": {"tenure": [0, 120], "MonthlyCharges": [0, 500],
                                     "TotalCharges": "optional"},
                         "categorical": {c: CATEGORY_VALUES[c] for c in INPUT_CATEGORICAL}},
    })


@app.get("/dataset")
def dataset():
    limit = min(max(request.args.get("limit", default=10, type=int), 0), 100)
    return jsonify({
        "rows": int(len(scored)), "columns": int(scored.shape[1]),
        "target": TARGET, "model_features": MODEL_FEATURES,
        "dtypes": {c: str(t) for c, t in scored.dtypes.items()},
        "missing_values": int(scored.isna().sum().sum()),
        "churn_rate": float(scored[TARGET].mean()),
        "sample": scored.head(limit).to_dict(orient="records"),
    })


@app.get("/kpis")
def kpis():
    return jsonify(ins.kpis(scored))


@app.get("/insights")
def insights():
    return jsonify(ins.build_insights(scored))


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400
    try:
        return jsonify(predictor.predict(payload))
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors}), 400


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
