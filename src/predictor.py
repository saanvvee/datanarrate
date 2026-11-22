"""
Prediction on new data
Refits the winning model on all rows of a finished run and uses it to predict a new CSV.

    bundle = train_bundle(df, profile, plan, result)      # once per run
    out    = predict_csv(bundle, new_df)                  # DataFrame with predictions
    blob   = bundle_to_bytes(bundle)                      # a .joblib file to download

New rows go through exactly the same steps as the training rows: hidden placeholders such
as "?" become real missing values (rule R0), and the saved pipeline then imputes, encodes
and scales them with values learned from the training data.
"""
import io
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.missing import normalize_missing
from src.model_data import model_pipeline, prepare_model_data

MAX_PREDICT_ROWS = 20_000


class PredictionInputError(ValueError):
    """The uploaded CSV cannot be used with this model (message is safe to show to the user)."""


def train_bundle(df: pd.DataFrame, profile, plan, result) -> Dict[str, Any]:
    data = prepare_model_data(df, profile, plan)
    pipe = model_pipeline(data, result.best_model_name).fit(data.X, data.y)
    return {
        "pipeline": pipe,
        "classes": data.classes,
        "target": profile.target_column,
        "best_model": result.best_model_name,
        "metric": result.metric_used,
        "cv_score": result.best_score,
        "trained_rows": int(len(data.X)),
        "feature_columns": data.X.columns.tolist(),
        "numeric_columns": [c for c in data.X.columns if pd.api.types.is_numeric_dtype(data.X[c])],
        "sample_rows": data.X.head(5).to_dict("records"),
    }


def bundle_info(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {"target": bundle["target"], "classes": bundle["classes"], "best_model": bundle["best_model"],
            "trained_rows": bundle["trained_rows"], "feature_columns": bundle["feature_columns"],
            "numeric_columns": bundle["numeric_columns"], "metric": bundle["metric"], "cv_score": bundle["cv_score"]}


def bundle_to_bytes(bundle: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def sample_csv(bundle: Dict[str, Any]) -> str:
    """A template the user can fill in: the right columns, a few example rows."""
    return pd.DataFrame(bundle["sample_rows"], columns=bundle["feature_columns"]).to_csv(index=False)


def prepare_rows(bundle: Dict[str, Any], raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(raw) == 0:
        raise PredictionInputError("The CSV has no rows.")
    if len(raw) > MAX_PREDICT_ROWS:
        raise PredictionInputError(f"Please upload at most {MAX_PREDICT_ROWS:,} rows at a time (this file has {len(raw):,}).")
    clean, report = normalize_missing(raw)
    missing = [c for c in bundle["feature_columns"] if c not in clean.columns]
    if missing:
        raise PredictionInputError(
            "The CSV is missing required column(s): " + ", ".join(missing)
            + ". Download the template to see the expected columns.")
    X = clean[bundle["feature_columns"]].copy()
    bad_numeric = {}
    for c in bundle["feature_columns"]:
        if c in bundle["numeric_columns"]:
            before = X[c].notna().sum()
            X[c] = pd.to_numeric(X[c], errors="coerce")
            lost = int(before - X[c].notna().sum())
            if lost:
                bad_numeric[c] = lost            # text in a numeric column: treated as missing
        else:
            X[c] = X[c].where(X[c].isna(), X[c].astype(str))
    notes = {"hidden_placeholders": int(report.n_hidden),
             "missing_cells": int(X.isna().sum().sum()),
             "non_numeric_values_treated_as_missing": bad_numeric,
             "extra_columns_ignored": [c for c in clean.columns if c not in bundle["feature_columns"]]}
    return X, notes


def predict_csv(bundle: Dict[str, Any], raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X, notes = prepare_rows(bundle, raw)
    pipe, classes = bundle["pipeline"], bundle["classes"]
    pred = pipe.predict(X)
    out = raw.copy()
    out[f"predicted_{bundle['target']}"] = [classes[int(i)] for i in pred]
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            out["confidence"] = np.round(proba.max(axis=1), 4)
            if len(classes) == 2:
                out[f"probability_{classes[1]}"] = np.round(proba[:, 1], 4)
        except Exception:
            pass
    notes["n_rows"] = int(len(out))
    notes["class_counts"] = out[f"predicted_{bundle['target']}"].value_counts().to_dict()
    return out, notes
