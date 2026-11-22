"""
Keep-vs-drop experiment for heavily-missing columns.
Usage:
    python run_missing_experiment.py

Compares two policies on the SAME data, SAME models and SAME CV folds:
  OLD policy: drop any column with 40% or more missing
  NEW policy: keep it (R2a/R2b): median fill + "was missing" flag
Only the drop threshold changes between the two runs.

Scenario A (synthetic): the strongest feature is missing in 60% of rows, and it
            goes missing more often for one class, so the fact that it is
            missing carries information.
Scenario B (real data, breast cancer): the most informative column is made
            missing, once depending on the class and once completely at random.
Results are printed and saved to results/missing_experiment.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

import src.preprocessor as P
from src.missing import normalize_missing
from src.model_selector import build_decided_preprocessor
from src.preprocessor import PreprocessingDecider
from src.profiler import DataProfiler

warnings.filterwarnings("ignore")
SEEDS = range(5)
OLD_DROP_AT, NEW_DROP_AT = 40.0, 95.0


def cv_f1(df, target, drop_at, seed):
    P.MISSING_DROP_THRESHOLD = drop_at          # the only thing that differs between policies
    clean, _ = normalize_missing(df)
    profile = DataProfiler(clean, target).profile()
    plan = PreprocessingDecider(profile).decide()
    dropped = [d.column for d in plan.column_decisions if d.missing_action == "drop_column"]
    X, y = clean.drop(columns=[target] + dropped), clean[target].values
    scores = {}
    for name, model in [("LogisticRegression", LogisticRegression(max_iter=3000)),
                        ("RandomForest", RandomForestClassifier(100, random_state=0, n_jobs=-1))]:
        pipe = Pipeline([("prep", build_decided_preprocessor(plan, X)), ("model", model)])
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        scores[name] = f1_score(y, cross_val_predict(pipe, X, y, cv=cv))
    return scores


def synthetic(seed):
    rng = np.random.RandomState(seed)
    n = 4000
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] * 1.5 + X["f1"] * 0.5 + rng.randn(n) > 0).astype(int)
    X["city"] = rng.choice(list("abcd"), n)
    df = X.assign(target=y)
    df.loc[rng.rand(n) < np.where(y == 1, 0.80, 0.40), "f0"] = np.nan
    return df


def breast_cancer(seed, informative):
    bc = load_breast_cancer(as_frame=True)
    df = bc.data.copy()
    df["target"] = bc.target
    col = df.drop(columns="target").corrwith(df["target"]).abs().idxmax()
    rng = np.random.RandomState(seed)
    p = np.where(df["target"] == 1, 0.85, 0.15) if informative else np.full(len(df), 0.5)
    df.loc[rng.rand(len(df)) < p, col] = np.nan
    return df


SCENARIOS = {
    "A. Synthetic, missingness is informative": lambda s: synthetic(s),
    "B1. Breast cancer, missing depends on class": lambda s: breast_cancer(s, True),
    "B2. Breast cancer, missing completely at random": lambda s: breast_cancer(s, False),
}

if __name__ == "__main__":
    rows = []
    for scenario, make in SCENARIOS.items():
        for seed in SEEDS:
            df = make(seed)
            miss = df.isna().any(axis=1).mean() * 100
            old = cv_f1(df, "target", OLD_DROP_AT, seed)
            new = cv_f1(df, "target", NEW_DROP_AT, seed)
            for model in old:
                rows.append((scenario, model, seed, round(miss, 1), old[model], new[model]))
    r = pd.DataFrame(rows, columns=["scenario", "model", "seed", "rows_missing_%", "F1_old_drop", "F1_new_keep"])
    g = r.groupby(["scenario", "model"]).agg(
        rows_missing_pct=("rows_missing_%", "mean"),
        F1_old_drop=("F1_old_drop", "mean"), F1_new_keep=("F1_new_keep", "mean"),
        sd_old=("F1_old_drop", "std"), sd_new=("F1_new_keep", "std"),
    )
    g["gain"] = g["F1_new_keep"] - g["F1_old_drop"]
    print(g.round(4).to_string())
    os.makedirs("results", exist_ok=True)
    g.round(4).to_csv("results/missing_experiment.csv")
    print("\nSaved to results/missing_experiment.csv")