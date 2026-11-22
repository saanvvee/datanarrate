"""
Shared set-up for the advanced analyses (explanations, tuning, repeated-CV
significance). Each of them needs the same X, y, metric and preprocessing that
ModelSelector used, so they build it here in exactly the same way.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.model_selector import _build_candidates, _build_naive_pipeline, build_decided_preprocessor
from src.preprocessor import PreprocessingPlan
from src.profiler import DatasetProfile


@dataclass
class ModelData:
    X: pd.DataFrame            # features after the plan's dropped columns are removed
    X_full: pd.DataFrame       # all features (the fixed baseline uses these)
    y: np.ndarray              # encoded target
    classes: List[str]         # class names in encoded order
    is_binary: bool
    metric: str                # "f1" or "accuracy", as chosen by ModelSelector
    scorer_name: str           # scikit-learn scorer name for that metric
    min_class_count: int
    plan: PreprocessingPlan


def prepare_model_data(df: pd.DataFrame, profile: DatasetProfile, plan: PreprocessingPlan) -> ModelData:
    target = profile.target_column
    X_full = df.drop(columns=[target])
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    is_binary = len(le.classes_) == 2
    dropped = [d.column for d in plan.column_decisions if d.missing_action == "drop_column"]
    X = X_full.drop(columns=[c for c in dropped if c in X_full.columns])
    metric = "f1" if (profile.is_imbalanced or not is_binary) else "accuracy"
    scorer_name = "accuracy" if metric == "accuracy" else ("f1" if is_binary else "f1_macro")
    return ModelData(X=X, X_full=X_full, y=y, classes=[str(c) for c in le.classes_], is_binary=is_binary,
                     metric=metric, scorer_name=scorer_name, min_class_count=int(np.bincount(y).min()), plan=plan)


def model_pipeline(data: ModelData, model_name: str) -> Pipeline:
    """The same preprocessing + model pipeline ModelSelector cross-validates for `model_name`."""
    candidates = _build_candidates(n_rows=0, min_class_count=data.min_class_count)   # n_rows=0 keeps the SVM
    if model_name not in candidates:
        raise ValueError(f"Unknown model '{model_name}'.")
    return Pipeline([("preprocess", build_decided_preprocessor(data.plan, data.X)),
                     ("model", clone(candidates[model_name]))])


def baseline_pipeline(data: ModelData) -> Pipeline:
    return _build_naive_pipeline(data.X_full, None)


def outer_cv(data: ModelData, folds: int = 5, random_state: int = 42) -> StratifiedKFold:
    """The exact folds ModelSelector uses, so scores can be compared fold by fold."""
    return StratifiedKFold(n_splits=min(folds, data.min_class_count), shuffle=True, random_state=random_state)
