"""
Model explanations for the selected model.

Global: permutation importance on a held-out split. Each original column is
shuffled in turn and the drop in the selection metric is measured, so the
importance is reported per input column (before one-hot encoding) and is the
same idea for every model family.

Local: for one row, each column is replaced in turn by a typical value from the
training split (median for numbers, most frequent value for categories) and the
change in the predicted probability of the predicted class is recorded. A large
drop means that column pushed the model towards its prediction.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from src.model_data import ModelData, model_pipeline


@dataclass
class FeatureImportance:
    feature: str
    importance: float          # mean drop in the metric when the column is shuffled
    std: float


@dataclass
class GlobalExplanation:
    model_name: str
    metric: str
    holdout_score: float
    n_train: int
    n_test: int
    n_repeats: int
    importances: List[FeatureImportance] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Contribution:
    feature: str
    value: Any
    reference: Any
    effect: float              # probability of the predicted class minus the probability after replacement


@dataclass
class LocalExplanation:
    row: int
    actual: str
    predicted: str
    confidence: Optional[float]      # probability of the predicted class, if the model gives probabilities
    score_kind: str                  # "probability" or "decision score"
    contributions: List[Contribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelExplainer:
    """Fits the selected model once on a training split and explains it."""

    def __init__(self, data: ModelData, model_name: str, test_size: float = 0.25, random_state: int = 42):
        self.data = data
        self.model_name = model_name
        idx = np.arange(len(data.y))
        tr, te = train_test_split(idx, test_size=test_size, stratify=data.y, random_state=random_state)
        self.X_train, self.X_test = data.X.iloc[tr], data.X.iloc[te]
        self.y_train, self.y_test = data.y[tr], data.y[te]
        self.pipe = model_pipeline(data, model_name).fit(self.X_train, self.y_train)
        self.reference = {
            c: (self.X_train[c].median() if pd.api.types.is_numeric_dtype(self.X_train[c])
                else (self.X_train[c].mode(dropna=True).iloc[0] if self.X_train[c].notna().any() else np.nan))
            for c in self.X_train.columns
        }

    # ---------- global ----------
    def global_importance(self, n_repeats: int = 5, random_state: int = 42) -> GlobalExplanation:
        scorer = get_scorer(self.data.scorer_name)
        holdout = float(scorer(self.pipe, self.X_test, self.y_test))
        r = permutation_importance(self.pipe, self.X_test, self.y_test, scoring=self.data.scorer_name,
                                   n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        imps = [FeatureImportance(feature=str(c), importance=float(m), std=float(s))
                for c, m, s in zip(self.X_test.columns, r.importances_mean, r.importances_std)]
        imps.sort(key=lambda f: f.importance, reverse=True)
        return GlobalExplanation(model_name=self.model_name, metric=self.data.metric, holdout_score=holdout,
                                 n_train=len(self.X_train), n_test=len(self.X_test), n_repeats=n_repeats,
                                 importances=imps)

    # ---------- local ----------
    def _scores(self, X: pd.DataFrame, cls: int):
        if hasattr(self.pipe, "predict_proba"):
            try:
                return self.pipe.predict_proba(X)[:, cls], "probability"
            except AttributeError:
                pass
        dec = self.pipe.decision_function(X)
        if dec.ndim == 1:                       # binary: positive margin means class 1
            return (dec if cls == 1 else -dec), "decision score"
        return dec[:, cls], "decision score"

    def explain_row(self, row: int, top: int = 10) -> LocalExplanation:
        if not 0 <= row < len(self.X_test):
            raise IndexError(f"Row must be between 0 and {len(self.X_test) - 1}.")
        x = self.X_test.iloc[[row]]
        pred = int(self.pipe.predict(x)[0])
        base, kind = self._scores(x, pred)
        variants = pd.concat([x] * x.shape[1], ignore_index=True)
        for i, c in enumerate(x.columns):
            variants.iat[i, i] = self.reference[c]
        changed, _ = self._scores(variants, pred)
        contribs = [Contribution(feature=str(c), value=_plain(x.iloc[0][c]), reference=_plain(self.reference[c]),
                                 effect=float(base[0] - changed[i]))
                    for i, c in enumerate(x.columns)]
        contribs.sort(key=lambda k: abs(k.effect), reverse=True)
        return LocalExplanation(row=row, actual=self.data.classes[int(self.y_test[row])],
                                predicted=self.data.classes[pred],
                                confidence=float(base[0]) if kind == "probability" else None,
                                score_kind=kind, contributions=contribs[:top])

    def holdout_rows(self, n: int = 200) -> List[Dict[str, Any]]:
        """A few held-out rows with actual and predicted class, so the user can pick one to explain."""
        preds = self.pipe.predict(self.X_test.iloc[:n])
        return [{"row": i, "actual": self.data.classes[int(self.y_test[i])], "predicted": self.data.classes[int(p)]}
                for i, p in enumerate(preds)]


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    return v if v is None or isinstance(v, (int, str, bool)) else str(v)
