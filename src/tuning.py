"""
Hyperparameter tuning of the selected model with nested cross-validation.

Outer loop: the same 5 stratified folds (seed 42) that ModelSelector used, so
the tuned score can be compared fold by fold with the default score.
Inner loop: RandomizedSearchCV with 3-fold stratified CV on the training part
of each outer fold only, so the test fold never influences the chosen settings.
The tuned score is the out-of-fold score over all outer test folds, the same
definition as the scores in the improvement ladder.
"""
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, get_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.model_data import ModelData, model_pipeline, outer_cv
from src.significance import paired_significance_test

# Search spaces for each model, as Pipeline parameter names
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "logistic_regression": {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
    "random_forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 8, 16, 24],
                      "model__min_samples_leaf": [1, 2, 5], "model__max_features": ["sqrt", 0.5]},
    "gradient_boosting": {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                          "model__max_depth": [2, 3, 4], "model__subsample": [0.8, 1.0]},
    "svm": {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0], "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
    "knn": {"model__n_neighbors": [5, 10, 15, 25, 40], "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]},
    "naive_bayes": {"model__nb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]},
    "mlp": {"model__hidden_layer_sizes": [(64, 32), (128, 64), (64,), (32,)],
            "model__alpha": [1e-4, 1e-3, 1e-2], "model__learning_rate_init": [1e-3, 3e-3]},
    "voting_ensemble": {"model__lr__C": [0.1, 1.0, 10.0], "model__rf__n_estimators": [100, 300],
                        "model__rf__max_depth": [None, 16], "model__gb__learning_rate": [0.05, 0.1, 0.2],
                        "model__gb__n_estimators": [100, 200]},
}


@dataclass
class TuningResult:
    model_name: str
    metric: str
    default_score: float               # out-of-fold score with the default settings
    tuned_score: float                 # out-of-fold score with settings tuned inside each fold
    gain: float
    default_fold_scores: List[float]
    tuned_fold_scores: List[float]
    p_value: Optional[float]           # paired t-test, tuned vs default, on the outer folds
    significant: bool
    best_params_per_fold: List[Dict[str, Any]] = field(default_factory=list)
    most_common_params: Dict[str, Any] = field(default_factory=dict)
    n_iter: int = 0
    inner_folds: int = 0
    outer_folds: int = 0
    search_space: Dict[str, list] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["search_space"] = {k: [str(v) for v in vs] for k, vs in self.search_space.items()}
        return d


def _oof_score(y, pred, metric, is_binary) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    return float(f1_score(y, pred, average="binary" if is_binary else "macro", zero_division=0))


def tune_model(data: ModelData, model_name: str, n_iter: int = 10, inner_folds: int = 3,
               random_state: int = 42, n_jobs: int = -1) -> TuningResult:
    space = SEARCH_SPACES[model_name]
    n_combos = int(np.prod([len(v) for v in space.values()]))
    n_iter = min(n_iter, n_combos)
    cv = outer_cv(data, random_state=random_state)
    scorer = get_scorer(data.scorer_name)
    X, y = data.X, data.y

    pred_default = np.empty(len(y), dtype=y.dtype)
    pred_tuned = np.empty(len(y), dtype=y.dtype)
    default_scores, tuned_scores, params = [], [], []
    for tr, te in cv.split(X, y):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        default = model_pipeline(data, model_name).fit(Xtr, ytr)
        pred_default[te] = default.predict(Xte)
        default_scores.append(float(scorer(default, Xte, yte)))

        inner = StratifiedKFold(n_splits=min(inner_folds, int(np.bincount(ytr).min())), shuffle=True,
                                random_state=random_state)
        search = RandomizedSearchCV(model_pipeline(data, model_name), space, n_iter=n_iter, cv=inner,
                                    scoring=data.scorer_name, random_state=random_state, n_jobs=n_jobs,
                                    refit=True, error_score="raise")
        search.fit(Xtr, ytr)
        pred_tuned[te] = search.predict(Xte)
        tuned_scores.append(float(scorer(search, Xte, yte)))
        params.append({k.replace("model__", ""): (v if isinstance(v, (int, float, str)) or v is None else str(v))
                       for k, v in search.best_params_.items()})

    d_score = _oof_score(y, pred_default, data.metric, data.is_binary)
    t_score = _oof_score(y, pred_tuned, data.metric, data.is_binary)
    p_value, significant = None, False
    if len(tuned_scores) > 1 and np.std(np.subtract(tuned_scores, default_scores)) > 0:
        _, p_value, significant = paired_significance_test(tuned_scores, default_scores)
    common = {}
    for k in params[0]:
        common[k] = Counter(str(p[k]) for p in params).most_common(1)[0][0]
    return TuningResult(model_name=model_name, metric=data.metric, default_score=d_score, tuned_score=t_score,
                        gain=t_score - d_score, default_fold_scores=default_scores, tuned_fold_scores=tuned_scores,
                        p_value=p_value, significant=significant, best_params_per_fold=params,
                        most_common_params=common, n_iter=n_iter, inner_folds=inner_folds,
                        outer_folds=cv.get_n_splits(), search_space=space)
