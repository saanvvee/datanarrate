"""
Statistical significance testing — checks whether the winning model's edge
over the runner-up is real or could plausibly be noise, using per-fold
cross-validation scores and a paired t-test.

The plain paired t-test treats the fold scores as independent, but the training
sets of different folds overlap, so it can overstate significance (Dietterich,
1998). corrected_resampled_ttest applies the Nadeau and Bengio (2003)
correction, and repeated_cv_comparison produces the repeated cross-validation
scores it is meant for.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def paired_significance_test(fold_scores_a: List[float], fold_scores_b: List[float], alpha: float = 0.05) -> Tuple[float, float, bool]:
    """Returns (t_statistic, p_value, is_significant) for two models' per-fold scores."""
    t_stat, p_value = stats.ttest_rel(fold_scores_a, fold_scores_b)
    return float(t_stat), float(p_value), bool(p_value < alpha)


def corrected_resampled_ttest(scores_a: List[float], scores_b: List[float], test_train_ratio: float,
                              alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Nadeau-Bengio corrected resampled t-test. The variance of the score
    differences is inflated by (1/J + n_test/n_train) instead of 1/J, which
    accounts for the overlap between training sets. Returns (t, p, significant);
    t and p are nan when every difference is identical.
    """
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    J = len(d)
    var = d.var(ddof=1)
    if J < 2 or var == 0:
        return float("nan"), float("nan"), False
    t = d.mean() / np.sqrt((1.0 / J + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p), bool(p < alpha)


@dataclass
class PairTest:
    a: str
    b: str
    mean_difference: float
    plain_t: float
    plain_p: float
    corrected_t: float
    corrected_p: float
    significant_plain: bool
    significant_corrected: bool


@dataclass
class RepeatedCVResult:
    metric: str
    n_splits: int
    n_repeats: int
    scores: Dict[str, List[float]] = field(default_factory=dict)      # per model, one score per fold and repeat
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    tests: List[PairTest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pair(name_a, sa, name_b, sb, ratio) -> PairTest:
    if np.allclose(sa, sb):
        pt, pp, ps = float("nan"), float("nan"), False
    else:
        pt, pp, ps = paired_significance_test(sa, sb)
    ct, cp, cs = corrected_resampled_ttest(sa, sb, ratio)
    return PairTest(a=name_a, b=name_b, mean_difference=float(np.mean(np.subtract(sa, sb))),
                    plain_t=pt, plain_p=pp, corrected_t=ct, corrected_p=cp,
                    significant_plain=ps, significant_corrected=cs)


def repeated_cv_comparison(pipelines: Dict[str, Tuple[Any, Any]], y, scorer_name: str, metric: str,
                           winner: str, runner_up: Optional[str], baseline: str = "baseline",
                           n_splits: int = 5, n_repeats: int = 3, random_state: int = 42,
                           n_jobs: int = -1) -> RepeatedCVResult:
    """
    pipelines maps a name to (pipeline, X). Every pipeline is scored on the same
    repeated stratified folds, then the winner is tested against the runner-up
    and against the baseline with both the plain and the corrected paired t-test.
    """
    n_splits = min(n_splits, int(np.bincount(y).min()))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    out = RepeatedCVResult(metric=metric, n_splits=n_splits, n_repeats=n_repeats)
    for name, (pipe, X) in pipelines.items():
        s = cross_val_score(pipe, X, y, cv=cv, scoring=scorer_name, n_jobs=n_jobs)
        out.scores[name] = [float(v) for v in s]
        out.mean[name] = float(np.mean(s))
        out.std[name] = float(np.std(s, ddof=1))
    ratio = 1.0 / (n_splits - 1)          # n_test / n_train for k-fold
    if runner_up and runner_up in out.scores:
        out.tests.append(_pair(winner, out.scores[winner], runner_up, out.scores[runner_up], ratio))
    if baseline in out.scores:
        out.tests.append(_pair(winner, out.scores[winner], baseline, out.scores[baseline], ratio))
    return out
