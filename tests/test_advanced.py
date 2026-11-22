"""Tests for the explanation, tuning and corrected-significance modules."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector
from src.model_data import prepare_model_data, model_pipeline, baseline_pipeline
from src.explainer import ModelExplainer
from src.tuning import tune_model, SEARCH_SPACES
from src.significance import corrected_resampled_ttest, repeated_cv_comparison


@pytest.fixture(scope="module")
def setup():
    df = load_breast_cancer(as_frame=True).frame
    profile = DataProfiler(df, target_column="target").profile()
    plan = PreprocessingDecider(profile).decide()
    return df, profile, plan, prepare_model_data(df, profile, plan)


def test_corrected_ttest_matches_formula_and_is_more_conservative():
    a = [0.80, 0.82, 0.81, 0.83, 0.82, 0.81, 0.84, 0.80, 0.82, 0.83]
    b = [0.79, 0.80, 0.80, 0.81, 0.80, 0.80, 0.82, 0.79, 0.80, 0.81]
    t, p, _ = corrected_resampled_ttest(a, b, 0.25)
    d = np.subtract(a, b)
    assert t == pytest.approx(d.mean() / np.sqrt((1 / 10 + 0.25) * d.var(ddof=1)))
    from scipy import stats
    assert p > stats.ttest_rel(a, b).pvalue          # the correction never makes a result more significant


def test_corrected_ttest_identical_scores_is_undefined():
    t, p, sig = corrected_resampled_ttest([1, 1, 1], [1, 1, 1], 0.25)
    assert np.isnan(t) and np.isnan(p) and not sig


def test_model_pipeline_reproduces_selector_score(setup):
    df, profile, plan, data = setup
    res = ModelSelector(df, profile, plan).select()
    t = tune_model(data, "logistic_regression", n_iter=2, n_jobs=1)
    lr = next(r for r in res.candidate_results if r.model_name == "logistic_regression")
    assert t.default_score == pytest.approx(getattr(lr, res.metric_used))
    assert len(t.best_params_per_fold) == 5 and "C" in t.most_common_params


def test_every_model_has_a_search_space():
    from src.model_selector import _build_candidates
    assert set(_build_candidates(0, 100)) == set(SEARCH_SPACES)


def test_global_importance_covers_every_column(setup):
    _, _, _, data = setup
    ex = ModelExplainer(data, "logistic_regression")
    g = ex.global_importance(n_repeats=2)
    assert len(g.importances) == data.X.shape[1]
    assert g.n_test == len(ex.X_test) and 0 < g.holdout_score <= 1


def test_local_explanation_and_bounds(setup):
    _, _, _, data = setup
    ex = ModelExplainer(data, "random_forest")
    loc = ex.explain_row(0, top=5)
    assert len(loc.contributions) == 5 and loc.score_kind == "probability"
    assert abs(loc.contributions[0].effect) >= abs(loc.contributions[-1].effect)
    with pytest.raises(IndexError):
        ex.explain_row(len(ex.X_test))


def test_local_explanation_without_probabilities(setup):
    _, _, _, data = setup
    loc = ModelExplainer(data, "svm").explain_row(1)
    assert loc.score_kind == "decision score" and loc.confidence is None


def test_repeated_cv_comparison(setup):
    _, _, _, data = setup
    pipes = {"logistic_regression": (model_pipeline(data, "logistic_regression"), data.X),
             "knn": (model_pipeline(data, "knn"), data.X), "baseline": (baseline_pipeline(data), data.X_full)}
    r = repeated_cv_comparison(pipes, data.y, data.scorer_name, data.metric, "logistic_regression", "knn",
                               n_repeats=2, n_jobs=1)
    assert all(len(v) == 10 for v in r.scores.values())
    assert [(t.a, t.b) for t in r.tests] == [("logistic_regression", "knn"), ("logistic_regression", "baseline")]
    for t in r.tests:
        if not np.isnan(t.corrected_p):
            assert t.corrected_p >= t.plain_p
