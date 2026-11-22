from src.fact_checker import check_narration


def facts():
    models = [
        {"model_name": "voting_ensemble", "accuracy": 0.8220, "precision": 0.6281, "recall": 0.6775, "f1": 0.6519, "roc_auc": 0.8847},
        {"model_name": "logistic_regression", "accuracy": 0.7893, "precision": 0.5497, "recall": 0.7940, "f1": 0.6497, "roc_auc": 0.8831},
        {"model_name": "random_forest", "accuracy": 0.8080, "precision": 0.6010, "recall": 0.6531, "f1": 0.6260, "roc_auc": 0.8651},
    ]
    return {
        "selection": {"metric_used": "f1", "best_model_name": "voting_ensemble", "best_score": 0.6519,
                      "baseline_score": 0.6295, "improvement_over_baseline": 0.0224,
                      "candidate_results": models,
                      "ladder": [{"label": "Fixed baseline", "score": 0.6295},
                                 {"label": "+ class balancing", "score": 0.6326}]},
        "missing_report": {"pct_missing": 0.88, "n_missing": 198},
        "missing_handling": [{"column": "workclass", "missing_pct": 5.93}],
        "profile": {"n_rows": 1500, "n_cols": 15},
        "preprocessing": {"column_decisions": [
            {"column": "workclass", "missing_action": "impute_mode"},
            {"column": "id", "missing_action": "drop_column"}]},
    }


def kinds(report):
    return [f["type"] for f in report["findings"]]


def test_correct_narration_has_no_flags():
    text = ("The dataset has 1,500 rows. The voting ensemble was selected with an f1 of 0.6519, "
            "compared with the baseline of 0.6295. About 0.88% of cells were missing.")
    r = check_narration(text, facts())
    assert r["summary"]["claims_flagged"] == 0
    assert r["summary"]["numbers_checked"] >= 3
    assert r["summary"]["numbers_verified"] == r["summary"]["numbers_checked"]


def test_invented_number_is_flagged():
    r = check_narration("The baseline scored 0.7123 on this data.", facts())
    assert "number" in kinds(r)


def test_percent_form_of_a_score_is_accepted():
    r = check_narration("The baseline reached 62.95% on the same metric.", facts())
    assert r["summary"]["claims_flagged"] == 0


def test_wrong_winner_is_flagged():
    r = check_narration("The random forest was selected as the best model.", facts())
    assert "winner" in kinds(r)


def test_correct_winner_is_not_flagged():
    r = check_narration("The voting ensemble was selected as the best model.", facts())
    assert "winner" not in kinds(r)


def test_wrong_comparison_is_flagged():
    r = check_narration("Random forest outperformed logistic regression.", facts())   # f1: 0.6260 < 0.6497
    assert "comparison" in kinds(r)


def test_right_comparison_passes():
    r = check_narration("Logistic regression outperformed random forest.", facts())
    assert "comparison" not in kinds(r)


def test_wrong_metric_value_for_a_model_is_flagged():
    r = check_narration("Logistic regression reached an accuracy of 0.8500.", facts())
    assert "metric" in kinds(r)


def test_baseline_described_as_majority_class_is_flagged():
    r = check_narration("The baseline always predicts the majority class.", facts())
    assert "baseline" in kinds(r)


def test_dropped_column_claim():
    bad = check_narration("The workclass column was dropped.", facts())
    good = check_narration("The id column was dropped.", facts())
    assert "column" in kinds(bad)
    assert "column" not in kinds(good)


def test_small_counts_and_ordinals_are_ignored():
    r = check_narration("Eight models and 5 folds were used; the 1st and 99th percentiles set the limits.", facts())
    assert r["summary"]["numbers_checked"] == 0


def test_empty_text():
    assert check_narration("", facts())["summary"]["claims_flagged"] == 0
