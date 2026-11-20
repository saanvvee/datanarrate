import numpy as np
import pandas as pd

from src.missing import normalize_missing, severity
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import build_decided_preprocessor


def kaggle_style_df():
    return pd.DataFrame({
        "workclass": ["Private", "?", "Private", " ?", "State-gov", "Private", "?", "Private", "Private", "Private"],
        "age": [25, 38, 28, 44, 18, 34, 29, 63, 24, 55],
        "label": [0, 1] * 5,
    })


def test_question_marks_become_real_missing_values():
    clean, rep = normalize_missing(kaggle_style_df())
    assert clean["workclass"].isna().sum() == 3        # '?' and ' ?' both found
    assert rep.n_hidden == 3 and rep.n_empty == 0
    assert rep.pct_missing == 10.0                      # 3 of 30 cells


def test_the_old_check_reports_zero_but_the_new_one_does_not():
    df = kaggle_style_df()
    assert df.isna().sum().sum() == 0                   # what the app used to show
    _, rep = normalize_missing(df)
    assert rep.pct_missing > 0


def test_input_is_not_modified():
    df = kaggle_style_df()
    normalize_missing(df)
    assert (df["workclass"] == "?").sum() == 2


def test_real_categories_are_left_alone():
    df = pd.DataFrame({"c": ["a", "b", "None", "Unknown", "a", "b", "a", "b", "a", "b"],
                       "label": [0, 1] * 5})
    clean, rep = normalize_missing(df)
    assert clean["c"].isna().sum() == 0                 # 'None' / 'Unknown' might be real: only warned about
    assert len(rep.warnings) == 2


def test_text_numbers_with_blank_placeholders_become_numeric():
    df = pd.DataFrame({"total": ["10.5", " ", "20", "3", "4", "5", "6", "7", "8", "9"], "label": [0, 1] * 5})
    clean, rep = normalize_missing(df)
    assert pd.api.types.is_numeric_dtype(clean["total"])
    assert rep.converted_to_numeric == ["total"]
    assert clean["total"].isna().sum() == 1


def test_sentinel_number_is_reported_not_changed():
    df = pd.DataFrame({"age": [30] * 90 + [9999] * 10, "label": [0, 1] * 50})
    clean, rep = normalize_missing(df)
    assert (clean["age"] == 9999).sum() == 10
    assert any("9999" in w for w in rep.warnings)


def test_empty_cells_and_placeholders_are_counted_separately():
    df = pd.DataFrame({"c": ["a", None, "?", "b"], "label": [0, 1, 0, 1]})
    _, rep = normalize_missing(df)
    assert rep.n_empty == 1 and rep.n_hidden == 1 and rep.n_missing == 2


def test_clean_data_reports_zero():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    clean, rep = normalize_missing(df)
    assert rep.pct_missing == 0.0 and rep.rows_with_missing == 0


def test_severity_bands():
    assert severity(0) == "None" and severity(3) == "Low" and severity(10) == "Moderate"
    assert severity(30) == "High" and severity(40) == "Severe"


def test_heavily_missing_columns_flow_through_the_pipeline_without_nan():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "num_60": np.where(rng.rand(n) < 0.6, np.nan, rng.randn(n)),
        "cat_50": np.where(rng.rand(n) < 0.5, "?", rng.choice(list("abc"), n)),
        "label": rng.choice([0, 1], n),
    })
    clean, _ = normalize_missing(df)
    profile = DataProfiler(clean, target_column="label").profile()
    plan = PreprocessingDecider(profile).decide()
    X = clean.drop(columns=["label"])
    out = build_decided_preprocessor(plan, X).fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert not np.isnan(out).any()
    # num_60 (scaled) + its flag + cat_50 one-hot (a, b, c, Missing)
    assert out.shape[1] == 1 + 1 + 4