"""
Builds the results table for the presentation.
Usage:
    python make_results_table.py

For each dataset: normalise missing values (R0), prepare the target, profile,
decide preprocessing, run the baseline and the 8 models, and run the paired
t-test between the top two models. No narration (no API key needed).
Output: printed table + results/results_table.csv
Add a dataset by adding a line to DATASETS. Big files are sampled to MAX_ROWS.
"""
import os
import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

from src.missing import normalize_missing
from src.model_selector import ModelSelector
from src.preprocessor import PreprocessingDecider
from src.profiler import DataProfiler
from src.significance import paired_significance_test
from src.target_preparer import prepare_target

warnings.filterwarnings("ignore")
MAX_ROWS = int(os.environ.get("MAX_ROWS", 5000))


def sk(loader):
    def go():
        d = loader(as_frame=True)
        return d.frame.rename(columns={"target": "target"})
    return go


DATASETS = {
    "Adult income": (lambda: pd.read_csv("adult_income.csv"), "class"),
    "Mushroom": (lambda: pd.read_csv("mushroom.csv"), "target"),
    "Breast cancer": (sk(load_breast_cancer), "target"),
    "Wine": (sk(load_wine), "target"),
}


def run_one(name, loader, target):
    raw = loader()
    df, miss = normalize_missing(raw)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
    df, _ = prepare_target(df, target)
    profile = DataProfiler(df, target_column=target).profile()
    plan = PreprocessingDecider(profile).decide()
    result = ModelSelector(df, profile, plan).select()
    ranked = sorted(result.candidate_results,
                    key=lambda r: getattr(r, result.metric_used), reverse=True)
    p = float("nan")
    if len(ranked) > 1:
        _, p, _ = paired_significance_test(result.fold_scores[ranked[0].model_name],
                                           result.fold_scores[ranked[1].model_name])
    ladder = {s.label: s.score for s in result.ladder}
    return {
        "Dataset": name, "Rows used": len(df), "Missing %": miss.pct_missing,
        "Metric": result.metric_used,
        "Baseline": round(result.baseline_score, 4),
        "+ class balancing": round(ladder.get("+ class balancing", float("nan")), 4),
        "+ DataNarrate preprocessing": round(ladder.get("+ DataNarrate preprocessing", float("nan")), 4),
        "Best model": result.best_model_name,
        "DataNarrate": round(result.best_score, 4),
        "Improvement": round(result.improvement_over_baseline, 4),
        "p (top 2 models)": round(p, 4),
    }


if __name__ == "__main__":
    rows = []
    for name, (loader, target) in DATASETS.items():
        print(f"Running {name} ...", flush=True)
        try:
            rows.append(run_one(name, loader, target))
        except Exception as e:
            print(f"  skipped {name}: {type(e).__name__}: {e}")
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    table.to_csv("results/results_table.csv", index=False)
    print("\nSaved to results/results_table.csv")