"""
DataNarrate web app: FastAPI backend.

Serves the single-page frontend in web/static and exposes a small JSON API
over the existing pipeline in src/. Nothing in src/ is changed: the API calls
the same modules the Streamlit app and the CLI use.

Run locally:
    uvicorn web.server:app --reload
then open http://127.0.0.1:8000

API
    POST /api/datasets               upload a CSV -> dataset summary + missing-data report
    POST /api/datasets/sample/{name} load a bundled demo dataset (adult, mushroom)
    POST /api/runs                   start a pipeline run -> job id
    GET  /api/jobs/{job_id}          job status, stage progress and, when done, the full result
    POST /api/jobs/{job_id}/ablation run the ablation study for a finished job
    POST /api/jobs/{job_id}/tasks/{kind}   start an extra analysis (explain, tuning, stats) in the background
    GET  /api/jobs/{job_id}/tasks/{kind}   its status and, when done, its result
    POST /api/jobs/{job_id}/explain/row    explain one held-out row (after the explain task)
    GET  /api/history                past runs from SQLite
    GET  /api/health                 liveness check
"""
import io
import math
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.missing import normalize_missing
from src.target_preparer import prepare_target, InvalidTargetError
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider, MISSING_HIGH_THRESHOLD, MISSING_DROP_THRESHOLD
from src.preprocessing_proof import build_preprocessing_proof
from src.model_selector import ModelSelector, InsufficientDataError
from src.significance import paired_significance_test
from src.outlier import outlier_rows
from src.ablation import run_ablation
from src.model_data import prepare_model_data, model_pipeline, baseline_pipeline
from src.explainer import ModelExplainer
from src.tuning import tune_model
from src.significance import repeated_cv_comparison
from src.storage import RunStorage

STATIC_DIR = Path(__file__).parent / "static"
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("DATANARRATE_DB", "datanarrate.db")
SAMPLES = {   # bundled demo datasets: (file in the repo, suggested target)
    "adult": ("adult_income.csv", "class"),
    "mushroom": ("mushroom.csv", "target"),
}
OUTLIER_SAMPLE = 1500            # values sent per clipped column for the box plots
FAST_MODE_ROWS = int(os.environ.get("DATANARRATE_FAST_ROWS", 10_000))   # same default as the Streamlit app
MAX_UPLOAD_MB = int(os.environ.get("DATANARRATE_MAX_UPLOAD_MB", 200))
MAX_DATASETS = 8                 # uploaded datasets kept in memory
PREVIEW_ROWS = 20

app = FastAPI(title="DataNarrate", version="1.0")

_datasets: Dict[str, Dict[str, Any]] = {}
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)   # one heavy run at a time
_task_executor = ThreadPoolExecutor(max_workers=1)   # extra analyses, one at a time
TASK_KINDS = ("explain", "tuning", "stats")
TUNING_ITER = int(os.environ.get("DATANARRATE_TUNING_ITER", 10))
STATS_REPEATS = int(os.environ.get("DATANARRATE_STATS_REPEATS", 3))


# ---------------------------------------------------------------- JSON helpers
def clean(obj: Any) -> Any:
    """Makes pipeline output JSON-safe: dataclasses, numpy types, NaN/inf -> None."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return clean(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict(orient="records"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def preview_rows(df: pd.DataFrame, n: int = PREVIEW_ROWS):
    head = df.head(n).astype(object).where(df.head(n).notna(), None)
    return clean(head.to_dict(orient="records"))


# ---------------------------------------------------------------- datasets
@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(400, f"Could not read this CSV: {e}")
    return _register(raw, file.filename)


@app.post("/api/datasets/sample/{name}")
def load_sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404, f"Unknown sample. Choose one of: {', '.join(SAMPLES)}")
    filename, target = SAMPLES[name]
    raw = pd.read_csv(ROOT_DIR / filename)
    out = _register(raw, filename)
    out["suggested_target"] = target
    return out


def _register(raw: pd.DataFrame, filename: str):
    if raw.shape[1] < 2 or raw.shape[0] < 10:
        raise HTTPException(400, "The file needs at least 2 columns and 10 rows.")

    df, report = normalize_missing(raw)   # rule R0
    dataset_id = uuid.uuid4().hex[:12]
    with _lock:
        if len(_datasets) >= MAX_DATASETS:
            _datasets.pop(next(iter(_datasets)))
        _datasets[dataset_id] = {"df": df, "name": filename}

    table = report.table
    return clean({
        "dataset_id": dataset_id,
        "name": filename,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [{"name": c, "dtype": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text",
                     "n_unique": int(df[c].nunique(dropna=True))} for c in df.columns],
        "suggested_target": df.columns[-1],
        "fast_mode_available": df.shape[0] > FAST_MODE_ROWS,
        "fast_mode_rows": FAST_MODE_ROWS,
        "preview": preview_rows(df),
        "missing": {
            "pct_cells": report.pct_missing,
            "pct_rows": report.pct_rows_with_missing,
            "n_missing": report.n_missing,
            "n_hidden": report.n_hidden,
            "summary": report.summary_lines(),
            "warnings": report.warnings,
            "table": table[table["Total missing"] > 0].sort_values("Missing %", ascending=False),
            "high_threshold": MISSING_HIGH_THRESHOLD,
        },
    })


# ---------------------------------------------------------------- runs
class RunRequest(BaseModel):
    dataset_id: str
    target: str
    fast_mode: bool = True


STAGES = [
    ("target", "Preparing the target column"),
    ("profile", "Profiling every column"),
    ("decide", "Deciding preprocessing (rules R1 to R8)"),
    ("proof", "Verifying each step on your data"),
    ("models", "Cross-validating 8 models and the baseline"),
    ("narrate", "Writing the narration"),
    ("save", "Saving the run"),
]

MISSING_ACTION_TEXT = {
    "none": "Nothing missing", "impute_median": "Filled with the median",
    "impute_mode": "Filled with the most frequent value", "impute_constant": "Filled with a 'Missing' category",
    "drop_column": "Dropped",
}


def _set(job_id: str, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def _stage(job_id: str, key: str):
    with _lock:
        job = _jobs[job_id]
        job["stage"] = key
        job["stage_started"] = time.time()
        job["log"].append({"stage": key, "t": round(time.time() - job["started"], 1)})


def _proof_payload(proof):
    if proof is None:
        return None
    cols = []
    for p in proof.columns:
        cols.append({
            "column": p.column, "actions": p.actions, "reasons": p.reasons,
            "before": p.before, "after": p.after,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in p.checks],
            "examples": p.examples, "warnings": p.warnings, "all_passed": p.all_passed,
        })
    return {
        "columns": cols, "n_rows": proof.n_rows, "n_features_before": proof.n_features_before,
        "n_features_after": proof.n_features_after, "nan_cells_after": proof.nan_cells_after,
        "full_check_passed": proof.full_check_passed, "notes": proof.notes,
        "n_checks": sum(len(c["checks"]) for c in cols),
        "n_passed": sum(1 for c in cols for ch in c["checks"] if ch["ok"]),
    }


def _missing_handling(profile, plan):
    pct = {c.name: c.missing_pct for c in profile.columns}
    rows = []
    for d in plan.column_decisions:
        if pct.get(d.column, 0) == 0 and d.missing_action != "drop_column":
            continue
        rule = next((r.split(":")[0] for r in d.reasons if r[:2] in ("R1", "R2", "R3", "R4")), "")
        rows.append({"column": d.column, "missing_pct": pct.get(d.column, 0),
                     "decision": MISSING_ACTION_TEXT.get(d.missing_action, d.missing_action),
                     "flag": d.missing_indicator, "rule": rule})
    return sorted(rows, key=lambda r: -r["missing_pct"])


def _run_pipeline(job_id: str, df: pd.DataFrame, dataset_name: str, target: str, fast_mode: bool):
    try:
        sampled = fast_mode and len(df) > FAST_MODE_ROWS
        run_df = df.sample(n=FAST_MODE_ROWS, random_state=42) if sampled else df

        _stage(job_id, "target")
        run_df, target_notes = prepare_target(run_df, target)

        _stage(job_id, "profile")
        profile = DataProfiler(run_df, target_column=target).profile()

        _stage(job_id, "decide")
        plan = PreprocessingDecider(profile).decide()
        plan.summary = [f"Target preparation: {n}" for n in target_notes] + plan.summary
        plan.step_trace = [f"[target: {target}] {n}" for n in target_notes] + plan.step_trace

        _stage(job_id, "proof")
        try:
            proof = build_preprocessing_proof(run_df, profile, plan)
            proof_error = None
        except Exception as e:   # the proof is evidence only; never block the run
            proof, proof_error = None, f"{type(e).__name__}: {e}"

        _stage(job_id, "models")
        result = ModelSelector(run_df, profile, plan).select()

        ranked = sorted(result.candidate_results, key=lambda r: getattr(r, result.metric_used), reverse=True)
        significance = None
        if len(ranked) > 1:
            t, p, sig = paired_significance_test(result.fold_scores[ranked[0].model_name],
                                                 result.fold_scores[ranked[1].model_name])
            significance = {"a": ranked[0].model_name, "b": ranked[1].model_name, "t": t, "p": p, "significant": sig}

        _stage(job_id, "narrate")
        narration, narration_error = None, None
        try:
            from src.narrator import Narrator   # imported lazily: needs groq + GROQ_API_KEY
            narration = Narrator().narrate(profile, plan, result)
        except Exception as e:
            narration_error = f"{type(e).__name__}: {e}"

        _stage(job_id, "save")
        run_id = None
        try:
            label = f"{dataset_name} -> {target}" + (f" ({FAST_MODE_ROWS // 1000}k sample)" if sampled else "")
            run_id = RunStorage(DB_PATH).save_run(label, profile, plan, result,
                                           narration or "Narration unavailable for this run.")
        except Exception:
            pass

        outlier_values = {}
        for row in outlier_rows(profile, plan):
            if row["action"] == "clip" and len(outlier_values) < 8:
                vals = pd.to_numeric(run_df[row["column"]], errors="coerce").dropna()
                if len(vals) > OUTLIER_SAMPLE:
                    vals = vals.sample(n=OUTLIER_SAMPLE, random_state=0)
                outlier_values[row["column"]] = vals.round(4).tolist()

        kept_high = [d.column for d in plan.column_decisions if any(r.startswith(("R2a", "R2b")) for r in d.reasons)]
        payload = {
            "dataset_name": dataset_name, "target": target, "sampled": sampled, "n_rows_used": len(run_df),
            "target_notes": target_notes,
            "profile": profile.to_dict(),
            "plan": plan.to_dict(),
            "kept_high": kept_high,
            "thresholds": {"high": MISSING_HIGH_THRESHOLD, "drop": MISSING_DROP_THRESHOLD},
            "missing_handling": _missing_handling(profile, plan),
            "outliers": outlier_rows(profile, plan),
            "outlier_values": outlier_values,
            "proof": _proof_payload(proof), "proof_error": proof_error,
            "result": result.to_dict(),
            "significance": significance,
            "narration": narration, "narration_error": narration_error,
            "run_id": run_id,
        }
        with _lock:
            _jobs[job_id]["context"] = {"df": run_df, "profile": profile, "plan": plan, "result": result}
        _set(job_id, status="done", stage="done", result=clean(payload), finished=time.time())
    except (InvalidTargetError, InsufficientDataError) as e:
        _set(job_id, status="error", error=str(e), finished=time.time())
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=f"Unexpected error: {type(e).__name__}: {e}", finished=time.time())


@app.post("/api/runs")
def start_run(req: RunRequest):
    with _lock:
        ds = _datasets.get(req.dataset_id)
    if ds is None:
        raise HTTPException(404, "This dataset is no longer in memory. Please upload it again.")
    if req.target not in ds["df"].columns:
        raise HTTPException(400, f"'{req.target}' is not a column of this dataset.")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "running", "stage": "queued", "started": time.time(), "log": [],
                         "target": req.target, "dataset_name": ds["name"]}
    _executor.submit(_run_pipeline, job_id, ds["df"], ds["name"], req.target, req.fast_mode)
    return {"job_id": job_id, "stages": [{"key": k, "label": v} for k, v in STAGES]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        out = {k: v for k, v in job.items() if k not in ("context", "tasks")}
    out["elapsed"] = round((out.get("finished") or time.time()) - out["started"], 1)
    return out


@app.post("/api/jobs/{job_id}/ablation")
def ablation(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        ctx = job.get("context") if job else None
    if ctx is None:
        raise HTTPException(404, "Run the pipeline first; this job has no finished result.")
    result = ctx["result"]
    started = time.time()
    rows = run_ablation(ctx["df"], ctx["profile"], ctx["plan"], result.best_model_name,
                        result.metric_used, cv_folds=result.cv_folds_used)
    return clean({"model": result.best_model_name, "metric": result.metric_used,
                  "seconds": round(time.time() - started, 1), "variants": rows})


def _run_task(job_id: str, kind: str):
    with _lock:
        ctx = _jobs[job_id]["context"]
        task = _jobs[job_id]["tasks"][kind]
    try:
        result, data = ctx["result"], prepare_model_data(ctx["df"], ctx["profile"], ctx["plan"])
        if kind == "explain":
            ex = ModelExplainer(data, result.best_model_name)
            out = {"global": ex.global_importance().to_dict(), "rows": ex.holdout_rows()}
            first = ex.explain_row(0).to_dict()
            out["local"] = first
            with _lock:
                ctx["explainer"] = ex
        elif kind == "tuning":
            out = tune_model(data, result.best_model_name, n_iter=TUNING_ITER).to_dict()
            out["ladder_before"] = result.best_score
        else:
            ranked = sorted(result.candidate_results, key=lambda r: getattr(r, result.metric_used), reverse=True)
            winner = ranked[0].model_name
            runner = ranked[1].model_name if len(ranked) > 1 else None
            pipes = {winner: (model_pipeline(data, winner), data.X), "baseline": (baseline_pipeline(data), data.X_full)}
            if runner:
                pipes[runner] = (model_pipeline(data, runner), data.X)
            out = repeated_cv_comparison(pipes, data.y, data.scorer_name, data.metric, winner, runner,
                                         n_repeats=STATS_REPEATS).to_dict()
        task.update(status="done", result=clean(out), finished=time.time())
    except Exception as e:
        traceback.print_exc()
        task.update(status="error", error=f"{type(e).__name__}: {e}", finished=time.time())


def _job_context(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        ctx = job.get("context") if job else None
    if ctx is None:
        raise HTTPException(404, "Run the pipeline first; this job has no finished result.")
    return job, ctx


@app.post("/api/jobs/{job_id}/tasks/{kind}")
def start_task(job_id: str, kind: str):
    if kind not in TASK_KINDS:
        raise HTTPException(404, f"Unknown analysis '{kind}'.")
    job, _ = _job_context(job_id)
    with _lock:
        tasks = job.setdefault("tasks", {})
        if tasks.get(kind, {}).get("status") == "running":
            return {"status": "running"}
        tasks[kind] = {"status": "running", "started": time.time()}
    _task_executor.submit(_run_task, job_id, kind)
    return {"status": "running"}


@app.get("/api/jobs/{job_id}/tasks/{kind}")
def task_status(job_id: str, kind: str):
    with _lock:
        job = _jobs.get(job_id)
        task = dict((job or {}).get("tasks", {}).get(kind) or {})
    if not task:
        raise HTTPException(404, "This analysis has not been started.")
    task["elapsed"] = round((task.get("finished") or time.time()) - task["started"], 1)
    return task


class RowRequest(BaseModel):
    row: int


@app.post("/api/jobs/{job_id}/explain/row")
def explain_row(job_id: str, req: RowRequest):
    _, ctx = _job_context(job_id)
    ex = ctx.get("explainer")
    if ex is None:
        raise HTTPException(400, "Compute the feature importance first.")
    try:
        return clean(ex.explain_row(req.row).to_dict())
    except IndexError as e:
        raise HTTPException(400, str(e))


@app.get("/api/history")
def history():
    try:
        runs = RunStorage(DB_PATH).list_runs()
    except Exception:
        runs = []
    return clean([vars(r) for r in runs])


@app.get("/vendor/plotly.min.js")
def plotly_js():
    """Plotly.js from the installed plotly package, so charts work without internet."""
    import plotly
    return FileResponse(Path(plotly.__file__).parent / "package_data" / "plotly.min.js",
                        media_type="application/javascript")


@app.get("/api/health")
def health():
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000)
