"""
DataNarrate REST API
A thin layer on top of the existing src/ pipeline. The Streamlit app and this
API use exactly the same code, so results are identical.

Run it:
    uvicorn api.main:app --reload
Web console:   http://127.0.0.1:8000/
API docs:      http://127.0.0.1:8000/docs

Flow:
    POST /datasets            upload a CSV      -> dataset_id + missing-data report
    GET  /datasets/{id}/missing-report
    POST /runs                start a run       -> run_id (runs in the background)
    GET  /runs/{id}           poll status       -> queued / running / done / failed
    GET  /runs/{id}/result    full results once the run is done
    GET  /history             past runs saved in the database
"""
import io
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.missing import normalize_missing
from src.model_selector import InsufficientDataError, ModelSelector
from src.preprocessor import PreprocessingDecider
from src.profiler import DataProfiler
from src.storage import RunStorage
from src.target_preparer import InvalidTargetError, prepare_target

MAX_UPLOAD_MB = 50
FAST_MODE_ROWS = 10000

app = FastAPI(
    title="DataNarrate API",
    description="Upload a CSV, get an accurate missing-data report, run the transparent "
                "preprocessing + model comparison pipeline, and fetch the results.",
    version="1.0.0",
    openapi_tags=[
        {"name": "system", "description": "Health check"},
        {"name": "datasets", "description": "Upload a CSV and get an accurate missing-data report "
                                            "(rule R0 converts hidden placeholders such as `?`)"},
        {"name": "runs", "description": "Start a run in the background, poll its status, "
                                        "fetch results and history"},
    ],
    swagger_ui_parameters={"docExpansion": "list", "defaultModelsExpandDepth": -1,
                           "displayRequestDuration": True, "tryItOutEnabled": True},
)

# one worker: a run trains 8 models, so runs are queued instead of fighting for the CPU
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


# ---------- small helpers ----------

def _data_dir() -> Path:
    base = Path(os.environ.get("DATANARRATE_DATA_DIR", "api_data"))
    (base / "datasets").mkdir(parents=True, exist_ok=True)
    (base / "runs").mkdir(parents=True, exist_ok=True)
    return base


def _storage() -> RunStorage:
    return RunStorage(os.environ.get("DATANARRATE_DB", "datanarrate.db"))


def _clean(obj: Any) -> Any:
    """Makes any result JSON-safe: numpy numbers become Python numbers, NaN/inf become null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _dataset_csv(dataset_id: str) -> Path:
    if not dataset_id.isalnum():
        raise HTTPException(404, "Unknown dataset_id")
    path = _data_dir() / "datasets" / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(404, "Unknown dataset_id")
    return path


def _dataset_name(dataset_id: str) -> str:
    try:
        meta = json.loads((_data_dir() / "datasets" / f"{dataset_id}.json").read_text())
        return meta.get("filename") or dataset_id
    except Exception:
        return dataset_id


def _report_to_dict(report) -> Dict[str, Any]:
    return {
        "n_cells": report.n_cells,
        "n_missing": report.n_missing,
        "pct_missing": report.pct_missing,
        "n_empty_cells": report.n_empty,
        "n_hidden_placeholders": report.n_hidden,
        "rows_with_missing": report.rows_with_missing,
        "pct_rows_with_missing": report.pct_rows_with_missing,
        "summary": report.summary_lines(),
        "warnings": report.warnings,
        "columns": report.table.to_dict("records"),
    }


def _missing_handling_rows(profile, plan) -> List[Dict[str, Any]]:
    pct = {c.name: c.missing_pct for c in profile.columns}
    rows = []
    for d in plan.column_decisions:
        if pct.get(d.column, 0) == 0 and d.missing_action != "drop_column":
            continue
        rule = next((r.split(":")[0] for r in d.reasons if r[:2] in ("R1", "R2", "R3", "R4")), "")
        rows.append({
            "column": d.column, "missing_pct": pct.get(d.column, 0),
            "action": d.missing_action, "was_missing_flag": d.missing_indicator, "rule": rule,
        })
    return rows


def _set(run_id: str, **fields) -> None:
    with _lock:
        _jobs[run_id].update(fields)


# ---------- the background job ----------

def _run_job(run_id: str, dataset_id: str, target: str, fast_mode: bool, cv_folds: int) -> None:
    try:
        _set(run_id, status="running", stage="loading data")
        raw = pd.read_csv(_dataset_csv(dataset_id))
        df, miss_report = normalize_missing(raw)          # rule R0
        if fast_mode and len(df) > FAST_MODE_ROWS:
            df = df.sample(n=FAST_MODE_ROWS, random_state=42)

        _set(run_id, stage="preparing the target column")
        df, target_notes = prepare_target(df, target)

        _set(run_id, stage="profiling the data")
        profile = DataProfiler(df, target_column=target).profile()

        _set(run_id, stage="deciding preprocessing")
        plan = PreprocessingDecider(profile).decide()

        _set(run_id, stage="comparing the baseline and 8 models")
        result = ModelSelector(df, profile, plan).select(cv_folds=cv_folds)

        _set(run_id, stage="saving")
        stored_id = _storage().save_run(f"{_dataset_name(dataset_id)} -> {target} (API)", profile, plan, result, "")

        payload = _clean({
            "run_id": run_id, "dataset_id": dataset_id, "target_column": target,
            "target_notes": target_notes,
            "missing_report": _report_to_dict(miss_report),
            "missing_handling": _missing_handling_rows(profile, plan),
            "preprocessing": plan.to_dict(),
            "selection": result.to_dict(),
            "database_run_id": stored_id,
        })
        (_data_dir() / "runs" / f"{run_id}.json").write_text(json.dumps(payload))
        _set(run_id, status="done", stage="done", finished=time.time())
    except (InvalidTargetError, InsufficientDataError) as e:
        _set(run_id, status="failed", stage="failed", error=str(e), finished=time.time())
    except Exception as e:  # anything unexpected: report it, never crash the worker
        _set(run_id, status="failed", stage="failed", error=f"{type(e).__name__}: {e}", finished=time.time())


# ---------- request models ----------

class RunRequest(BaseModel):
    dataset_id: str = Field(..., description="id returned by POST /datasets")
    target_column: str = Field(..., description="the column to predict")
    fast_mode: bool = Field(True, description="sample 10,000 rows if the dataset is larger")
    cv_folds: int = Field(5, ge=2, le=10, description="cross-validation folds")


# ---------- endpoints ----------

@app.get("/", include_in_schema=False)
def console():
    """The DataNarrate web console (a single page that uses the API below)."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.post("/datasets", tags=["datasets"], summary="Upload a CSV and get its missing-data report")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not read this file as a CSV ({type(e).__name__})")
    if raw.shape[0] == 0 or raw.shape[1] < 2:
        raise HTTPException(400, "The CSV needs at least 2 columns and 1 row")

    dataset_id = uuid.uuid4().hex[:12]
    (_data_dir() / "datasets" / f"{dataset_id}.csv").write_bytes(content)
    (_data_dir() / "datasets" / f"{dataset_id}.json").write_text(json.dumps({"filename": file.filename}))
    _, report = normalize_missing(raw)
    return _clean({
        "dataset_id": dataset_id, "filename": file.filename,
        "n_rows": raw.shape[0], "n_columns": raw.shape[1], "columns": raw.columns.tolist(),
        "missing_report": _report_to_dict(report),
    })


@app.get("/datasets/{dataset_id}/missing-report", tags=["datasets"])
def missing_report(dataset_id: str):
    raw = pd.read_csv(_dataset_csv(dataset_id))
    _, report = normalize_missing(raw)
    return _clean(_report_to_dict(report))


@app.post("/runs", tags=["runs"], status_code=202, summary="Start a run (returns immediately)")
def start_run(req: RunRequest):
    path = _dataset_csv(req.dataset_id)
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if req.target_column not in columns:
        raise HTTPException(422, f"'{req.target_column}' is not a column. Available: {columns}")

    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[run_id] = {"run_id": run_id, "status": "queued", "stage": "queued",
                         "dataset_id": req.dataset_id, "target_column": req.target_column,
                         "created": time.time(), "error": None}
    _executor.submit(_run_job, run_id, req.dataset_id, req.target_column, req.fast_mode, req.cv_folds)
    return {"run_id": run_id, "status": "queued", "status_url": f"/runs/{run_id}"}


def _job_or_404(run_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(run_id)
    if job is not None:
        return dict(job)
    if run_id.isalnum() and (_data_dir() / "runs" / f"{run_id}.json").exists():
        return {"run_id": run_id, "status": "done", "stage": "done", "error": None}   # finished before a restart
    raise HTTPException(404, "Unknown run_id")


@app.get("/runs/{run_id}", tags=["runs"], summary="Poll the status of a run")
def run_status(run_id: str):
    job = _job_or_404(run_id)
    return _clean({k: job.get(k) for k in ("run_id", "status", "stage", "dataset_id", "target_column", "error")})


@app.get("/runs/{run_id}/result", tags=["runs"], summary="Full results of a finished run")
def run_result(run_id: str):
    job = _job_or_404(run_id)
    if job["status"] == "failed":
        raise HTTPException(422, f"The run failed: {job['error']}")
    if job["status"] != "done":
        raise HTTPException(409, f"The run is not finished yet (status: {job['status']}, stage: {job['stage']})")
    return json.loads((_data_dir() / "runs" / f"{run_id}.json").read_text())


@app.get("/history", tags=["runs"], summary="Past runs saved in the database")
def history():
    runs = []
    for r in _storage().list_runs():
        d = asdict(r)
        d.pop("narration", None)
        d.pop("final_justification", None)
        runs.append(d)
    return _clean(runs)