"""
Use & export routes for the DataNarrate web app (items 5 to 7).

    POST /api/jobs/{id}/tasks/model   train the final model (uses the existing task endpoints)
    GET  /api/jobs/{id}/model         info about the trained model
    GET  /api/jobs/{id}/model.joblib  download the trained model
    GET  /api/jobs/{id}/sample.csv    template CSV with the columns the model expects
    POST /api/jobs/{id}/predict       upload a CSV -> predictions (JSON preview)
    GET  /api/jobs/{id}/predictions.csv   all predictions from the last upload
    POST /api/jobs/{id}/fact-check    check the narration against the computed numbers
    GET  /api/jobs/{id}/report.pdf    one-click PDF report

Installed from web/server.py with install(app, lock, jobs, clean).
"""
import io
from typing import Any, Callable, Dict

import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response

from src.fact_checker import check_narration
from src.predictor import (PredictionInputError, bundle_info, bundle_to_bytes, predict_csv, sample_csv)

PREVIEW = 15


def facts_from_payload(payload: Dict[str, Any], missing_report: Dict[str, Any]) -> Dict[str, Any]:
    """Re-labels the web payload with the keys the fact-checker and the PDF builder expect."""
    actions = {d["column"]: d.get("missing_action") for d in payload["plan"]["column_decisions"]}
    handling = [{"column": r["column"], "missing_pct": r["missing_pct"], "action": actions.get(r["column"], "none"),
                 "was_missing_flag": bool(r.get("flag")), "rule": r.get("rule", "")}
                for r in payload.get("missing_handling", [])]
    return {
        "dataset_name": payload.get("dataset_name"), "target_column": payload.get("target"),
        "target_notes": payload.get("target_notes", []),
        "selection": payload["result"], "profile": payload["profile"], "preprocessing": payload["plan"],
        "missing_report": missing_report or {"summary": [], "columns": [], "warnings": []},
        "missing_handling": handling, "narration": payload.get("narration"),
        "fact_check": payload.get("fact_check"),
    }


def install(app, lock, jobs: Dict[str, Any], clean: Callable[[Any], Any]):
    def finished(job_id: str):
        with lock:
            job = jobs.get(job_id)
        if job is None or job.get("status") != "done":
            raise HTTPException(404, "This run is not finished (or the server was restarted). Run it again.")
        return job

    def bundle_of(job_id: str):
        job = finished(job_id)
        bundle = (job.get("context") or {}).get("bundle")
        if bundle is None:
            raise HTTPException(409, "Train the final model first (button: Train the final model).")
        return job, bundle

    @app.get("/api/jobs/{job_id}/model")
    def model_info(job_id: str):
        _, bundle = bundle_of(job_id)
        return clean(bundle_info(bundle))

    @app.get("/api/jobs/{job_id}/model.joblib")
    def model_download(job_id: str):
        _, bundle = bundle_of(job_id)
        return Response(bundle_to_bytes(bundle), media_type="application/octet-stream",
                        headers={"Content-Disposition": 'attachment; filename="datanarrate_model.joblib"'})

    @app.get("/api/jobs/{job_id}/sample.csv")
    def sample(job_id: str):
        _, bundle = bundle_of(job_id)
        return Response(sample_csv(bundle), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="prediction_template.csv"'})

    @app.post("/api/jobs/{job_id}/predict")
    async def predict(job_id: str, file: UploadFile = File(...)):
        job, bundle = bundle_of(job_id)
        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(400, "Please upload a .csv file.")
        try:
            raw = pd.read_csv(io.BytesIO(await file.read()))
        except Exception:
            raise HTTPException(400, "That file could not be read as a CSV.")
        try:
            out, notes = predict_csv(bundle, raw)
        except PredictionInputError as e:
            raise HTTPException(422, str(e))
        with lock:
            job["context"]["last_predictions"] = out
        return clean({"notes": notes, "columns": out.columns.tolist(),
                      "rows": out.head(PREVIEW).astype(object).where(out.head(PREVIEW).notna(), None).to_dict("records"),
                      "shown": min(PREVIEW, len(out)), "target": bundle["target"], "classes": bundle["classes"]})

    @app.get("/api/jobs/{job_id}/predictions.csv")
    def predictions_csv(job_id: str):
        job = finished(job_id)
        out = (job.get("context") or {}).get("last_predictions")
        if out is None:
            raise HTTPException(404, "Upload a CSV for prediction first.")
        return Response(out.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'})

    @app.post("/api/jobs/{job_id}/fact-check")
    def fact_check(job_id: str):
        job = finished(job_id)
        payload = job["result"]
        if not payload.get("narration"):
            raise HTTPException(400, "This run has no narration to check (no GROQ_API_KEY was set).")
        return clean(check_narration(payload["narration"], facts_from_payload(payload, job.get("missing_report"))))

    @app.get("/api/jobs/{job_id}/report.pdf")
    def report(job_id: str):
        job = finished(job_id)
        from src.report import build_pdf          # imported here: matplotlib/reportlab are only loaded when needed
        payload = job["result"]
        facts = facts_from_payload(payload, job.get("missing_report"))
        pdf = build_pdf(facts, dataset_name=payload.get("dataset_name", ""))
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="datanarrate_report.pdf"'})
