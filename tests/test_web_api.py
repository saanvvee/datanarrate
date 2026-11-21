"""End-to-end tests for the FastAPI web app (web/server.py)."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_wine

import web.server as server


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    server.DB_PATH = str(tmp_path_factory.mktemp("db") / "runs.db")   # never touch the real run history
    return TestClient(server.app)


def wine_csv(placeholder=False):
    df = load_wine(as_frame=True).frame
    if placeholder:
        df["alcohol"] = df["alcohol"].astype(object)
        df.loc[:4, "alcohol"] = "?"
    return io.BytesIO(df.to_csv(index=False).encode())


@pytest.fixture(scope="module")
def finished_job(client):
    up = client.post("/api/datasets", files={"file": ("wine.csv", wine_csv(), "text/csv")}).json()
    job = client.post("/api/runs", json={"dataset_id": up["dataset_id"], "target": "target"}).json()["job_id"]
    for _ in range(300):
        status = client.get(f"/api/jobs/{job}").json()
        if status["status"] != "running":
            return job, status
        time.sleep(0.5)
    raise AssertionError("run did not finish")


def test_frontend_and_plotly_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/vendor/plotly.min.js").status_code == 200


def test_upload_reports_shape_and_hidden_placeholders(client):
    r = client.post("/api/datasets", files={"file": ("wine.csv", wine_csv(placeholder=True), "text/csv")})
    assert r.status_code == 200
    d = r.json()
    assert d["n_rows"] == 178 and d["n_cols"] == 14
    assert d["missing"]["n_hidden"] == 5            # the five '?' became real missing values (R0)
    assert d["suggested_target"] == "target"
    assert len(d["preview"]) == 20


def test_non_csv_upload_is_rejected(client):
    r = client.post("/api/datasets", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 400


def test_unknown_target_is_rejected(client):
    up = client.post("/api/datasets", files={"file": ("wine.csv", wine_csv(), "text/csv")}).json()
    r = client.post("/api/runs", json={"dataset_id": up["dataset_id"], "target": "nope"})
    assert r.status_code == 400


def test_run_finishes_with_full_result(finished_job):
    _, status = finished_job
    assert status["status"] == "done", status.get("error")
    assert [entry["stage"] for entry in status["log"]] == ["target", "profile", "decide", "proof", "models", "narrate", "save"]
    res = status["result"]
    assert len(res["result"]["candidate_results"]) == 8
    assert len(res["result"]["ladder"]) == 4
    assert res["proof"]["n_passed"] == res["proof"]["n_checks"]
    assert res["significance"]["a"] == res["result"]["best_model_name"]
    assert res["run_id"] is not None


def test_ablation_runs_for_a_finished_job(client, finished_job):
    job, _ = finished_job
    a = client.post(f"/api/jobs/{job}/ablation").json()
    assert {v["variant_name"] for v in a["variants"]} == {
        "full_pipeline", "no_scaling", "no_smart_encoding_forced_onehot",
        "no_imputation_zero_fill", "no_outlier_clipping_R8"}


def test_history_lists_the_saved_run(client, finished_job):
    runs = client.get("/api/history").json()
    assert any(r["dataset_name"].startswith("wine.csv") for r in runs)
