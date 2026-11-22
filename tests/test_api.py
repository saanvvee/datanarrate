import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATANARRATE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATANARRATE_DB", str(tmp_path / "test.db"))
    from api.main import app
    return TestClient(app)


def csv_bytes(n=240, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n)
    df = pd.DataFrame({
        "x": x,
        "y": rng.randn(n),
        "city": rng.choice(["a", "b", "c"], n),
        "label": (x + rng.randn(n) * 0.5 > 0).astype(int),
    })
    df.loc[rng.rand(n) < 0.1, "city"] = "?"          # hidden placeholders
    df.loc[rng.rand(n) < 0.05, "y"] = np.nan         # real empty cells
    return df.to_csv(index=False).encode()


def upload(client, content=None):
    return client.post("/datasets", files={"file": ("data.csv", content or csv_bytes(), "text/csv")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_console_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "DataNarrate" in r.text and "text/html" in r.headers["content-type"]


def test_upload_returns_accurate_missing_report(client):
    r = upload(client)
    assert r.status_code == 200
    body = r.json()
    rep = body["missing_report"]
    assert rep["n_hidden_placeholders"] > 0          # the '?' values were found
    assert rep["n_empty_cells"] > 0
    assert rep["pct_missing"] > 0
    assert body["columns"] == ["x", "y", "city", "label"]
    again = client.get(f"/datasets/{body['dataset_id']}/missing-report").json()
    assert again["n_missing"] == rep["n_missing"]


def test_upload_rejects_non_csv(client):
    r = client.post("/datasets", files={"file": ("x.csv", b"\x00\x01\x02", "text/csv")})
    assert r.status_code == 400


def test_unknown_dataset_and_run_return_404(client):
    assert client.get("/datasets/doesnotexist/missing-report").status_code == 404
    assert client.get("/runs/doesnotexist").status_code == 404


def test_run_with_bad_target_is_rejected(client):
    ds = upload(client).json()["dataset_id"]
    r = client.post("/runs", json={"dataset_id": ds, "target_column": "nope"})
    assert r.status_code == 422


def test_full_run_end_to_end(client):
    ds = upload(client).json()["dataset_id"]
    start = client.post("/runs", json={"dataset_id": ds, "target_column": "label", "cv_folds": 3})
    assert start.status_code == 202
    run_id = start.json()["run_id"]

    status = None
    deadline = time.time() + 240
    while time.time() < deadline:
        status = client.get(f"/runs/{run_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert status["status"] == "done", status

    res = client.get(f"/runs/{run_id}/result")
    assert res.status_code == 200
    body = res.json()
    json.dumps(body, allow_nan=False)                # valid JSON: no NaN anywhere
    assert body["selection"]["best_model_name"]
    assert len(body["selection"]["candidate_results"]) >= 6
    assert body["missing_report"]["n_hidden_placeholders"] > 0
    assert any(row["column"] == "city" for row in body["missing_handling"])

    hist = client.get("/history").json()
    assert any(h["id"] == body["database_run_id"] for h in hist)