"""Tests for items 5-7: prediction + model download, narration fact-check, PDF report (web/extras.py)."""
import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_wine

import web.server as server


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    server.DB_PATH = str(tmp_path_factory.mktemp("db") / "runs.db")
    return TestClient(server.app)


@pytest.fixture(scope="module")
def job(client):
    buf = io.BytesIO(load_wine(as_frame=True).frame.to_csv(index=False).encode())
    up = client.post("/api/datasets", files={"file": ("wine.csv", buf, "text/csv")}).json()
    jid = client.post("/api/runs", json={"dataset_id": up["dataset_id"], "target": "target"}).json()["job_id"]
    for _ in range(300):
        s = client.get(f"/api/jobs/{jid}").json()
        if s["status"] != "running":
            assert s["status"] == "done", s.get("error")
            return jid
        time.sleep(0.5)
    raise AssertionError("run did not finish")


def train(client, jid):
    assert client.post(f"/api/jobs/{jid}/tasks/model").status_code == 200
    for _ in range(300):
        t = client.get(f"/api/jobs/{jid}/tasks/model").json()
        if t["status"] != "running":
            return t
        time.sleep(0.5)
    raise AssertionError("training did not finish")


def test_predict_needs_a_trained_model_first(client, job):
    assert client.get(f"/api/jobs/{job}/model.joblib").status_code == 409


def test_train_download_template_and_predict(client, job):
    t = train(client, job)
    assert t["status"] == "done", t.get("error")
    assert t["result"]["classes"] and len(t["result"]["feature_columns"]) == 13

    assert client.get(f"/api/jobs/{job}/model.joblib").content[:2] != b""
    tpl = client.get(f"/api/jobs/{job}/sample.csv")
    assert tpl.status_code == 200
    template = pd.read_csv(io.StringIO(tpl.text))
    assert len(template.columns) == 13

    # fill one value with a hidden placeholder: must be treated as missing, not crash
    template["alcohol"] = template["alcohol"].astype(object)
    template.loc[0, "alcohol"] = "?"
    r = client.post(f"/api/jobs/{job}/predict", files={"file": ("new.csv", template.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["notes"]["n_rows"] == 5 and d["notes"]["hidden_placeholders"] == 1
    assert "predicted_target" in d["columns"] and "confidence" in d["columns"]
    csv = client.get(f"/api/jobs/{job}/predictions.csv")
    assert csv.status_code == 200 and len(pd.read_csv(io.StringIO(csv.text))) == 5


def test_predict_rejects_bad_files(client, job):
    train(client, job)
    r = client.post(f"/api/jobs/{job}/predict", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422 and "missing required column" in r.json()["detail"]
    assert client.post(f"/api/jobs/{job}/predict", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
    assert client.get("/api/jobs/nope/model.joblib").status_code == 404


def test_pdf_report(client, job):
    r = client.get(f"/api/jobs/{job}/report.pdf")
    assert r.status_code == 200 and r.content[:4] == b"%PDF" and len(r.content) > 5000


def test_fact_check_needs_narration(client, job):
    r = client.post(f"/api/jobs/{job}/fact-check")
    assert r.status_code in (200, 400)       # 400 when no GROQ key was available for this run


def test_fact_check_catches_a_wrong_winner_and_number(client, job):
    payload = server._jobs[job]["result"]
    wrong = "Gradient boosting was the best model." if payload["result"]["best_model_name"] != "gradient_boosting" \
        else "Naive Bayes was the best model."
    server._jobs[job]["result"] = dict(payload, narration=wrong + " It scored 0.1234 on accuracy.")
    try:
        r = client.post(f"/api/jobs/{job}/fact-check").json()
    finally:
        server._jobs[job]["result"] = payload
    kinds = {f["type"] for f in r["findings"]}
    assert "winner" in kinds and "number" in kinds
