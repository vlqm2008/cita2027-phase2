"""TASK-004B.2 CPU-only fault injection; never load model checkpoints."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_development_cache as fc
from test_task004b1_plan import fixture_df, plan as toy_full_plan
from cita2027.forecasting.cache import digest
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import canonical_json_bytes, sha256_file


class FakeAdapter:
    calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, context):
        FakeAdapter.calls += 1
        # Avoid future observations: model accepts only the context.
        return np.array([-0.01, 0.0, 0.01])


def mocked_env(monkeypatch, tmp_path):
    df = fixture_df()
    p = toy_full_plan(df)
    sha = digest(p)
    view = {**p, "origins": [
        {k: o[k] for k in ("origin_index", "target_index", "origin_date", "target_date", "context_sha256")}
        for o in p["origins"]]}
    monkeypatch.setattr(fc, "prepare", lambda *args: ({}, df, [], p, view, sha))
    monkeypatch.setattr(fc, "identity", lambda *args: (FakeAdapter, "toy-model", "a" * 40, "b" * 64, {}))
    monkeypatch.setattr(fc, "preflight_worker", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(fc, "seed_inference", lambda *args: None)
    import torch
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1048576)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    FakeAdapter.calls = 0
    return p, view, sha, df


def execute(tmp_path, count=5, interrupt=None, model="chronos", batch_size=2):
    return fc.worker(tmp_path, tmp_path / "cfg", tmp_path / "audit", model,
                     count=count, batch_size=batch_size, force_limit=interrupt)


def test_record_schema_seed_and_alignment(monkeypatch, tmp_path):
    p, view, sha, df = mocked_env(monkeypatch, tmp_path)
    o = p["origins"][0]
    w = make_window(df, o["origin_index"])
    r = fc.make_record(view, sha, "chronos", "toy-model", "a" * 40, "b" * 64,
                       w, [-.01, 0, .01], o["seed"], .04, 5.0, True, 0.0)
    fc.validate_record(r, view, sha, "chronos", "toy-model", "a" * 40, "b" * 64, df)
    assert r["origin_index"] + 1 == r["target_index"]
    assert r["context_first_index"] == 25 and r["seed"] == 2027 + 280
    assert r["target_units"] == "daily_log_return_decimal"
    for field, value in (("seed", -1), ("q10", .99), ("target_index", 999),
                         ("contamination_free_verified", True)):
        broken = {**r, field: value}
        with pytest.raises(ValueError):
            fc.validate_record(broken, view, sha, "chronos", "toy-model", "a" * 40, "b" * 64, df)


def test_partial_batch_no_manifest_and_idempotent_resume(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    first = execute(tmp_path, count=5)
    assert first["status"] == "PARTIAL_BATCH_VERIFIED_NOT_COMPLETE"
    run = fc.paths(tmp_path, sha, "chronos")
    assert len(list((run / "records").glob("*.json"))) == 5
    assert not (run / "manifest.json").exists()
    assert not (run / "predictions.csv").exists()
    saved = {p.name: sha256_file(p) for p in (run / "records").glob("*.json")}
    n = FakeAdapter.calls
    again = execute(tmp_path, count=5)
    assert again["new_records"] == 0
    assert FakeAdapter.calls == n
    assert {p.name: sha256_file(p) for p in (run / "records").glob("*.json")} == saved


def test_interrupted_batch_resume_only_missing(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="INTENTIONAL_INTERRUPTION"):
        execute(tmp_path, count=5, interrupt=3)
    run = fc.paths(tmp_path, sha, "chronos")
    assert len(list((run / "records").glob("*.json"))) == 3
    assert not (run / "manifest.json").exists()
    assert not (run / ".full.lock").exists()
    calls = FakeAdapter.calls
    report = execute(tmp_path, count=5)
    assert report["new_records"] == 2
    assert FakeAdapter.calls == calls + 2


def test_corrupt_record_rejected_no_overwrite(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    execute(tmp_path, count=5)
    path = fc.paths(tmp_path, sha, "chronos") / "records" / "000280.json"
    path.write_text('{"bad":')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        execute(tmp_path, count=5)
    assert path.read_text() == '{"bad":'


def test_extra_record_and_stray_stage_handling(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    execute(tmp_path, count=5)
    rec = fc.paths(tmp_path, sha, "chronos") / "records"
    (rec / ".stage-interrupted").write_text("incomplete")
    execute(tmp_path, count=5)
    assert not (rec / ".stage-interrupted").exists()
    (rec / "009999.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected"):
        execute(tmp_path, count=5)


def test_duplicate_origin_file_detected(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    execute(tmp_path, count=5)
    rec = fc.paths(tmp_path, sha, "chronos") / "records"
    (rec / "000280-copy.json").write_bytes((rec / "000280.json").read_bytes())
    with pytest.raises(ValueError, match="Unexpected"):
        execute(tmp_path, count=5)


def test_cross_model_partial_audit(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    execute(tmp_path, count=5, model="chronos")
    execute(tmp_path, count=5, model="moirai")
    report = fc.audit_cache(tmp_path, tmp_path / "cfg", tmp_path / "audit", 5)
    assert report["status"] == "SMALL_BATCH_AUDIT_PASS_NOT_COMPLETE"
    assert report["cross_model_alignment"] == "PASS"
    assert report["contamination_free_verified"] is False
    assert report["point_in_time_adjusted_prices_verified"] is False


def test_wrong_model_record_and_date_rejected(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    execute(tmp_path, count=5)
    rec_dir = fc.paths(tmp_path, sha, "chronos") / "records"
    rec = json.loads((rec_dir / "000280.json").read_text())
    rec["target_date"] = "2035-01-01"
    (rec_dir / "000280.json").write_bytes(canonical_json_bytes(rec))
    with pytest.raises(ValueError, match="causal window"):
        execute(tmp_path, count=5)


def test_lock_blocks_concurrent_execution(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    run = fc.paths(tmp_path, sha, "chronos")
    run.mkdir(parents=True)
    lock = run / ".full.lock"
    lock.write_text("pid=999")
    with pytest.raises(RuntimeError, match="locked"):
        execute(tmp_path, count=5)
    assert lock.is_file()


def test_full_completion_manifest_corruption_and_reaudit(monkeypatch, tmp_path):
    p, view, sha, df = mocked_env(monkeypatch, tmp_path)
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda n: "FAKE CUDA TEST GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda n: (12, 0))
    results = {}
    for model in ("chronos", "moirai"):
        report = execute(tmp_path, count=p["count"], model=model, batch_size=17)
        assert report["status"] == fc.COMPLETE_STATUS
        results[model] = report
    audit = fc.audit_cache(tmp_path, tmp_path / "cfg", tmp_path / "audit", p["count"])
    assert audit["status"] == fc.AUDITED_STATUS
    assert audit["count_per_model"] == p["count"]
    run = fc.paths(tmp_path, sha, "chronos")
    assert execute(tmp_path, count=p["count"])["status"] == "ALREADY_COMPLETE_VERIFIED"
    file = run / "predictions.csv"
    file.write_bytes(file.read_bytes() + b"\nCORRUPT")
    with pytest.raises(ValueError, match="CSV"):
        fc.audit_cache(tmp_path, tmp_path / "cfg", tmp_path / "audit", p["count"])


def test_small_batch_gate_created_and_verifiable(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    for model in ("chronos", "moirai"):
        execute(tmp_path, count=5, model=model)
    evidence = fc.gate_evidence(tmp_path, tmp_path / "cfg", tmp_path / "audit", 5)
    assert evidence["status"] == "SMALL_BATCH_GATE_ACCEPTED"
    path = fc.gate_path(tmp_path, sha)
    fc.store_or_verify(path, evidence)
    assert fc.verify_gate(tmp_path, tmp_path / "cfg", tmp_path / "audit") == evidence
    changed = fc.paths(tmp_path, sha, "moirai") / "records" / "000280.json"
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError):
        fc.verify_gate(tmp_path, tmp_path / "cfg", tmp_path / "audit")


def test_full_execution_requires_existing_accepted_batch_gate(monkeypatch, tmp_path):
    _, view, sha, df = mocked_env(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="gate missing"):
        fc.verify_gate(tmp_path, tmp_path / "cfg", tmp_path / "audit")
