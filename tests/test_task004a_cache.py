"""TASK-004A CPU-only unit and integration tests. No downloads or model inference."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cita2027.forecasting.cache import (
    collect_records, digest, make_plan, read_valid_record, select_pilot_origins,
    store_or_verify, validate_cross_model, validate_record, write_once_atomic,
)
from cita2027.forecasting.protocol import make_window
from cita2027.utils.provenance import canonical_json_bytes


def sample_df(n=650):
    dates = pd.bdate_range("2018-01-02", periods=n)
    returns = np.arange(n, dtype=float) / 100000
    returns[0] = np.nan
    return pd.DataFrame({"date": dates, "row_index": range(n),
                         "split": ["development"] * n, "log_return": returns,
                         "previous_date": dates.to_series().shift(1).to_numpy(),
                         "previous_row_index": [np.nan] + list(range(n - 1))})


def sample_plan(df, count=24):
    return make_plan(df, count, "snapshot", "a" * 64, "b" * 64,
                     "c" * 64, 2027, 2, 1e-5)


def record(df, plan, model="chronos", pos=0):
    origin = plan["origins"][pos]["origin_index"]
    w = make_window(df, origin)
    return {
        "schema_version": plan["schema_version"], "plan_sha256": digest(plan),
        "model": model, "checkpoint_id": "checkpoint-" + model,
        "checkpoint_commit": "d" * 40, "checkpoint_weight_sha256": "e" * 64,
        "processed_snapshot_id": plan["source_snapshot_id"],
        "processed_sha256": plan["processed_sha256"],
        "origin_index": origin, "target_index": origin + 1,
        "origin_date": w.origin_date, "target_date": w.target_date,
        "context_first_index": origin - 255, "context_last_index": origin,
        "context_sha256": w.context_sha256,
        "target_log_return": w.target_log_return,
        "q10": -0.1, "q50": 0., "q90": 0.1,
        "seed": plan["seed"] + origin, "context_length": 256, "horizon": 1,
        "samples": 100, "quantile_levels": [0.1, 0.5, 0.9],
        "target_units": "daily_log_return_decimal", "elapsed_seconds": 0.2,
        "peak_allocated_mib": 80., "replay_checked": pos < plan["replay_origins"],
        "replay_max_abs_diff": 0.0, "contamination_free_verified": False,
        "point_in_time_adjusted_prices_verified": False,
    }


def validate(rec, df, plan):
    validate_record(rec, plan, digest(plan), rec["model"], rec["checkpoint_id"],
                    rec["checkpoint_commit"], rec["checkpoint_weight_sha256"], df)


def test_pilot_origin_selection_20_to_50_and_frozen_plan():
    df = sample_df()
    p = sample_plan(df)
    assert len(p["origins"]) == 24
    assert len(set(select_pilot_origins(df, 24))) == 24
    assert p["origins"][0]["origin_index"] >= 256
    for o in p["origins"]:
        assert o["target_index"] == o["origin_index"] + 1
    with pytest.raises(ValueError, match="20-50"):
        select_pilot_origins(df, 12)
    df.loc[630:, "split"] = "final_test"
    assert all(df.loc[i, "split"] == df.loc[i + 1, "split"] == "development"
               for i in select_pilot_origins(df, 24))


def test_future_target_perturbation_does_not_affect_forecast_context():
    df = sample_df()
    p = sample_plan(df)
    i = p["origins"][0]["origin_index"]
    before = make_window(df, i)
    changed = df.copy(deep=True)
    changed.loc[i + 1:, "log_return"] = 999.0
    after = make_window(changed, i)
    np.testing.assert_array_equal(before.context, after.context)
    assert before.context_sha256 == after.context_sha256
    assert before.target_log_return != after.target_log_return


def test_record_schema_target_context_replay_and_quantile_validation():
    df = sample_df()
    p = sample_plan(df)
    good = record(df, p)
    validate(good, df, p)
    for key, bad in [
        ("target_index", good["target_index"] + 1),
        ("context_first_index", 0),
        ("seed", 2027),
        ("target_log_return", 10000.0),
        ("q50", 1.0),
        ("point_in_time_adjusted_prices_verified", True),
        ("replay_checked", False),
        ("replay_max_abs_diff", 1e-3),
        ("checkpoint_weight_sha256", "wrong"),
    ]:
        bad_record = dict(good)
        bad_record[key] = bad
        with pytest.raises(ValueError):
            validate_record(bad_record, p, digest(p), good["model"],
                            good["checkpoint_id"], good["checkpoint_commit"],
                            good["checkpoint_weight_sha256"], df)


def test_atomic_publish_rejects_overwrite_and_plan_mutation(tmp_path):
    path = tmp_path / "plans" / "plan.json"
    p = sample_plan(sample_df())
    assert store_or_verify(path, p) == digest(p)
    assert store_or_verify(path, p) == digest(p)
    with pytest.raises(FileExistsError):
        write_once_atomic(path, b"evil")
    modified = copy.deepcopy(p)
    modified["seed"] += 1
    with pytest.raises(ValueError, match="Immutable plan mismatch"):
        store_or_verify(path, modified)


def test_resume_detects_truncated_modified_unexpected_and_duplicate_files(tmp_path):
    df = sample_df()
    p = sample_plan(df)
    sha = digest(p)
    a = record(df, p)
    root = tmp_path / "records"
    file = root / f'{a["origin_index"]:06d}.json'
    write_once_atomic(file, canonical_json_bytes(a))
    kwargs = (root, p, sha, "chronos", a["checkpoint_id"], a["checkpoint_commit"],
              a["checkpoint_weight_sha256"], df)
    assert len(collect_records(*kwargs)) == 1
    assert read_valid_record(file, p, sha, "chronos", a["checkpoint_id"],
                             a["checkpoint_commit"], a["checkpoint_weight_sha256"], df) == a
    with pytest.raises(FileExistsError):
        write_once_atomic(file, canonical_json_bytes(a))
    extra = root / "999999.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="Unexpected/duplicate"):
        collect_records(*kwargs)
    extra.unlink()
    file.write_text("{" + "not json")
    with pytest.raises(json.JSONDecodeError):
        collect_records(*kwargs)


def test_cross_model_alignment_and_mismatches():
    df = sample_df()
    p = sample_plan(df)
    c = [record(df, p, "chronos", i) for i in range(3)]
    m = [record(df, p, "moirai", i) for i in range(3)]
    validate_cross_model(c, m)
    m[1]["context_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="mismatch"):
        validate_cross_model(c, m)
    m[1]["context_sha256"] = c[1]["context_sha256"]
    with pytest.raises(ValueError, match="Unequal"):
        validate_cross_model(c, m + [m[0]])


def test_resume_rejects_tampered_future_target_or_seed(tmp_path):
    df = sample_df()
    p = sample_plan(df)
    a = record(df, p)
    a["target_log_return"] += 0.001
    file = tmp_path / f'{a["origin_index"]:06d}.json'
    write_once_atomic(file, canonical_json_bytes(a))
    with pytest.raises(ValueError, match="target mismatch"):
        read_valid_record(file, p, digest(p), "chronos", a["checkpoint_id"],
                          a["checkpoint_commit"], a["checkpoint_weight_sha256"], df)


def test_pilot_runner_interruption_resume_and_two_model_audit(tmp_path, monkeypatch):
    """End-to-end CPU simulation of the *runner*: no real checkpoint/GPU claims."""
    import importlib.util
    import sys
    from types import SimpleNamespace

    script = Path(__file__).resolve().parents[1] / "scripts" / "pilot_forecast_cache.py"
    spec = importlib.util.spec_from_file_location("task004a_pilot_script", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    df = sample_df()
    state = {"created": 0, "fail_after": 4, "predicted": 0}

    class StubAdapter:
        def __init__(self, revision, device, samples):
            state["created"] += 1

        def predict(self, context):
            state["predicted"] += 1
            if state["fail_after"] and state["predicted"] > state["fail_after"]:
                raise RuntimeError("simulated interruption")
            return np.asarray([-0.01, 0., 0.01])

    class FakeCuda:
        def is_available(self):
            return True

        def get_device_capability(self, idx):
            return (12, 0)

        def reset_peak_memory_stats(self):
            pass

        def synchronize(self):
            pass

        def max_memory_allocated(self):
            return 30 * 2**20

        def empty_cache(self):
            pass

        def get_device_name(self, idx):
            return "TEST_FAKE_GPU_NOT_ACTUAL"

    class FakeTorch:
        cuda = FakeCuda()
        version = SimpleNamespace(cuda="TEST")

    # Only the dry-run test injects a fake torch object; actual GPU claims are prohibited.
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    sid = "snapshot"
    root = tmp_path
    (root / "configs").mkdir()
    (root / "artifacts" / "checkpoints").mkdir(parents=True)
    (root / "artifacts" / "manifests").mkdir(parents=True)
    cfg = root / "configs" / "smoke.yaml"
    cfg.write_text("placeholder")
    audit = root / "artifacts" / "checkpoints" / "hf_audit.json"
    audit.write_text("placeholder")
    data_sha = "a" * 64
    conf = {"source_snapshot_id": sid, "context_length": 256, "horizon": 1,
            "quantiles": [0.1, 0.5, 0.9], "samples": 100, "seed": 2027,
            "replay_origins": 2, "replay_tolerance": 1e-5,
            "model_ids": {"chronos": "amazon/chronos-t5-small",
                          "moirai": "Salesforce/moirai-1.0-R-small"}}
    checkpoint_list = [
        {"model_id": id_, "commit_sha": letter * 40,
         "files": {"model.safetensors": {"lfs_sha256": letter * 64}}}
        for id_, letter in ((conf["model_ids"]["chronos"], "c"),
                            (conf["model_ids"]["moirai"], "d"))]
    monkeypatch.setattr(runner, "sources", lambda *a: (conf, df, {"processed_sha256": data_sha}, checkpoint_list))
    monkeypatch.setattr(runner, "verify_local_checkpoint", lambda model, rev, files: next(iter(files.values())))
    monkeypatch.setattr(runner, "ChronosT5Adapter", type("FakeChronos", (StubAdapter,),
                                                    {"checkpoint_id": conf["model_ids"]["chronos"]}))
    monkeypatch.setattr(runner, "Moirai10Adapter", type("FakeMoirai", (StubAdapter,),
                                                   {"checkpoint_id": conf["model_ids"]["moirai"]}))
    monkeypatch.setattr(runner, "seed_inference", lambda seed: None)
    args = SimpleNamespace(root=root, config=cfg, checkpoint_audit=audit,
                           count=24, model="chronos", device="cuda")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run(args)
    p, psha, _ = runner.load_plan(root, cfg, audit, conf, df,
                                  {"processed_sha256": data_sha}, 24)
    run_dir = root / "artifacts" / "forecast_cache" / "pilot" / "runs" / psha / "chronos"
    assert len(list((run_dir / "records").glob("*.json"))) == 2
    assert not (run_dir / ".pilot.lock").exists()
    state["fail_after"] = 0
    state["predicted"] = 0
    runner.run(args)
    assert state["predicted"] == 24 - 2  # 22 new; first 2 were replayed before crash
    assert len(list((run_dir / "records").glob("*.json"))) == 24
    state["predicted"] = 0
    runner.run(args)
    assert state["predicted"] == 0  # no repeated inference after completion
    # Simulate crash between publishing CSV and manifest, with complete records.
    (run_dir / "manifest.json").unlink()
    state["predicted"] = 0
    runner.run(args)
    assert state["predicted"] == 0  # safe reconstruction of manifest, CSV retained
    args.model = "moirai"
    runner.run(args)
    args.model = None
    runner.audit_pilot(args)
    report = root / "artifacts" / "forecast_cache" / "pilot" / "runs" / psha / "audit_report.json"
    assert json.loads(report.read_text())["cross_model_alignment"] == "PASS"
