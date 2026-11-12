"""Focused regression for NumPy index/seed types in Validation worker."""
import json
import random
import numpy as np
import pytest
from task006a_validation import normalize_worker_row
from cita2027.utils.provenance import canonical_json_bytes

def test_normalizes_numpy_ints_without_mutating_original_plan_row():
    original = {"origin_index": np.int64(4530), "target_index": np.int64(4531),
                "seed": np.int64(6557), "context_first_index": np.int64(4275),
                "origin_date": "2022-12-30"}
    before = canonical_json_bytes(original)
    normalized = normalize_worker_row(original)
    assert all(type(normalized[k]) is int for k in
               ("origin_index", "target_index", "seed", "context_first_index"))
    assert normalized["origin_date"] == original["origin_date"]
    assert canonical_json_bytes(original) == before  # exact SHA anchor unchanged
    assert original["seed"] == normalized["seed"]
    random.seed(normalized["seed"])
    assert isinstance(json.loads(canonical_json_bytes(normalized))["seed"], int)
    assert isinstance(json.loads(canonical_json_bytes(normalized))["origin_index"], int)

def test_python_ints_preserved_and_no_changes_to_floats():
    row = {"seed": 6557, "origin_index": 4530, "value": np.float64(0.5)}
    result = normalize_worker_row(row)
    assert result["seed"] == 6557 and type(result["seed"]) is int
    assert result["origin_index"] == 4530 and type(result["origin_index"]) is int
    assert result["value"] == 0.5
