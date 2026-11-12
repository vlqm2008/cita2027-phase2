"""Hashing and atomic writes for research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n").encode("utf-8")


def write_new_bytes(path: Path, data: bytes) -> None:
    """Create a new immutable-by-policy artifact; refuse replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a derived report, never a raw snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temp:
        temp.write(canonical_json_bytes(value))
        name = temp.name
    os.replace(name, path)
