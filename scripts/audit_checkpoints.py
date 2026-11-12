"""Resolve and report immutable Hugging Face checkpoint commits; no weights downloaded."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def audit(repo_ids: list[str]) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    result = []
    for repo_id in repo_ids:
        info = api.model_info(repo_id, revision="main", files_metadata=True)
        if not info.sha or len(info.sha) != 40:
            raise RuntimeError(f"Unable to resolve immutable checkpoint commit: {repo_id}")
        interesting = {}
        for f in info.siblings or []:
            if f.rfilename.endswith((".safetensors", ".bin", "config.json")):
                interesting[f.rfilename] = {
                    "size_bytes": getattr(f, "size", None),
                    "lfs_sha256": (getattr(f.lfs, "sha256", None)
                                   if getattr(f, "lfs", None) is not None else None),
                }
        result.append({
            "model_id": repo_id, "commit_sha": info.sha,
            "last_modified": info.last_modified.isoformat() if info.last_modified else None,
            "library_name": info.library_name, "pipeline_tag": info.pipeline_tag,
            "files": interesting,
            "training_data_exclusion_of_spy_confirmed": False,
            "training_cutoff_verified": False,
        })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="Optional new file; refuses overwrite")
    args = ap.parse_args()
    result = audit(["amazon/chronos-t5-small", "Salesforce/moirai-1.0-R-small"])
    payload = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
