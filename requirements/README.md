# Dependency management

Base runtime dependencies are constrained in `pyproject.toml`. This is a reproducible
installation specification, **not a fully resolved cross-platform lock**.

On the actual local workstation, after environment inspection and successful install:

```bash
python -m pip freeze > requirements/local-verified-<OS>-py<version>.txt
python -m pip check
```

Commit the generated lock and environment report only after hardware verification.
Avoid pinning a CUDA-specific torch wheel inside the CPU-only data dependencies.
Phase 2 does not require torch. TSFM backends will be pinned independently in Phase 3.
