"""
Forecast Cache and Provenance Tracking Contract.

Tracks full forecast provenance without look-ahead leakage:
- Dataset and series identifier
- Forecast origin timestamp
- Forecast target timestamp
- Forecast horizon
- Ground-truth availability timestamp
- Model/checkpoint identifier
- Forecast samples (M draws)
- Transform and preprocessing metadata
- State/configuration version
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ForecastProvenanceEntry:
    """Immutable record tracking provenance of a multi-horizon probabilistic forecast."""
    dataset_name: str
    series_id: str
    model_id: str
    origin_t: int
    target_t: int
    horizon: int
    availability_t: int
    num_samples: int
    samples: np.ndarray
    base_location: float
    base_dispersion: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Deterministic fingerprint of origin, horizon, and model configuration."""
        raw_key = f"{self.dataset_name}_{self.series_id}_{self.model_id}_t{self.origin_t}_h{self.horizon}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


class ForecastCache:
    """
    Causal, disk-backed or in-memory forecast cache for multi-horizon prequential evaluation.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.in_memory_store: Dict[str, ForecastProvenanceEntry] = {}

    def store_forecast(self, entry: ForecastProvenanceEntry) -> str:
        """Store forecast provenance record."""
        key = entry.compute_hash()
        self.in_memory_store[key] = entry

        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.npz"
            np.savez_compressed(
                file_path,
                samples=entry.samples,
                metadata=json.dumps({
                    "dataset_name": entry.dataset_name,
                    "series_id": entry.series_id,
                    "model_id": entry.model_id,
                    "origin_t": entry.origin_t,
                    "target_t": entry.target_t,
                    "horizon": entry.horizon,
                    "availability_t": entry.availability_t,
                    "base_location": entry.base_location,
                    "base_dispersion": entry.base_dispersion,
                    "extra": entry.metadata,
                }),
            )
        return key

    def get_forecast(
        self,
        dataset_name: str,
        series_id: str,
        model_id: str,
        origin_t: int,
        horizon: int,
    ) -> Optional[ForecastProvenanceEntry]:
        """Retrieve cached forecast if present."""
        raw_key = f"{dataset_name}_{series_id}_{model_id}_t{origin_t}_h{horizon}"
        key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        if key in self.in_memory_store:
            return self.in_memory_store[key]

        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.npz"
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                meta = json.loads(str(data["metadata"]))
                entry = ForecastProvenanceEntry(
                    dataset_name=meta["dataset_name"],
                    series_id=meta["series_id"],
                    model_id=meta["model_id"],
                    origin_t=meta["origin_t"],
                    target_t=meta["target_t"],
                    horizon=meta["horizon"],
                    availability_t=meta["availability_t"],
                    num_samples=len(data["samples"]),
                    samples=data["samples"],
                    base_location=meta["base_location"],
                    base_dispersion=meta["base_dispersion"],
                    metadata=meta.get("extra", {}),
                )
                self.in_memory_store[key] = entry
                return entry

        return None
