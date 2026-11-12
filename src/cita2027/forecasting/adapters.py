"""Frozen sample-generating adapters. Optional TSFM dependencies are lazy-loaded."""

from __future__ import annotations

import gc
import numpy as np

from .protocol import check_quantiles


def samples_to_quantiles(samples: object) -> np.ndarray:
    """Compute empirical 10/50/90% return quantiles from one-step samples."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 10 or not np.isfinite(arr).all():
        raise ValueError("Expected at least ten finite one-step return samples")
    return check_quantiles(np.quantile(arr, [0.1, 0.5, 0.9], method="linear"))


def seed_inference(seed: int) -> None:
    """Seed stochastic prediction without enabling model training."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChronosT5Adapter:
    """Legacy Chronos-T5 sample-path API; not Chronos-Bolt/Chronos-2."""

    checkpoint_id = "amazon/chronos-t5-small"

    def __init__(self, revision: str, device: str = "cuda", samples: int = 100):
        import torch
        from chronos import ChronosPipeline

        self.device = device
        self.samples = samples
        self.pipeline = ChronosPipeline.from_pretrained(
            self.checkpoint_id, revision=revision, device_map=device, torch_dtype=torch.float32
        )
        self.pipeline.model.model.eval()
        if any(p.requires_grad for p in self.pipeline.model.model.parameters()):
            for p in self.pipeline.model.model.parameters():
                p.requires_grad_(False)

    def predict(self, context: np.ndarray) -> np.ndarray:
        import torch

        if context.shape != (256,) or not np.isfinite(context).all():
            raise ValueError("Invalid 256-step log-return context")

        context_tensor = torch.from_numpy(
            np.array(context, dtype=np.float32, copy=True)
        )
        with torch.inference_mode():
            samples = self.pipeline.predict(
                context_tensor,
                prediction_length=1,
                num_samples=self.samples,
                limit_prediction_length=True,
            )
        sample_array = samples.detach().cpu().numpy()
        if sample_array.shape != (1, self.samples, 1):
            raise ValueError(f"Unexpected Chronos output shape: {sample_array.shape}")
        return samples_to_quantiles(sample_array[0, :, 0])


class Moirai10Adapter:
    """Moirai-1.0-R GluonTS predictor, univariate returns with no covariates."""

    checkpoint_id = "Salesforce/moirai-1.0-R-small"

    def __init__(self, revision: str, device: str = "cuda", samples: int = 100):
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        module = MoiraiModule.from_pretrained(self.checkpoint_id, revision=revision)
        model = MoiraiForecast(
            module=module, prediction_length=1, context_length=256, patch_size="auto",
            num_samples=samples, target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        self.model = model
        self.samples = samples
        self.device = device
        self.predictor = model.create_predictor(batch_size=1, device=device)

    def predict(self, context: np.ndarray) -> np.ndarray:
        import pandas as pd
        import torch
        from gluonts.dataset.common import ListDataset

        if context.shape != (256,) or not np.isfinite(context).all():
            raise ValueError("Invalid 256-step log-return context")
        # Synthetic index is intentionally not market time: GluonTS needs a regular
        # index, while the model gets *only* the 256 observed scalar returns.
        # Calendar labels and target alignment remain in the external protocol.
        ds = ListDataset(
            [{"start": pd.Period("2000-01-03", freq="B"),
              "target": context.astype(np.float32)}],
            freq="B",
        )
        with torch.inference_mode():
            forecasts = list(self.predictor.predict(ds))
        if len(forecasts) != 1:
            raise ValueError("Expected exactly one Moirai forecast")
        samples = np.asarray(forecasts[0].samples, dtype=np.float64)
        if samples.shape != (self.samples, 1):
            raise ValueError(f"Unexpected Moirai output shape: {samples.shape}")
        return samples_to_quantiles(samples[:, 0])


def release_adapter(adapter: object) -> None:
    """Drop the model before loading another backbone."""
    del adapter
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
