"""
Unit Tests for Cross-Horizon Self-Attention Online Adaptation (Attn-H3).
"""

import numpy as np
import pytest

from cita2027.adaptation.attention_cross_horizon import (
    CrossHorizonAttentionModule,
    HypothesisAttnH3_Corrector,
)
from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


def test_attention_module_dimensions_and_masking():
    module = CrossHorizonAttentionModule(embed_dim=8, num_heads=2, seed=42)
    horizons = [1, 2, 4, 8, 16]

    items = [
        ForecastFeedbackItem(
            origin_t=0,
            horizon=h,
            target_t=10,
            y_true=5.0,
            base_samples=np.ones(10) * 4.0,
        )
        for h in [1, 2]
    ]

    out_vecs, attn_weights = module.forward(
        target_horizons=horizons,
        feedback_items=items,
        current_t=10,
        age_decay_gamma=0.1,
    )

    assert out_vecs.shape == (5, 8), "Output vectors must match (num_targets, embed_dim)"
    assert attn_weights.shape == (5, 2), "Attention weights must match (num_targets, num_feedbacks)"
    # Sum of attention weights across feedbacks must equal 1.0
    row_sums = np.sum(attn_weights, axis=1)
    assert np.allclose(row_sums, 1.0), "Attention weights must sum to 1 per target"


def test_attn_h3_corrector_lifecycle_and_boundedness():
    horizons = [1, 2, 4, 8]
    corrector = HypothesisAttnH3_Corrector(horizons=horizons, embed_dim=8, seed=123)

    delta_mu, scale_mult = corrector.get_corrections(4)
    assert np.isfinite(delta_mu)
    assert scale_mult > 0.0

    # Feed 15 steps of feedback
    for t in range(15):
        items = [
            ForecastFeedbackItem(
                origin_t=t,
                horizon=1,
                target_t=t + 1,
                y_true=10.0 + np.random.normal(0, 0.2),
                base_samples=np.ones(20) * 5.0,
            )
        ]
        info = corrector.update_with_feedback(items, current_t=t + 1)
        assert info["head_norm"] <= 8.0, "State must remain bounded in compact ball"

    assert corrector.latest_attention_matrix is not None
