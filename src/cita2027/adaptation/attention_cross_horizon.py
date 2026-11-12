"""
Cross-Horizon Self-Attention Online Adaptation (Attn-H3).

Replaces static low-rank projections with a dynamic, parameter-efficient
causal self-attention mechanism across forecasting horizons:
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) - M_delay ) V
where:
- Q_h represents query tokens for target forecasting horizons h in H.
- K_{h'} represents key tokens for matured feedback items h' in A_t.
- V_{h'} carries observation residuals and dispersion innovations.
- M_delay is a causal delay attenuation mask penalizing stale feedback.

Provides fully interpretable cross-horizon attention weight matrices A(t)
revealing dynamic information routing across lead times during structural breaks.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class CrossHorizonAttentionModule:
    """
    Lightweight, strictly causal cross-horizon attention mechanism.
    """

    def __init__(
        self,
        embed_dim: int = 8,
        num_heads: int = 2,
        seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)

        # Linear projection matrices
        self.W_q = rng.normal(0.0, scale, size=(embed_dim, embed_dim))
        self.W_k = rng.normal(0.0, scale, size=(embed_dim, embed_dim))
        self.W_v = rng.normal(0.0, scale, size=(embed_dim, embed_dim))
        self.W_o = rng.normal(0.0, scale, size=(embed_dim, embed_dim))

    def compute_positional_token(self, horizon: int) -> np.ndarray:
        """Dyadic sinusoidal positional encoding token for horizon h."""
        token = np.zeros(self.embed_dim, dtype=np.float64)
        norm_h = math.log2(max(1, horizon)) / 4.0
        for i in range(self.embed_dim // 2):
            freq = 1.0 / (100.0 ** (2.0 * i / self.embed_dim))
            token[2 * i] = math.sin(norm_h * freq)
            token[2 * i + 1] = math.cos(norm_h * freq)
        return token

    def forward(
        self,
        target_horizons: List[int],
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
        age_decay_gamma: float = 0.08,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention-weighted cross-horizon transfer vectors.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - out_vectors: shape (|target_horizons|, embed_dim)
            - attn_weights: shape (|target_horizons|, |feedback_items|)
        """
        num_targets = len(target_horizons)
        num_feedbacks = len(feedback_items)

        if num_feedbacks == 0:
            return np.zeros((num_targets, self.embed_dim)), np.zeros((num_targets, 0))

        # Query matrix: shape (num_targets, embed_dim)
        q_tokens = np.array([self.compute_positional_token(h) for h in target_horizons])
        Q = np.dot(q_tokens, self.W_q)

        # Key matrix & Value matrix from feedback items
        k_tokens = []
        v_tokens = []
        delay_penalties = []

        for item in feedback_items:
            h_prime = item.horizon
            age = max(0, current_t - item.origin_t)
            delay_penalties.append(age_decay_gamma * float(age))

            pos_token = self.compute_positional_token(h_prime)
            k_tokens.append(pos_token)

            # Value vector carries normalized residual and dispersion mismatch
            base_loc = getattr(item, "base_location", None)
            if base_loc is None:
                base_loc = float(np.median(item.base_samples)) if len(item.base_samples) > 0 else 0.0
            base_disp = getattr(item, "base_dispersion", None)
            if base_disp is None:
                base_disp = float(np.std(item.base_samples)) if len(item.base_samples) > 0 else 1.0
            base_disp = max(1e-4, base_disp)

            norm_res = (item.y_true - base_loc) / base_disp
            v_vec = pos_token * np.clip(norm_res, -3.0, 3.0)
            v_tokens.append(v_vec)

        K = np.dot(np.array(k_tokens), self.W_k)
        V = np.dot(np.array(v_tokens), self.W_v)

        # Raw attention logits: Q K^T / sqrt(d_k)
        scale_factor = 1.0 / math.sqrt(self.embed_dim)
        raw_scores = np.dot(Q, K.T) * scale_factor

        # Subtract delay mask penalties
        delay_mask = np.array(delay_penalties).reshape(1, -1)  # shape (1, num_feedbacks)
        masked_scores = raw_scores - delay_mask

        # Numerical stable Softmax across feedbacks (axis=1)
        max_scores = np.max(masked_scores, axis=1, keepdims=True)
        exp_scores = np.exp(masked_scores - max_scores)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-12)

        # Attention output: shape (num_targets, embed_dim)
        context = np.dot(attn_weights, V)
        out = np.dot(context, self.W_o)

        return out, attn_weights


class HypothesisAttnH3_Corrector:
    """
    Hypothesis Attn-H3: Dynamic Self-Attention Cross-Horizon Adapter.
    """

    def __init__(
        self,
        horizons: List[int],
        embed_dim: int = 8,
        learning_rate: float = 0.05,
        age_decay_gamma: float = 0.08,
        overlap_discount_beta: float = 0.5,
        max_correction: float = 4.0,
        seed: int = 42,
    ):
        self.horizons = sorted(list(horizons))
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.age_decay_gamma = age_decay_gamma
        self.overlap_discount_beta = overlap_discount_beta
        self.max_correction = max_correction

        self.attn_module = CrossHorizonAttentionModule(
            embed_dim=embed_dim,
            num_heads=2,
            seed=seed,
        )

        # Online head projecting attention output vector to scalar mean and log-scale
        self.head_mean = np.zeros(embed_dim, dtype=np.float64)
        self.head_scale = np.zeros(embed_dim, dtype=np.float64)
        self.local_bias = {h: 0.0 for h in self.horizons}
        self.local_log_scale = {h: 0.0 for h in self.horizons}

        # Store latest attention matrix for interpretability analysis
        self.latest_attention_matrix: Optional[np.ndarray] = None
        self.latest_attention_horizons: List[int] = []

    def get_corrections(self, h: int) -> Tuple[float, float]:
        """Compute current scalar mean shift and scale multiplier for horizon h."""
        pos_token = self.attn_module.compute_positional_token(h)
        delta_mu = float(np.dot(pos_token, self.head_mean) + self.local_bias[h])
        delta_mu = float(np.clip(delta_mu, -self.max_correction, self.max_correction))

        log_scale = float(np.dot(pos_token, self.head_scale) + self.local_log_scale[h])
        log_scale = float(np.clip(log_scale, -1.6, 1.6))
        scale_mult = float(math.exp(log_scale))

        return delta_mu, scale_mult

    def adapt_samples(
        self,
        base_samples: np.ndarray,
        base_location: float,
        h: int,
    ) -> np.ndarray:
        delta_mu, scale_mult = self.get_corrections(h)
        return (base_location + delta_mu) + scale_mult * (base_samples - base_location)

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
    ) -> Dict[str, float]:
        if not feedback_items:
            return {"k_eff": 0.0, "step_norm": 0.0}

        k = len(feedback_items)

        # Forward pass through attention mechanism
        out_vectors, attn_weights = self.attn_module.forward(
            target_horizons=self.horizons,
            feedback_items=feedback_items,
            current_t=current_t,
            age_decay_gamma=self.age_decay_gamma,
        )

        self.latest_attention_matrix = attn_weights
        self.latest_attention_horizons = [item.horizon for item in feedback_items]

        # Kish effective sample size calculation
        raw_weights = [
            math.exp(-self.age_decay_gamma * max(0, current_t - item.origin_t))
            for item in feedback_items
        ]
        sum_w = sum(raw_weights)
        sum_sq = sum(w ** 2 for w in raw_weights)
        k_eff = (sum_w ** 2) / (sum_sq + 1e-8)

        overlap_factor = (k_eff / k) ** self.overlap_discount_beta
        eta_eff = self.learning_rate * overlap_factor

        # Compute gradient updates on head_mean and head_scale
        grad_mean = np.zeros(self.embed_dim, dtype=np.float64)
        grad_scale = np.zeros(self.embed_dim, dtype=np.float64)

        for idx, h in enumerate(self.horizons):
            # Attention feature vector for horizon h
            feat_h = out_vectors[idx]

            # Collect errors from feedback items that match horizon h
            matching_errors = []
            for item in feedback_items:
                if item.horizon == h:
                    base_loc = getattr(item, "base_location", None)
                    if base_loc is None:
                        base_loc = float(np.median(item.base_samples)) if len(item.base_samples) > 0 else 0.0
                    d_mu, s_mult = self.get_corrections(h)
                    res = item.y_true - (base_loc + d_mu)
                    matching_errors.append(res)
                    # Local bias leak
                    self.local_bias[h] += 0.02 * res

            if matching_errors:
                avg_err = float(np.mean(matching_errors))
                grad_mean += -avg_err * feat_h

        # Update heads with gradient clipping
        step_m = -eta_eff * grad_mean
        norm_m = float(np.linalg.norm(step_m))
        if norm_m > 1.2:
            step_m = (step_m / norm_m) * 1.2

        self.head_mean += step_m

        # Projection onto compact ball
        total_norm = float(np.linalg.norm(self.head_mean))
        if total_norm > 8.0:
            self.head_mean = (self.head_mean / total_norm) * 8.0

        return {
            "k_eff": float(k_eff),
            "step_norm": float(norm_m),
            "head_norm": float(np.linalg.norm(self.head_mean)),
        }
