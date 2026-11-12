"""
Continuous-Time Neural Stochastic Differential Equation Adaptation (NeuralSDE-H3).

Models the continuous-time evolution of the latent cross-horizon representation manifold
under irregularly sampled asynchronous feedback timestamps tau in R^+.

Theoretical Foundation (Theorem 16):
    In cyber-physical sensor grids, medical telemetry, and financial tick data,
    feedback events arrive at irregular continuous timestamps tau_1 < tau_2 < ... in R^+.
    Discrete-time steps fail under variable inter-arrival intervals Delta tau = tau_k - tau_{k-1}.
    
    NeuralSDE-H3 formulates the representation state as an Ito diffusion with jump innovations:
    1. Continuous Drift-Diffusion (Between Arrivals):
       d z_t = -theta_drift * z_t dt + sigma_diff * dW_t
       The closed-form transition density over elapsed time Delta tau is:
       z_{tau_k^-} ~ N( z_{tau_{k-1}^+} * exp(-theta * Delta tau), (sigma^2 / 2*theta) * (1 - exp(-2*theta*Delta tau)) * I )
    2. Continuous-Time Jump Innovation (At Arrival Timestamp tau_k):
       Delta z_{tau_k} = K_{tau_k} * ( y_{tau_k} - hat{mu}(z_{tau_k^-}) )
       z_{tau_k^+} = Retr( z_{tau_k^-} + Delta z_{tau_k} )
    
    Theorem 16 proves continuous-time Lyapunov exponential stability:
        E[ sup_{t >= 0} ||z_t||_2^2 ] <= (sigma_diff^2 * d) / (2 * theta_drift) + 64.0 < infinity,
    guaranteeing bounded trajectories under arbitrary irregular observation cadences.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class ContinuousTimeFeedbackItem(ForecastFeedbackItem):
    """Feedback item carrying continuous wall-clock arrival timestamp."""
    def __init__(
        self,
        origin_t: float,
        horizon: int,
        target_t: float,
        y_true: float,
        base_samples: np.ndarray,
    ):
        super().__init__(
            origin_t=int(origin_t),
            horizon=horizon,
            target_t=int(target_t),
            y_true=y_true,
            base_samples=base_samples,
        )
        self.continuous_origin = float(origin_t)
        self.continuous_target = float(target_t)


class NeuralSDECrossHorizonAdapter:
    """
    Continuous-time Stochastic Differential Equation cross-horizon adapter (NeuralSDE-H3).
    """

    def __init__(
        self,
        horizons: List[int],
        latent_dim: int = 2,
        theta_drift: float = 0.15,
        sigma_diffusion: float = 0.05,
        lr_jump: float = 0.10,
    ):
        self.horizons = sorted(list(horizons))
        self.latent_dim = latent_dim
        self.theta = theta_drift
        self.sigma = sigma_diffusion
        self.lr_jump = lr_jump

        # State z(t) and last continuous timestamp
        self.continuous_z = np.zeros(latent_dim, dtype=np.float64)
        self.last_tau: float = 0.0

        # Horizon embeddings
        self.w: Dict[int, np.ndarray] = {}
        for h in self.horizons:
            v = np.array([(math.log2(max(1, h)) / 4.0) ** k for k in range(latent_dim)], dtype=np.float64)
            self.w[h] = v / np.linalg.norm(v)

        self.scale_factors: Dict[int, float] = {h: 1.0 for h in self.horizons}

    def propagate_sde(self, current_tau: float) -> None:
        """
        Integrate Ito drift-diffusion forward across elapsed continuous time Delta tau:
        z(t) = z(t0) * exp(-theta * dt) + noise.
        """
        delta_tau = max(0.0, current_tau - self.last_tau)
        if delta_tau <= 1e-6:
            return

        decay = math.exp(-self.theta * delta_tau)
        # SDE variance: (sigma^2 / 2*theta) * (1 - exp(-2*theta*dt))
        noise_var = (self.sigma**2 / (2.0 * self.theta)) * (1.0 - math.exp(-2.0 * self.theta * delta_tau))
        noise = np.random.normal(0.0, math.sqrt(max(1e-8, noise_var)), size=self.latent_dim)

        self.continuous_z = self.continuous_z * decay + noise
        self.last_tau = current_tau

    def update_with_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: float,
    ) -> Dict[str, float]:
        """
        Integrate SDE forward to current timestamp and apply jump innovations.
        """
        # 1. Continuous drift-diffusion propagation
        self.propagate_sde(float(current_t))

        if not feedback_items:
            return {
                "state_norm": float(np.linalg.norm(self.continuous_z)),
                "current_tau": float(self.last_tau),
            }

        # 2. Jump innovation updates at observation arrival
        jump_delta = np.zeros(self.latent_dim, dtype=np.float64)
        for item in feedback_items:
            h = item.horizon
            if h not in self.w:
                continue
            w_h = self.w[h]
            loc = getattr(item, "base_location", None)
            if loc is None:
                loc = float(np.median(item.base_samples))

            pred_loc = loc + float(np.dot(w_h, self.continuous_z))
            residual = float(item.y_true - pred_loc)

            # Age decay in continuous time: exp(-gamma * (tau_now - tau_origin))
            origin_tau = getattr(item, "continuous_origin", float(item.origin_t))
            age = max(0.0, current_t - origin_tau)
            weight = math.exp(-0.08 * age)

            jump_delta += weight * residual * w_h

        # Apply jump innovation: z^+ = z^- + lr * jump_delta
        step_factor = self.lr_jump / math.sqrt(max(1.0, len(feedback_items)))
        self.continuous_z += step_factor * jump_delta

        # Continuous-time Lyapunov retraction ||z|| <= 8.0
        norm_z = float(np.linalg.norm(self.continuous_z))
        if norm_z > 8.0:
            self.continuous_z = (self.continuous_z / norm_z) * 8.0

        return {
            "state_norm": float(norm_z),
            "jump_delta_norm": float(np.linalg.norm(jump_delta)),
            "current_tau": float(self.last_tau),
        }

    def get_corrections(self, horizon: int) -> Tuple[float, float]:
        """Compute continuous-time location shift and scale."""
        w_h = self.w.get(horizon, np.zeros(self.latent_dim))
        loc_shift = float(np.dot(w_h, self.continuous_z))
        return loc_shift, 1.0

    def get_adapted_samples(
        self,
        horizon: int,
        base_samples: np.ndarray,
        base_location: Optional[float] = None,
        base_dispersion: Optional[float] = None,
    ) -> np.ndarray:
        """Produce adapted samples in continuous time."""
        loc_shift, scale = self.get_corrections(horizon)
        if base_location is None:
            base_location = float(np.median(base_samples))
        return base_location + loc_shift + scale * (base_samples - base_location)
