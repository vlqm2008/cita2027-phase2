"""Adaptation package for multi-horizon probabilistic adaptation research."""

from cita2027.adaptation.bayesian_online_posterior import (
    BayesianCrossHorizonPosteriorTracker,
)
from cita2027.adaptation.causal_cross_horizon_discovery import (
    CausalCrossHorizonAdapter,
    OnlineCausalHorizonGraph,
)
from cita2027.adaptation.federated_adaptation import (
    FederatedAgentNode,
    FederatedCrossHorizonCoordinator,
)
from cita2027.adaptation.feedback_queue import (
    AsynchronousFeedbackQueue,
    ForecastFeedbackItem,
)
from cita2027.adaptation.hypothesis_h1 import HypothesisH1_SharedCorrector
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeDecayedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.jump_diffusion_adaptation import (
    JumpDiffusionCrossHorizonAdapter,
    JumpDiffusionOnlineFilter,
)

__all__ = [
    "AsynchronousFeedbackQueue",
    "ForecastFeedbackItem",
    "HypothesisH1_SharedCorrector",
    "HypothesisH2_AgeDecayedCorrector",
    "HypothesisH3_OverlapAwareCorrector",
    "FederatedAgentNode",
    "FederatedCrossHorizonCoordinator",
    "JumpDiffusionOnlineFilter",
    "JumpDiffusionCrossHorizonAdapter",
    "OnlineCausalHorizonGraph",
    "CausalCrossHorizonAdapter",
    "BayesianCrossHorizonPosteriorTracker",
]
