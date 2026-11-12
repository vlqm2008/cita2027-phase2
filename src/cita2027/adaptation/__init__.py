"""Adaptation package for multi-horizon probabilistic adaptation research."""

from cita2027.adaptation.bayesian_online_posterior import (
    BayesianCrossHorizonPosteriorTracker,
)
from cita2027.adaptation.causal_cross_horizon_discovery import (
    CausalCrossHorizonAdapter,
    OnlineCausalHorizonGraph,
)
from cita2027.adaptation.changepoint_martingale import (
    ChangePointMartingaleAdapter,
    StreamingChangePointMartingale,
)
from cita2027.adaptation.correctors import HypothesisH1_SharedStateCorrector
from cita2027.adaptation.dirichlet_process_adaptation import (
    DirichletProcessMixtureAdapter,
    DPComponent,
)
from cita2027.adaptation.extreme_conformal_tails import (
    ExtremeValueConformalAdapter,
    OnlineExtremeConformalCalibrator,
)
from cita2027.adaptation.federated_adaptation import (
    FederatedAgentNode,
    FederatedCrossHorizonCoordinator,
)
from cita2027.adaptation.feedback_queue import (
    CausalFeedbackQueue,
    ForecastFeedbackItem,
)
from cita2027.adaptation.game_theoretic_adaptation import (
    GameTheoreticMinimaxAdapter,
)
from cita2027.adaptation.hyperbolic_adaptation import (
    HyperbolicCrossHorizonAdapter,
    mobius_addition,
    poincare_dist,
)
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.jump_diffusion_adaptation import (
    JumpDiffusionCrossHorizonAdapter,
    JumpDiffusionOnlineFilter,
)
from cita2027.adaptation.multifidelity_adaptation import (
    MultiFidelityCrossHorizonAdapter,
    MultiFidelityFeedbackItem,
)
from cita2027.adaptation.neural_sde_adaptation import (
    ContinuousTimeFeedbackItem,
    NeuralSDECrossHorizonAdapter,
)
from cita2027.adaptation.riemannian_natural_gradient import (
    RiemannianNaturalGradientAdapter,
)

__all__ = [
    "CausalFeedbackQueue",
    "ForecastFeedbackItem",
    "HypothesisH1_SharedStateCorrector",
    "HypothesisH2_AgeWeightedCorrector",
    "HypothesisH3_OverlapAwareCorrector",
    "FederatedAgentNode",
    "FederatedCrossHorizonCoordinator",
    "JumpDiffusionOnlineFilter",
    "JumpDiffusionCrossHorizonAdapter",
    "OnlineCausalHorizonGraph",
    "CausalCrossHorizonAdapter",
    "BayesianCrossHorizonPosteriorTracker",
    "StreamingChangePointMartingale",
    "ChangePointMartingaleAdapter",
    "RiemannianNaturalGradientAdapter",
    "MultiFidelityFeedbackItem",
    "MultiFidelityCrossHorizonAdapter",
    "OnlineExtremeConformalCalibrator",
    "ExtremeValueConformalAdapter",
    "HyperbolicCrossHorizonAdapter",
    "poincare_dist",
    "mobius_addition",
    "GameTheoreticMinimaxAdapter",
    "DirichletProcessMixtureAdapter",
    "DPComponent",
    "NeuralSDECrossHorizonAdapter",
    "ContinuousTimeFeedbackItem",
]
