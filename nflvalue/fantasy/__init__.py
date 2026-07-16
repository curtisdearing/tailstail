"""Leakage-safe fantasy-football projections and correlated simulations.

The fantasy package intentionally sits beside the prop engine.  It reuses the
same public NFL inputs but owns its scoring, model, calibration, simulation,
and season/trade contracts so a sportsbook-oriented change cannot silently
alter fantasy output.
"""

from .config import LineupRules, ModelConfig, ScoringRules, SimulationConfig
from .scoring import score_components

__all__ = [
    "LineupRules",
    "ModelConfig",
    "ScoringRules",
    "SimulationConfig",
    "score_components",
]
