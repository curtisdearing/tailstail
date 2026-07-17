"""Configuration contracts for fantasy modeling and simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ScoringRules:
    """Fantasy scoring rules.

    Defaults are conventional full PPR.  Bonuses are explicit instead of
    hidden in the model, which lets one event simulation serve many leagues.
    """

    reception: float = 1.0
    passing_yard: float = 0.04
    passing_td: float = 4.0
    interception: float = -2.0
    rushing_yard: float = 0.1
    rushing_td: float = 6.0
    receiving_yard: float = 0.1
    receiving_td: float = 6.0
    two_point: float = 2.0
    fumble_lost: float = -2.0
    passing_300_bonus: float = 0.0
    rushing_100_bonus: float = 0.0
    receiving_100_bonus: float = 0.0

    @classmethod
    def preset(cls, name: str) -> "ScoringRules":
        presets = {
            "ppr": cls(reception=1.0),
            "half_ppr": cls(reception=0.5),
            "standard": cls(reception=0.0),
        }
        try:
            return presets[name.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown scoring preset {name!r}; choose {sorted(presets)}") from exc

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Training controls chosen to keep walk-forward runs reproducible."""

    random_seed: int = 6102026
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE")
    model_families: tuple[str, ...] = ("bayesian", "gradient_boosting", "random_forest", "mlp")
    stack_validation_seasons: int = 3
    stack_weight_cap: float = 0.65
    stack_l2: float = 0.04
    min_train_rows: int = 120
    min_position_rows: int = 60
    conformal_alpha: float = 0.20
    fast: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.stack_weight_cap <= 1:
            raise ValueError("stack_weight_cap must be in (0, 1]")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal_alpha must be in (0, 1)")
        if self.stack_weight_cap * len(self.model_families) < 1:
            raise ValueError("stack_weight_cap makes sum-to-one weights impossible")


@dataclass(frozen=True)
class SimulationConfig:
    simulations: int = 10_000
    random_seed: int = 6102026
    target_share_concentration: float = 36.0
    carry_share_concentration: float = 32.0
    team_volume_cv: float = 0.12
    efficiency_cv: float = 0.20
    implied_points_offense_share: float = 0.78
    model_center_weight: float = 1.0
    # Scenario-mixture role-state engine (research flag; default OFF keeps the
    # simulator bit-identical to the validated baseline).
    role_scenario_mixture: bool = False
    role_width_inflation: float = 1.15
    role_width_budget: float = 0.975
    capture_team_totals: bool = False

    def __post_init__(self) -> None:
        if self.simulations < 100:
            raise ValueError("at least 100 simulations are required")
        if self.target_share_concentration <= 0 or self.carry_share_concentration <= 0:
            raise ValueError("Dirichlet concentrations must be positive")
        if not 0 <= self.model_center_weight <= 1:
            raise ValueError("model_center_weight must be in [0, 1]")
        if self.role_width_inflation < 0:
            raise ValueError("role_width_inflation cannot be negative")
        if not 0.5 <= self.role_width_budget <= 1.5:
            raise ValueError("role_width_budget must be in [0.5, 1.5]")


@dataclass(frozen=True)
class LineupRules:
    """Starting-lineup contract used by rest-of-season and trade analysis."""

    starters: Mapping[str, int] = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    )
    flex_positions: tuple[str, ...] = ("RB", "WR", "TE")

    def __post_init__(self) -> None:
        if any(int(v) < 0 for v in self.starters.values()):
            raise ValueError("starter counts cannot be negative")
