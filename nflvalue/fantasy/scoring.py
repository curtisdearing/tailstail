"""Translate simulated football events into configurable fantasy points."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import ScoringRules


COMPONENT_COLUMNS = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "fumbles_lost",
)


def _value(components: Mapping[str, object], key: str) -> np.ndarray:
    return np.asarray(components.get(key, 0.0), dtype=float)


def score_components(
    components: Mapping[str, object], rules: ScoringRules | None = None
) -> np.ndarray:
    """Score scalars or arrays of football outcomes.

    The returned NumPy value has the broadcast shape of the supplied fields.
    The function is deliberately pure so historical targets and Monte Carlo
    samples use exactly the same scoring implementation.
    """

    r = rules or ScoringRules()
    passing_yards = _value(components, "passing_yards")
    rushing_yards = _value(components, "rushing_yards")
    receiving_yards = _value(components, "receiving_yards")
    points = (
        passing_yards * r.passing_yard
        + _value(components, "passing_tds") * r.passing_td
        + _value(components, "passing_interceptions") * r.interception
        + rushing_yards * r.rushing_yard
        + _value(components, "rushing_tds") * r.rushing_td
        + _value(components, "receptions") * r.reception
        + receiving_yards * r.receiving_yard
        + _value(components, "receiving_tds") * r.receiving_td
        + (
            _value(components, "passing_2pt_conversions")
            + _value(components, "rushing_2pt_conversions")
            + _value(components, "receiving_2pt_conversions")
        )
        * r.two_point
        + _value(components, "fumbles_lost") * r.fumble_lost
    )
    if r.passing_300_bonus:
        points = points + (passing_yards >= 300) * r.passing_300_bonus
    if r.rushing_100_bonus:
        points = points + (rushing_yards >= 100) * r.rushing_100_bonus
    if r.receiving_100_bonus:
        points = points + (receiving_yards >= 100) * r.receiving_100_bonus
    return np.asarray(points, dtype=float)


def add_fantasy_points(frame, rules: ScoringRules | None = None, output: str = "fantasy_points"):
    """Return a copy of a pandas frame with a score built from raw events."""

    out = frame.copy()
    components = {column: out[column] for column in COMPONENT_COLUMNS if column in out.columns}
    out[output] = score_components(components, rules)
    return out
