import numpy as np

from nflvalue.fantasy.config import ScoringRules
from nflvalue.fantasy.scoring import score_components


def test_standard_ppr_translation_is_exact():
    points = score_components({
        "passing_yards": 300,
        "passing_tds": 2,
        "passing_interceptions": 1,
        "rushing_yards": 20,
        "receptions": 3,
        "receiving_yards": 40,
        "receiving_tds": 1,
        "fumbles_lost": 1,
    })
    assert float(points) == 31.0


def test_scoring_presets_reuse_same_events():
    events = {"receptions": np.array([0, 4]), "receiving_yards": np.array([50, 50])}
    ppr = score_components(events, ScoringRules.preset("ppr"))
    half = score_components(events, ScoringRules.preset("half_ppr"))
    standard = score_components(events, ScoringRules.preset("standard"))
    assert np.allclose(ppr, [5, 9])
    assert np.allclose(half, [5, 7])
    assert np.allclose(standard, [5, 5])


def test_bonus_rules_are_explicit():
    rules = ScoringRules(passing_300_bonus=3, rushing_100_bonus=2)
    points = score_components({"passing_yards": [299, 300], "rushing_yards": [100, 99]}, rules)
    assert np.allclose(points, [23.96, 24.9])
