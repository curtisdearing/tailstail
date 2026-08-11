"""Data-independent tests for scripts/build_role_multipliers.py.

They confirm the committed regeneration recipe for the role-shock simulator's
frozen ``DEFAULT_STATE_MULTIPLIERS`` table reproduces the documented invariants
(``stable`` normalized to identity per channel, non-negative robust variance,
every position x state cell present) on a synthetic training frame -- so the
table is reproducible from committed code, not an ephemeral /tmp script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.role_state import ACTIVE_STATES, POSITIONS
from scripts import build_role_multipliers as brm

STATE_RATIO = {
    "limited_decrease": 0.5,
    "stable": 1.0,
    "moderate_increase": 1.7,
    "major_increase": 2.6,
}


def _synthetic_frame(seed: int = 11, per_cell: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for position in POSITIONS:
        for state in ACTIVE_STATES:
            for _ in range(per_cell):
                ratio = STATE_RATIO[state] * rng.uniform(0.9, 1.1)
                pre_opp = rng.uniform(6.0, 18.0)
                opp = pre_opp * ratio
                pre_ts = rng.uniform(0.05, 0.25)
                pre_cs = rng.uniform(0.05, 0.25)
                rows.append({
                    "season": 2021, "week": rng.integers(1, 18), "player_id": f"p{pid}",
                    "position": position, "role_state_label": state,
                    "opportunities": opp, "pre_opportunities_ewm4": pre_opp,
                    "target_share_calc": pre_ts * ratio, "pre_target_share_calc_ewm4": pre_ts,
                    "carry_share": pre_cs * ratio, "pre_carry_share_ewm4": pre_cs,
                    "model_eligible": True, "played": True,
                })
                pid += 1
    return pd.DataFrame(rows)


def test_stable_normalized_ratio_centers_stable_near_one():
    frame = _synthetic_frame()
    normalized = brm.stable_normalized_opportunity_ratio(frame)
    stable = frame["role_state_label"].eq("stable")
    assert abs(float(normalized[stable].median()) - 1.0) < 0.05


def test_robust_var_is_zero_for_constant_series():
    assert brm._robust_var(pd.Series([3.0, 3.0, 3.0, 3.0])) == 0.0


def test_robust_var_matches_closed_form_for_known_iqr():
    # 1..10: numpy q1=3.25, q3=7.75, IQR=4.5; robust var = (4.5/1.349)**2.
    expected = ((7.75 - 3.25) / brm.ROBUST_IQR_TO_SIGMA) ** 2
    assert brm._robust_var(pd.Series(range(1, 11))) == pytest.approx(expected)


def test_variance_table_is_non_negative_and_complete():
    frame = _synthetic_frame()
    variance = brm.opportunity_variance_table(frame, max_season=2021)
    assert len(variance) == len(POSITIONS) * len(ACTIVE_STATES)
    assert all(v >= 0.0 for v in variance.values())
    # the synthetic frame has genuine within-state spread, so the computed
    # variance must not collapse to all zeros (guards a no-op implementation)
    assert any(v > 0.0 for v in variance.values())


def test_built_table_has_stable_identity_and_all_cells():
    frame = _synthetic_frame()
    table = brm.build_state_multiplier_table(frame, max_season=2021)
    assert len(table) == len(POSITIONS) * len(ACTIVE_STATES)
    for position in POSITIONS:
        stable = table[f"{position}:stable"]
        for channel in ("opportunity", "target", "carry"):
            assert stable[channel] == pytest.approx(1.0)
        assert stable["opportunity_var"] >= 0.0
    # a clear role increase scales opportunity above stable
    assert table["RB:moderate_increase"]["opportunity"] > 1.2


def test_python_literal_round_trips():
    frame = _synthetic_frame()
    table = brm.build_state_multiplier_table(frame, max_season=2021)
    literal = brm._as_python_literal(table)
    namespace: dict[str, object] = {}
    exec(literal, namespace)
    assert set(namespace["DEFAULT_STATE_MULTIPLIERS"]) == set(table)
