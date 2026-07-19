"""Invariant tests for the role-shock scenario-mixture engine.

Two safety gates must hold before the mixture flag
(``SimulationConfig.role_scenario_mixture``) can ever be turned on:

1. **As-of safety** -- every pregame role predictor is built from strictly
   completed prior weeks.  Realized (post-game) opportunity outcomes may enter
   the pipeline only as *labels*, never as features, so nothing a predictor
   sees could depend on the week being predicted.
2. **Sum-to-team** -- the state-conditional share reallocation inside the
   scenario-mixture Monte Carlo conserves team volume exactly.  Applying any
   role-shock multiplier renormalizes across the whole team (modeled players
   plus the unmodeled-roster bucket), so a role shock can redistribute volume
   but can never manufacture or destroy it.

These are contract tests: they need no historical parquet frame and no
network.  The single classifier test skips cleanly where scikit-learn is
absent (it is a declared dependency, so CI runs it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy import role_state as rs
from nflvalue.fantasy import simulation as sim

# ---------------------------------------------------------------------------
# As-of safety -- pregame features never read the current or a future week
# ---------------------------------------------------------------------------

def _one_player(values: list[float]) -> pd.DataFrame:
    """A single player's weekly series in the value column ``x``."""
    return pd.DataFrame(
        {
            "player_id": ["p"] * len(values),
            "season": [2021] * len(values),
            "week": list(range(1, len(values) + 1)),
            "x": [float(v) for v in values],
        }
    )


def test_shifted_ewm_first_observation_is_nan():
    """Week 1 has no prior evidence, so the lagged EWM is NaN -- never itself."""
    out = rs._grouped_shifted_ewm(_one_player([10.0, 20.0, 30.0]), "x", halflife=3.0)
    assert np.isnan(out.iloc[0])
    # week 2 consumes only week 1 (single prior point) -> exactly 10.0
    assert out.iloc[1] == pytest.approx(10.0)


def test_shifted_ewm_never_reads_current_or_future_week():
    """Mutating the LAST week changes no feature: a week never enters its own
    (or an earlier) lagged value.  This is the core no-look-ahead guarantee."""
    base = rs._grouped_shifted_ewm(_one_player([10.0, 20.0, 30.0, 40.0]), "x", 3.0)
    bumped = rs._grouped_shifted_ewm(_one_player([10.0, 20.0, 30.0, 999.0]), "x", 3.0)
    # Row 3's feature is built from weeks 1-3 only, so the whole series is equal.
    assert np.allclose(base.to_numpy(), bumped.to_numpy(), equal_nan=True)


def test_shifted_ewm_information_flows_only_forward():
    """A change in an early week DOES move later weeks -- the feature is a real
    lag, not a constant that ignores history."""
    a = rs._grouped_shifted_ewm(_one_player([10.0, 20.0, 30.0, 40.0]), "x", 3.0)
    b = rs._grouped_shifted_ewm(_one_player([999.0, 20.0, 30.0, 40.0]), "x", 3.0)
    assert not np.allclose(a.to_numpy()[1:], b.to_numpy()[1:])  # weeks 2+ shifted


def test_shifted_ewm_ignores_did_not_play_weeks():
    """DNP weeks arrive as NaN (``_obs`` is masked by ``played``); ignore_na
    means they are skipped rather than treated as a zero-usage observation."""
    out = rs._grouped_shifted_ewm(_one_player([10.0, np.nan, 30.0]), "x", 3.0)
    # week 3's lagged EWM sees weeks 1-2 = {10, skip} -> 10.0, not NaN or 30.0
    assert out.iloc[2] == pytest.approx(10.0)


def test_role_predictors_exclude_realized_outcomes():
    """The classifier's predictor list must contain no post-game outcome -- the
    guard that keeps realized results on the label side of the as-of clock."""
    features = set(rs.role_model_features())
    realized_outcomes = {
        "opportunities", "target_share_calc", "carry_share", "offense_pct",
        "route_share", "targets", "carries", "attempts", "receptions",
        "completions", "fantasy_points", "points", "yards", "role_state_label",
    }
    leaked = features & realized_outcomes
    assert not leaked, f"realized outcomes leaked into predictors: {sorted(leaked)}"


def test_realized_labels_are_pregame_separated():
    """Labels are defined from REALIZED opportunity change vs the pregame EWM,
    with the documented thresholds and an inactive override."""
    frame = pd.DataFrame(
        {
            "played": [1, 1, 1, 1, 0],
            "opportunities": [10.0, 2.0, 17.0, 25.0, 0.0],
            "pre_opportunities_ewm4": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    labels = rs.realized_role_state_labels(frame).tolist()
    assert labels == [
        "stable",             # delta 0
        "limited_decrease",   # delta -8 <= -5
        "moderate_increase",  # delta +7 >= 5
        "major_increase",     # delta +15 >= 12
        "inactive",           # did not play -> overrides delta 0
    ]
    # The label genuinely depends on the realized column, confirming the split.
    shifted = frame.assign(opportunities=frame["opportunities"] + 100.0)
    assert rs.realized_role_state_labels(shifted).iloc[0] == "major_increase"


# ---------------------------------------------------------------------------
# Sum-to-team -- role reallocation conserves team volume exactly
# ---------------------------------------------------------------------------

def _team(n_players: int = 4, n_draws: int = 512):
    base = np.array([0.28, 0.18, 0.11, 0.05])[:n_players]
    available = np.ones((n_draws, n_players))
    return base, available


def test_draw_shares_conserves_team_volume():
    """Every simulated draw sums to 1.0 across modeled players + the unmodeled
    bucket: the whole team's share is accounted for, none created or lost."""
    rng = np.random.default_rng(1)
    base, available = _team()
    shares = sim._draw_shares(rng, base, available, concentration=36.0)
    assert shares.shape == (available.shape[0], base.shape[0] + 1)  # + bucket col
    assert np.allclose(shares.sum(axis=1), 1.0)


def test_draw_shares_conserved_under_role_multiplier():
    """A role-shock multiplier (even a large or heterogeneous one) redistributes
    share but leaves the per-draw team total at exactly 1.0."""
    rng = np.random.default_rng(2)
    base, available = _team()
    uniform_shock = np.full_like(available, 2.7)          # everyone major-increase
    shares = sim._draw_shares(rng, base, available, 36.0, multiplier=uniform_shock)
    assert np.allclose(shares.sum(axis=1), 1.0)

    mixed = np.tile(np.array([2.7, 0.3, 1.0, 1.75]), (available.shape[0], 1))
    shares_mixed = sim._draw_shares(rng, base, available, 36.0, multiplier=mixed)
    assert np.allclose(shares_mixed.sum(axis=1), 1.0)


def test_draw_shares_zeros_unavailable_players_and_conserves():
    """An unavailable player receives zero share in every draw, and the volume
    that would have been his is redistributed -- the team total stays 1.0."""
    rng = np.random.default_rng(3)
    base, available = _team()
    available[:, 1] = 0.0  # player index 1 inactive across all draws
    shares = sim._draw_shares(rng, base, available, 36.0)
    assert np.allclose(shares[:, 1], 0.0)
    assert np.allclose(shares.sum(axis=1), 1.0)


def test_state_multiplier_table_keeps_stable_at_identity():
    """The ``stable`` state must be exactly 1.0 on every channel for every
    position; a stable player is never rescaled, so unshocked teams reproduce
    baseline behaviour."""
    table = rs.state_multiplier_table()
    stable = rs.ACTIVE_STATES.index("stable")
    for position in ("QB", "RB", "WR", "TE"):
        for channel in ("target", "carry", "opportunity"):
            arr = table[position][channel]
            assert arr.shape == (len(rs.ACTIVE_STATES),)
            assert arr[stable] == pytest.approx(1.0)


def test_relative_multipliers_normalize_stable_to_one():
    """The raw->relative normalizer pins ``stable`` to 1.0 per channel and
    expresses active states as cell/stable ratios."""
    raw = {}
    for position in ("QB", "RB", "WR", "TE"):
        raw[f"{position}:limited_decrease"] = {"opportunity": 0.4, "target": 0.5, "carry": 0.6, "n": 100}
        raw[f"{position}:stable"] = {"opportunity": 0.8, "target": 0.9, "carry": 0.7, "n": 1000}
        raw[f"{position}:moderate_increase"] = {"opportunity": 1.6, "target": 1.5, "carry": 1.4, "n": 100}
        raw[f"{position}:major_increase"] = {"opportunity": 2.4, "target": 2.2, "carry": 2.0, "n": 100}
    rel = rs.relative_state_multipliers(raw)
    for position in ("QB", "RB", "WR", "TE"):
        for channel in ("opportunity", "target", "carry"):
            assert rel[f"{position}:stable"][channel] == pytest.approx(1.0)
        assert rel[f"{position}:moderate_increase"]["opportunity"] == pytest.approx(round(1.6 / 0.8, 3))


def test_role_mixture_inputs_fall_back_to_stable_when_probs_missing():
    """A player with no pregame state probabilities collapses to certain-stable
    (zero shock mass), so enabling the flag with missing probabilities is a
    no-op rather than an uncontrolled perturbation."""
    players = pd.DataFrame({"player_id": ["a", "b"], "position": ["RB", "WR"]})
    probs, _target_rows, _carry_rows, up_mass, down_mass, has_probs = sim._role_mixture_inputs(players)
    assert np.allclose(up_mass, 0.0)
    assert np.allclose(down_mass, 0.0)
    assert not has_probs.any()
    # each fallback row is a valid distribution centred on stable
    assert np.allclose(probs.sum(axis=1), 1.0)

    with_probs = players.assign(
        p_state_inactive=0.0, p_state_limited_decrease=0.0, p_state_stable=0.5,
        p_state_moderate_increase=0.5, p_state_major_increase=0.0,
    )
    _, _, _, up2, _down2, has2 = sim._role_mixture_inputs(with_probs)
    assert has2.all()
    assert np.allclose(up2, 0.5)  # moderate + major increase mass


@pytest.mark.parametrize("seed", [0, 11])
def test_role_model_predict_proba_sums_to_one(seed):
    """P(role state) is a proper distribution over the five states, with every
    state column present even when a class is unseen in training."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(seed)
    n = 400
    frame = pd.DataFrame({f: rng.normal(size=n) for f in rs.role_model_features()})
    frame["season"] = rng.integers(2018, 2022, size=n)
    frame["week"] = rng.integers(1, 18, size=n)
    frame["player_id"] = [f"p{i}" for i in range(n)]
    frame["position"] = rng.choice(["QB", "RB", "WR", "TE"], size=n)
    frame["model_eligible"] = True
    frame["role_state_label"] = rng.choice(list(rs.ROLE_STATES), size=n)

    model = rs.RoleStateModel(random_seed=1).fit(frame, max_season=int(frame["season"].max()))
    probs = model.predict_proba(frame.head(64))
    assert list(probs.columns) == list(rs.STATE_PROB_COLUMNS)
    assert (probs.to_numpy() >= 0).all()
    assert np.allclose(probs.to_numpy().sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Frame contract -- structural guards the builder asserts on every frame
# ---------------------------------------------------------------------------

def _valid_contract_frame(n: int = 3) -> pd.DataFrame:
    columns = {name: np.zeros(n) for name in rs.ROLE_FEATURES}
    columns["season"] = [2021] * n
    columns["week"] = list(range(1, n + 1))
    columns["player_id"] = [f"p{i}" for i in range(n)]
    return pd.DataFrame(columns)


def test_contract_accepts_clean_frame():
    rs.assert_role_state_contract(_valid_contract_frame())  # must not raise


def test_contract_rejects_seat_cohort_overlap():
    """Short-notice and long-term vacancy cohorts are mutually exclusive by
    construction; an overlap signals a builder bug and must fail loudly."""
    frame = _valid_contract_frame()
    frame["seat_qb1_out_shortnotice"] = 1.0
    frame["seat_qb1_out_longterm"] = 1.0
    with pytest.raises(ValueError, match="overlap"):
        rs.assert_role_state_contract(frame)


def test_contract_rejects_duplicate_player_weeks():
    frame = _valid_contract_frame()
    frame.loc[1, ["season", "week", "player_id"]] = frame.loc[0, ["season", "week", "player_id"]].to_numpy()
    with pytest.raises(ValueError, match="duplicate"):
        rs.assert_role_state_contract(frame)


def test_contract_rejects_missing_columns():
    frame = _valid_contract_frame().drop(columns=["depth_rank_asof"])
    with pytest.raises(ValueError, match="missing columns"):
        rs.assert_role_state_contract(frame)
