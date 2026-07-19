"""Tests for the factor-families study (wave 2).

Covers: the predeclared-count contract (code == frozen report header), as-of
safety of every constructed-feature pathway (shared trailing helper, terciles,
schedule-derived after-OT flag, built exposure frames when present), the
matched-effect estimator (sign recovery, NaN-strata exclusion regression), and
BH-FDR. Heavy artifacts under /tmp/exp_families/ are exercised only when they
exist so the suite stays green on a fresh checkout.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import factor_families_study as fam

ROOT = fam.ROOT
EXP = fam.EXP


# ---------------------------------------------------------------------------
# Predeclaration contract
# ---------------------------------------------------------------------------
def test_predeclared_battery_count_is_40():
    assert len(fam.BATTERY) == 40
    ids = [t["id"] for t in fam.BATTERY]
    assert len(set(ids)) == 40
    by_fam = {f: sum(1 for t in fam.BATTERY if t["family"] == f) for f in "ABCD"}
    assert by_fam == {"A": 17, "B": 12, "C": 10, "D": 1}


def test_report_header_predeclares_same_count():
    rp = os.path.join(ROOT, "reports", "factor_families_audit.md")
    with open(rp) as handle:
        text = handle.read()
    assert "exactly 40 tests" in text
    assert fam.MARKER in text
    # header (predeclaration) must precede any results
    assert text.find("PREDECLARATION") < text.find(fam.MARKER)


def test_battery_fields_well_formed():
    for t in fam.BATTERY:
        assert t["outcome"] in ("d_fp", "d_pass_yards", "d_rush_yards", "d_rec_yards",
                                "d_plays", "d_pass_att")
        assert t["exposure"].startswith("x_")
        assert t["level"] in ("player", "team")
        assert t["prior_sd"] > 0


# ---------------------------------------------------------------------------
# As-of safety: the shared trailing helper used by EVERY constructed
# trailing feature (opponent allowed-stats, offense trailing, PROE proto).
# ---------------------------------------------------------------------------
def _toy_team_frame():
    return pd.DataFrame({
        "team": ["A"] * 5 + ["B"] * 5,
        "season": [2023] * 3 + [2024] * 2 + [2023] * 5,
        "week": [1, 2, 3, 1, 2, 1, 2, 3, 4, 5],
        "v": [10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_ewm_trailing_is_strictly_prior():
    df = _toy_team_frame()
    out = fam._ewm_trailing(df, ["team"], ["team", "season", "week"], ["v"],
                            halflife=5, min_periods=1)
    a = out[out.team == "A"].sort_values(["season", "week"])
    # first game has no prior information
    assert np.isnan(a.tr_v.iloc[0])
    # second game sees exactly the first value, never its own
    assert a.tr_v.iloc[1] == pytest.approx(10.0)
    assert (a.tr_v.iloc[2] < 20.0) and (a.tr_v.iloc[2] > 10.0)


def test_ewm_trailing_future_change_does_not_leak_backwards():
    df = _toy_team_frame()
    base = fam._ewm_trailing(df, ["team"], ["team", "season", "week"], ["v"],
                             halflife=5, min_periods=1)
    mut = df.copy()
    mut.loc[(mut.team == "A") & (mut.season == 2024) & (mut.week == 2), "v"] = 9999.0
    out = fam._ewm_trailing(mut, ["team"], ["team", "season", "week"], ["v"],
                            halflife=5, min_periods=1)
    for wk_key in [(2023, 1), (2023, 2), (2023, 3), (2024, 1), (2024, 2)]:
        b = base[(base.team == "A") & (base.season == wk_key[0]) & (base.week == wk_key[1])].tr_v.iloc[0]
        m = out[(out.team == "A") & (out.season == wk_key[0]) & (out.week == wk_key[1])].tr_v.iloc[0]
        if np.isnan(b):
            assert np.isnan(m)
        else:
            assert b == pytest.approx(m)  # nothing at-or-before week changes


def test_ewm_trailing_min_periods_gate():
    df = _toy_team_frame()
    out = fam._ewm_trailing(df, ["team"], ["team", "season", "week"], ["v"],
                            halflife=5, min_periods=3)
    b = out[out.team == "B"].sort_values(["season", "week"])
    assert b.tr_v.isna().iloc[:3].all()      # games 1-3 lack 3 PRIOR games
    assert np.isfinite(b.tr_v.iloc[3])       # game 4 has exactly 3 prior


def test_tercile_within_week_maps_and_gates():
    n = 30
    df = pd.DataFrame({"season": 2023, "week": [1] * n + [2] * 3,
                       "val": list(range(n)) + [1, 2, 3]})
    x = fam._tercile_within_week(df, "val", min_valid=20)
    wk1 = x.iloc[:n]
    assert (wk1.iloc[:10] == 0.0).all()      # bottom tercile -> control
    assert (wk1.iloc[-10:] == 1.0).all()     # top tercile -> exposed
    assert wk1.iloc[12:18].isna().all()      # middle excluded
    assert x.iloc[n:].isna().all()           # week 2 fails min_valid


def test_after_ot_flag_is_previous_game_only():
    gm = fam.load_game_meta()
    g = gm[(gm.team == "KC") & (gm.season == 2023)].sort_values("gameday")
    # first tracked game of a (team, season) never inherits OT from the future
    assert g.after_ot.iloc[0] in (0.0,) or np.isnan(g.after_ot.iloc[0])
    ot_rows = gm[(gm.overtime == 1)]
    assert len(ot_rows) > 50
    # for a sample of OT games, the SAME team's next game that season has after_ot=1
    checked = 0
    for _, r in ot_rows.head(40).iterrows():
        nxt = gm[(gm.team == r.team) & (gm.season == r.season) & (gm.week > r.week)]
        if len(nxt):
            assert nxt.sort_values("week").after_ot.iloc[0] == 1.0
            checked += 1
    assert checked > 10


# ---------------------------------------------------------------------------
# Matched-effect estimator
# ---------------------------------------------------------------------------
def _synth(effect=2.0, n_weeks=40, per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_weeks):
        for i in range(per):
            x = float(i % 2)
            stratum = i % 3
            rows.append(dict(season=2019 + w % 6, week=w % 18 + 1, team=f"T{i}",
                             x=x, s=stratum,
                             y=rng.normal(effect * x + stratum, 1.0)))
    return pd.DataFrame(rows)


def test_matched_effect_recovers_known_effect():
    d = _synth(effect=2.0)
    r = fam.matched_effect(d, "x", "y", ["s"], prior_sd=5.0, n_boot=300, seed=1)
    assert not r["empty"]
    assert r["effect"] == pytest.approx(2.0, abs=0.25)
    assert r["ci_lo"] < 2.0 < r["ci_hi"]
    assert r["p"] < 0.01
    assert r["n_exposed"] > 0 and r["n_control"] > 0
    assert set(r["per_season"]) == {2019, 2020, 2021, 2022, 2023, 2024}


def test_matched_effect_nan_strata_are_excluded_not_wrapped():
    """Regression: NaN strata used to ngroup() to -1 and silently np.add.at-wrap
    into the LAST stratum. Poisoned rows must not move the estimate."""
    d = _synth(effect=1.0, seed=2)
    poison = d.sample(200, random_state=3).copy()
    poison["s"] = np.nan
    poison["y"] = 1e6
    r_clean = fam.matched_effect(d, "x", "y", ["s"], prior_sd=5.0, n_boot=200, seed=4)
    r_mixed = fam.matched_effect(pd.concat([d, poison], ignore_index=True),
                                 "x", "y", ["s"], prior_sd=5.0, n_boot=200, seed=4)
    assert r_mixed["effect"] == pytest.approx(r_clean["effect"], abs=1e-9)
    assert r_mixed["n_missing"] >= 200


def test_matched_effect_empty_cohort_reports_ns():
    d = _synth(seed=5)
    d["x"] = np.nan
    r = fam.matched_effect(d, "x", "y", ["s"], prior_sd=1.0, n_boot=50)
    assert r["empty"] and r["n_exposed"] == 0 and r["n_control"] == 0


def test_bh_fdr_matches_reference():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = fam._bh(p)
    ref = np.array([0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.10571428571428572,
                    0.216, 0.216, 0.216])
    assert np.allclose(q, ref, atol=1e-9)
    assert (np.diff(q[np.argsort(p)]) >= -1e-12).all()  # monotone in p
    assert (q >= p).all() and (q <= 1.0).all()


# ---------------------------------------------------------------------------
# Built frames (exercised only when the chunked run has produced them)
# ---------------------------------------------------------------------------
needs_frames = pytest.mark.skipif(
    not os.path.exists(os.path.join(EXP, "player_frame.parquet")),
    reason="run --stage frames first")


@needs_frames
def test_player_frame_exposures_binary_and_present():
    pf = pd.read_parquet(os.path.join(EXP, "player_frame.parquet"))
    for t in fam.BATTERY:
        if t["level"] == "player":
            assert t["exposure"] in pf.columns, t["id"]
            vals = pf[t["exposure"]].dropna().unique()
            assert set(vals) <= {0.0, 1.0}, t["id"]


@needs_frames
def test_player_frame_rest_flags_consistent():
    pf = pd.read_parquet(os.path.join(EXP, "player_frame.parquet"))
    sr = pf[pf.x_short_rest == 1.0]
    assert (sr.rest <= 5).all() and (sr.week > 1).all()
    pb = pf[pf.x_post_bye == 1.0]
    assert (pb.rest >= 12).all() and (pb.week > 1).all()
    ctrl = pf[pf.x_short_rest == 0.0]
    assert ctrl.rest.between(6, 11).all()


@needs_frames
def test_player_frame_outcomes_are_vs_trailing():
    pf = pd.read_parquet(os.path.join(EXP, "player_frame.parquet"))
    s = pf.dropna(subset=["d_fp"]).head(500)
    assert np.allclose(s.d_fp, s.fantasy_points - s.pre_fantasy_points_ewm4)


@needs_frames
def test_battery_results_when_present_cover_all_40():
    bp = os.path.join(EXP, "battery_results.json")
    if not os.path.exists(bp):
        pytest.skip("battery not yet run")
    with open(bp) as handle:
        res = json.load(handle)
    assert len(res) == 40
    for t in fam.BATTERY:
        r = res[t["id"]]
        assert "q" in r and "battery_status" in r
        if not r.get("empty"):
            assert r["n_exposed"] > 0 and r["n_control"] > 0
