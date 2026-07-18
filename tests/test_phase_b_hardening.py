"""Phase B invariants: fail loudly, run deterministically, reject bad frames.

Each test here corresponds to a defect that was live in the tree before Phase B
and would have produced a plausible-looking published number from a broken run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import contracts, montecarlo  # noqa: E402
from nflvalue.failures import (  # noqa: E402
    ConfigError,
    SchemaViolation,
    SourceFetchError,
    SourceRejected,
    SourceTimeout,
)
from nflvalue.reproducibility import canonical_csv_sha256  # noqa: E402
from nflvalue.sources import _http  # noqa: E402


# --------------------------------------------------------------------------- #
# B.1 -- every failure mode surfaces a typed error
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen_sequence(monkeypatch, outcomes):
    """Feed ``urlopen`` a scripted list of exceptions / payloads."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        index = calls["n"]
        calls["n"] += 1
        outcome = outcomes[min(index, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    return calls


def _http_error(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("http://x", code, "boom", headers, None)


def test_transient_5xx_is_retried_then_succeeds(monkeypatch):
    slept = []
    calls = _urlopen_sequence(monkeypatch, [_http_error(503), _http_error(503), b'{"ok": 1}'])
    out = _http.get_json("http://x", source="test", sleep=slept.append)
    assert out == {"ok": 1}
    assert calls["n"] == 3
    assert len(slept) == 2 and slept[0] < slept[1]      # exponential backoff


def test_client_error_is_not_retried(monkeypatch):
    """Retrying a 401 burns API credit and delays the real error."""
    calls = _urlopen_sequence(monkeypatch, [_http_error(401)])
    with pytest.raises(SourceRejected) as caught:
        _http.get_json("http://x", source="oddsapi", sleep=lambda s: None)
    assert calls["n"] == 1
    assert "HTTP 401" in str(caught.value)
    assert caught.value.source == "oddsapi"


def test_retry_after_header_overrides_backoff(monkeypatch):
    slept = []
    _urlopen_sequence(monkeypatch, [_http_error(429, retry_after=4), b"{}"])
    _http.get_json("http://x", source="discord", sleep=slept.append)
    assert slept == [4.0]


def test_exhausted_timeouts_raise_typed_error_with_attempt_log(monkeypatch):
    _urlopen_sequence(monkeypatch, [TimeoutError("timed out")])
    with pytest.raises(SourceTimeout) as caught:
        _http.get_json("http://x", attempts=3, source="nflverse", sleep=lambda s: None)
    assert len(caught.value.attempts) == 3
    assert "3 attempt(s) failed" in str(caught.value)


def test_unparseable_body_is_a_source_failure_not_a_crash(monkeypatch):
    _urlopen_sequence(monkeypatch, [b"<html>maintenance</html>"])
    with pytest.raises(SourceFetchError, match="unparseable body"):
        _http.get_json("http://x", source="espn", sleep=lambda s: None)


def test_corrupt_config_raises_instead_of_reverting_to_defaults(tmp_path, monkeypatch):
    from nflvalue import config as cfgmod
    bad = tmp_path / "config.json"
    bad.write_text('{"prop_markets": [')
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(bad))
    with pytest.raises(ConfigError, match="not valid JSON"):
        cfgmod.load_config()


def test_absent_file_still_falls_back_but_corrupt_one_does_not(tmp_path):
    from nflvalue import config as cfgmod
    assert cfgmod.load_json(str(tmp_path / "nope.json"), {"a": 1}) == {"a": 1}
    corrupt = tmp_path / "ratings.json"
    corrupt.write_text("{oops")
    with pytest.raises(ConfigError):
        cfgmod.load_json(str(corrupt), {"a": 1})


def test_weather_failure_is_not_reported_as_calm_outdoor_conditions(monkeypatch):
    """`{"dome": False}` on error was indistinguishable from a real reading."""
    from nflvalue.sources import weather

    def boom(*args, **kwargs):
        raise SourceTimeout("weather", "https://open-meteo", [])

    monkeypatch.setattr(weather, "get_json", boom)
    outdoor = next(name for name, stadium in weather.factmod.STADIUMS.items()
                   if not stadium.get("dome"))
    with pytest.raises(SourceFetchError):
        weather.forecast_for_game(outdoor, "2026-09-14T17:00:00Z")


def test_discord_partial_send_reports_how_far_it_got(monkeypatch):
    from nflvalue import notify
    posted = {"n": 0}

    def fake_request(url, **kwargs):
        posted["n"] += 1
        if posted["n"] == 2:
            raise SourceTimeout("discord", url, [])
        return b""

    monkeypatch.setattr(notify._http, "request_bytes", fake_request)
    monkeypatch.setattr(notify, "resolve_webhook", lambda: "http://hook")
    monkeypatch.setattr(notify, "build_messages", lambda payload: [{"content": "a"},
                                                                   {"content": "b"},
                                                                   {"content": "c"}])
    with pytest.raises(SourceFetchError, match="1 already posted"):
        notify.post_weekly({"season": 2026, "week": 1, "publish": True, "leans": [{}]},
                           dry_run=False)


# --------------------------------------------------------------------------- #
# B.2 -- determinism
# --------------------------------------------------------------------------- #
def _matchup():
    home = {"off": 2.0, "def": -1.0}
    away = {"off": -0.5, "def": 0.5}
    # The real shipped priors, so the test exercises production shapes.
    priors = json.loads(
        (ROOT / "data" / "league_priors.json").read_text())
    return home, away, priors


def test_montecarlo_is_reproducible_under_a_fixed_seed():
    home, away, priors = _matchup()
    a = montecarlo.simulate(home, away, priors, spread_line=-3.0, total_line=44.5, n=4000, seed=7)
    b = montecarlo.simulate(home, away, priors, spread_line=-3.0, total_line=44.5, n=4000, seed=7)
    assert a == b


def test_montecarlo_default_seed_is_not_entropy():
    """It defaulted to None -- OS entropy -- so published picks moved per run."""
    home, away, priors = _matchup()
    a = montecarlo.simulate(home, away, priors, spread_line=-3.0, n=2000)
    b = montecarlo.simulate(home, away, priors, spread_line=-3.0, n=2000)
    assert a == b
    assert montecarlo.simulate(home, away, priors, spread_line=-3.0, n=2000, seed=99) != a


def test_derived_seeds_differ_per_game_but_repeat_exactly():
    first = montecarlo.derive_seed(montecarlo.DEFAULT_SEED, "2026_01_KC_BAL")
    assert first == montecarlo.derive_seed(montecarlo.DEFAULT_SEED, "2026_01_KC_BAL")
    assert first != montecarlo.derive_seed(montecarlo.DEFAULT_SEED, "2026_01_SF_SEA")


def test_derived_seed_survives_a_different_hash_salt():
    """``hash()`` is PYTHONHASHSEED-salted: a seed built from it only LOOKS fixed."""
    script = ("import sys; sys.path.insert(0, %r); "
              "from nflvalue import montecarlo as m; "
              "print(m.derive_seed(m.DEFAULT_SEED, 'KC'))" % str(ROOT))
    outputs = set()
    for salt in ("0", "1", "12345"):
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"},
                                check=True)
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"seed varied with PYTHONHASHSEED: {outputs}"


from tests.test_pipeline_weekly import env  # noqa: E402,F401 - shared isolation fixture


def test_weekly_pipeline_double_run_is_byte_identical(env, monkeypatch):
    """The B.2 requirement: run the weekly pipeline twice, compare content hashes.

    The clock is frozen because wall-clock stamps are the pipeline's only
    non-model source of run-to-run difference; freezing it isolates the
    question actually being asked (are the NUMBERS stable?).
    """
    import pipeline_weekly as pwmod
    from nflvalue import freshness
    from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs

    frozen = "2026-09-10T12:00:00Z"
    monkeypatch.setattr(freshness, "stamp_now", lambda: frozen)
    monkeypatch.setattr(pwmod, "stamp_now", lambda: frozen)

    feeds = {"injury_rows": [], "injuries_fetched_at": frozen,
             "sleeper_df": None, "sleeper_fetched_at": frozen}

    def run_once():
        result = pwmod.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                                inject_feeds=dict(feeds))
        leans = pd.DataFrame(result.get("leans") or [])
        if leans.empty:
            return "EMPTY", result.get("publish")
        keys = [c for c in ("game_id", "player_id", "market") if c in leans.columns]
        return canonical_csv_sha256(leans, row_keys=keys), result.get("publish")

    first = run_once()
    second = run_once()
    assert first == second, f"weekly pipeline is not reproducible: {first} vs {second}"


# --------------------------------------------------------------------------- #
# B.3 -- schema contracts fail closed and name the offenders
# --------------------------------------------------------------------------- #
def test_duplicate_player_weeks_are_rejected_by_key():
    frame = pd.DataFrame({
        "season": [2026, 2026, 2026], "week": [1, 1, 2],
        "player_id": ["a", "a", "b"], "team": ["KC"] * 3, "role": ["WR"] * 3,
    })
    with pytest.raises(SchemaViolation) as caught:
        contracts.check_frame(frame, "player_week", **contracts.PLAYER_WEEK)
    message = str(caught.value)
    assert "uniqueness" in message
    assert "(2026, 1, 'a')" in message      # the offending key is named, not just counted


def test_missing_column_names_what_is_missing():
    with pytest.raises(SchemaViolation, match=r"missing required column\(s\) \['role'\]"):
        contracts.check_frame(pd.DataFrame({"season": [1], "week": [1], "player_id": ["a"],
                                            "team": ["KC"]}),
                              "player_week", **contracts.PLAYER_WEEK)


def test_dtype_contract_rejects_a_stringified_numeric_column():
    frame = pd.DataFrame({"season": ["2026"], "week": [1], "player_id": ["a"],
                          "team": ["KC"], "role": ["WR"]})
    with pytest.raises(SchemaViolation, match="expected numeric"):
        contracts.check_frame(frame, "player_week", **contracts.PLAYER_WEEK)


def test_null_key_is_rejected():
    frame = pd.DataFrame({"season": [2026], "week": [1], "player_id": [None],
                          "team": ["KC"], "role": ["WR"]})
    with pytest.raises(SchemaViolation, match="null values"):
        contracts.check_frame(frame, "player_week", **contracts.PLAYER_WEEK)


def test_empty_frame_is_rejected_where_rows_are_required():
    with pytest.raises(SchemaViolation, match="requires rows"):
        contracts.check_frame(pd.DataFrame(), "stage_input", allow_empty=False)


def test_contract_returns_the_frame_so_it_cannot_be_left_unwired():
    frame = pd.DataFrame({"season": [2026], "week": [1], "team": ["KC"]})
    assert contracts.check_frame(frame, "team_week", **contracts.TEAM_WEEK) is frame


def test_offender_list_is_truncated_not_unbounded():
    frame = pd.DataFrame({"season": [2026] * 60, "week": [1] * 60,
                          "player_id": [f"p{i // 2}" for i in range(60)],
                          "team": ["KC"] * 60, "role": ["WR"] * 60})
    with pytest.raises(SchemaViolation, match=r"\+20 more"):
        contracts.check_frame(frame, "player_week", **contracts.PLAYER_WEEK)


# --------------------------------------------------------------------------- #
# B.4 -- one regression test per optimized hot path
# --------------------------------------------------------------------------- #
def _rolling_fixture(rows: int = 400, groups: int = 25) -> pd.DataFrame:
    """Ragged panel: uneven group lengths, gaps, NaNs, and single-row groups --
    the shapes where a rolling rewrite is most likely to diverge."""
    import numpy as np
    rng = np.random.default_rng(4242)
    frame = pd.DataFrame({
        "player_id": [f"p{i % groups}" for i in range(rows)],
        "value": rng.normal(10, 4, rows),
        "other": rng.integers(0, 30, rows).astype(float),
    })
    frame.loc[rng.choice(rows, 40, replace=False), "value"] = float("nan")
    frame = frame.sort_values(["player_id"], kind="mergesort").reset_index(drop=True)
    return frame


@pytest.mark.parametrize("how", ["mean", "ewm", "count"])
def test_vectorized_rolling_matches_the_transform_it_replaced(how):
    """The optimized path must be numerically identical to the slow original.

    This is the regression guard for the 1.8x feature-build speedup: it pins
    equivalence to the exact expression that was replaced, so a future pandas
    upgrade that changes groupby-rolling semantics fails here rather than
    silently moving every rolling feature in the model.
    """
    from nflvalue import features

    frame = _rolling_fixture()
    original = frame.groupby("player_id")["value"].transform(
        lambda s: features._rolling_shifted(s, how=how))
    optimized = features._rolling_shifted_grouped(frame, "player_id", "value", how=how)
    pd.testing.assert_series_equal(original, optimized, check_names=False)


def test_vectorized_rolling_matches_on_multi_key_groups():
    from nflvalue import features

    frame = _rolling_fixture()
    frame["role"] = ["WR", "TE"] * (len(frame) // 2)
    frame = frame.sort_values(["player_id", "role"], kind="mergesort").reset_index(drop=True)
    original = frame.groupby(["player_id", "role"])["value"].transform(features._rolling_shifted)
    optimized = features._rolling_shifted_grouped(frame, ["player_id", "role"], "value")
    pd.testing.assert_series_equal(original, optimized, check_names=False)


def test_vectorized_rolling_preserves_row_order_for_an_unsorted_frame():
    """Row alignment, not just values: groupby-rolling returns a re-indexed
    Series, and a silent misalignment would shift every feature by a row."""
    from nflvalue import features

    frame = _rolling_fixture().sample(frac=1.0, random_state=11).reset_index(drop=True)
    original = frame.groupby("player_id")["value"].transform(features._rolling_shifted)
    optimized = features._rolling_shifted_grouped(frame, "player_id", "value")
    pd.testing.assert_series_equal(original, optimized, check_names=False)


def test_rolling_features_never_see_the_current_week():
    """The optimization must not have quietly dropped the shift(1) leak guard."""
    from nflvalue import features

    frame = pd.DataFrame({"player_id": ["a"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    rolled = features._rolling_shifted_grouped(frame, "player_id", "value")
    assert pd.isna(rolled.iloc[0])                    # no history before game 1
    assert rolled.iloc[1] == pytest.approx(1.0)       # game 2 sees only game 1
    assert rolled.iloc[4] == pytest.approx(2.5)       # game 5 sees games 1-4
