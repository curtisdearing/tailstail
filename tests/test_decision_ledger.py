"""Phase A invariants: append-only decisions, as-of safety, refusal below n=150.

These tests exist to make a specific class of self-deception mechanically
impossible: editing a recorded decision, letting closing information reach a
decision row, treating correlated selections as independent, publishing a
number from a sample too small to support one, and relaxing the kill bar after
seeing the results.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import decision_ledger as ledger  # noqa: E402
from nflvalue import killcheck  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "phase_a.db"))
    yield c
    c.close()


def _decide(conn, **overrides):
    kwargs = dict(
        decided_at_utc="2026-09-10T12:00:00Z", season=2026, week=1,
        game_id="2026_01_KC_BAL", player_id="00-A1", player_name="Travis Kelce",
        market="receiving_yards", side="over", book="draftkings",
        price=1.87, point=52.5, model_prob=0.58, devig_prob=0.5105,
        model_version="ranker-2026.1", commit_sha="deadbeef",
    )
    kwargs.update(overrides)
    return ledger.record_decision(conn, **kwargs)


def _close(ts, prob, point=52.5, kind="devig", n_books=3):
    def fetcher(conn, decision, cutoff_ts):
        return {"ts": ts, "prob": prob, "point": point,
                "n_books": n_books, "prob_kind": kind}
    return fetcher


# --------------------------------------------------------------------------- #
# A.1 -- immutability
# --------------------------------------------------------------------------- #
def test_decision_id_is_the_content_hash_and_verifies(conn):
    row = _decide(conn)
    assert row["decision_id"] == ledger.canonical_record_sha256(
        row, ledger.DECISION_HASH_FIELDS)
    assert ledger.verify_decision(conn, row["decision_id"])
    assert row["edge"] == pytest.approx(0.58 - 0.5105)


def test_decision_hash_distinguishes_none_from_zero(conn):
    """An untagged hash would merge these two into one decision."""
    a = _decide(conn, point=0.0)
    b = _decide(conn, point=None)
    assert a["decision_id"] != b["decision_id"]


def test_update_is_refused_at_the_database_level(conn):
    row = _decide(conn)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE decision_snapshots SET devig_prob=0.9 WHERE decision_id=?",
                     (row["decision_id"],))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM decision_snapshots WHERE decision_id=?",
                     (row["decision_id"],))
    conn.rollback()
    assert ledger.verify_decision(conn, row["decision_id"])


def test_upsert_cannot_launder_a_rewrite(conn):
    """INSERT OR REPLACE deletes first, so the append-only trigger catches it."""
    row = _decide(conn)
    tampered = {**{k: row[k] for k in row}, "devig_prob": 0.40}
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        dbmod.upsert(conn, "decision_snapshots", [tampered], ["decision_id"])
    conn.rollback()


def test_identical_replay_is_idempotent(conn):
    row = _decide(conn)
    assert _decide(conn)["decision_id"] == row["decision_id"]
    assert int(conn.execute("SELECT COUNT(*) FROM decision_snapshots").fetchone()[0]) == 1


def test_id_reused_for_different_content_is_refused(conn):
    """A row whose stored content no longer hashes to its id is not accepted
    as a replay -- it is history that has been tampered with."""
    row = _decide(conn, player_id="00-REAL")
    forged = dict(row)
    forged["decision_id"] = ledger.canonical_record_sha256(
        {**row, "player_id": "00-FORGED"}, ledger.DECISION_HASH_FIELDS)
    forged["devig_prob"] = 0.40
    cols = list(forged.keys())
    conn.execute(f"INSERT INTO decision_snapshots ({', '.join(cols)}) "
                 f"VALUES ({', '.join(['?'] * len(cols))})",
                 [forged[c] for c in cols])
    conn.commit()
    assert ledger.verify_decision(conn, forged["decision_id"]) is False
    with pytest.raises(ledger.AppendOnlyViolation, match="different content"):
        _decide(conn, player_id="00-FORGED")


# --------------------------------------------------------------------------- #
# A.2 -- as-of safety
# --------------------------------------------------------------------------- #
def test_decision_row_carries_no_closing_columns(conn):
    row = _decide(conn)
    ledger.capture_close(conn, row["decision_id"], kickoff_ts="2026-09-14T17:00:00Z",
                         close_fetcher=_close("2026-09-14T16:30:00Z", 0.5497))
    reread = ledger.load_decision(conn, row["decision_id"])
    assert set(reread) == set(row)
    assert not any(key.startswith("close") for key in reread)
    assert reread["devig_prob"] == pytest.approx(0.5105)
    assert ledger.verify_decision(conn, row["decision_id"])


def test_close_before_decision_is_not_a_close(conn):
    row = _decide(conn)
    result = ledger.capture_close(
        conn, row["decision_id"], kickoff_ts="2026-09-14T17:00:00Z",
        close_fetcher=_close("2026-09-10T09:00:00Z", 0.60))
    assert result is None
    assert ledger.unresolved_count(conn) == 1


def test_kickoff_before_decision_raises(conn):
    row = _decide(conn)
    with pytest.raises(ledger.AsOfViolation):
        ledger.capture_close(conn, row["decision_id"],
                             kickoff_ts="2026-09-10T11:00:00Z",
                             close_fetcher=_close("2026-09-10T10:00:00Z", 0.60))


def test_snapshot_after_cutoff_raises(conn):
    row = _decide(conn)
    with pytest.raises(ledger.AsOfViolation, match="after cutoff"):
        ledger.capture_close(conn, row["decision_id"],
                             kickoff_ts="2026-09-14T17:00:00Z",
                             close_fetcher=_close("2026-09-14T16:59:00Z", 0.60))


def test_devig_and_raw_implied_are_never_pooled(conn):
    row = _decide(conn, market="anytime_td", prob_kind="raw_implied")
    with pytest.raises(ledger.AsOfViolation, match="prob_kind"):
        ledger.capture_close(conn, row["decision_id"],
                             kickoff_ts="2026-09-14T17:00:00Z",
                             close_fetcher=_close("2026-09-14T16:30:00Z", 0.55, kind="devig"))


def test_closing_snapshots_are_append_only(conn):
    row = _decide(conn)
    ledger.capture_close(conn, row["decision_id"], kickoff_ts="2026-09-14T17:00:00Z",
                         close_fetcher=_close("2026-09-14T16:30:00Z", 0.5497))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE closing_snapshots SET close_devig_prob=0.99")
    conn.rollback()


# --------------------------------------------------------------------------- #
# A.3 -- CLV + bootstrap determinism
# --------------------------------------------------------------------------- #
def test_end_to_end_pair_math(conn):
    row = _decide(conn)
    ledger.capture_close(conn, row["decision_id"], kickoff_ts="2026-09-14T17:00:00Z",
                         close_fetcher=_close("2026-09-14T16:30:00Z", 0.5497, point=54.5))
    pairs = ledger.resolved_pairs(conn)
    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert pair["clv_prob"] == pytest.approx(0.5497 - 0.5105)
    assert pair["point_move"] == pytest.approx(2.0)
    assert bool(pair["beat_close"]) is True


def _seed_pairs(conn, n, clv_value, weeks=6):
    for i in range(n):
        week = (i % weeks) + 1
        row = _decide(
            conn, week=week, game_id=f"2026_{week:02d}_G{i}", player_id=f"P{i}",
            decided_at_utc=f"2026-09-10T12:{i % 60:02d}:00Z", devig_prob=0.50)
        ledger.capture_close(
            conn, row["decision_id"], kickoff_ts="2026-09-14T17:00:00Z",
            close_fetcher=_close("2026-09-14T16:30:00Z", 0.50 + clv_value))


def test_bootstrap_is_deterministic_under_a_fixed_seed(conn):
    _seed_pairs(conn, 60, 0.01)
    pairs = ledger.resolved_pairs(conn)
    # Heterogeneous values: a constant column has a degenerate (zero-width)
    # bootstrap interval and would pass a determinism check vacuously.
    pairs["clv_prob"] = [0.01 + 0.004 * ((i % 7) - 3) for i in range(len(pairs))]
    first = ledger.paired_week_bootstrap_clv(pairs, iterations=2000, random_seed=7)
    second = ledger.paired_week_bootstrap_clv(pairs, iterations=2000, random_seed=7)
    assert first == second
    different = ledger.paired_week_bootstrap_clv(pairs, iterations=2000, random_seed=8)
    assert different["mean_clv_prob"] == first["mean_clv_prob"]   # observed is not resampled
    assert different["ci95"] != first["ci95"]                     # the interval is
    assert first["ci95"][0] < first["mean_clv_prob"] < first["ci95"][1]


def test_bootstrap_blocks_by_week_not_by_selection(conn):
    """Resampling weeks must be wider than pretending selections are independent."""
    _seed_pairs(conn, 120, 0.01, weeks=4)
    pairs = ledger.resolved_pairs(conn)
    pairs.loc[pairs["week"] == 1, "clv_prob"] = 0.06     # one week carries the signal
    blocked = ledger.paired_week_bootstrap_clv(pairs, iterations=4000, random_seed=11)
    naive = pairs.assign(week=range(len(pairs)))
    unblocked = ledger.paired_week_bootstrap_clv(naive, iterations=4000, random_seed=11)
    width = lambda r: r["ci95"][1] - r["ci95"][0]
    assert width(blocked) > width(unblocked)


# --------------------------------------------------------------------------- #
# A.4 -- refusal below the precommitted sample
# --------------------------------------------------------------------------- #
def test_refuses_to_emit_any_estimate_below_threshold(conn):
    _seed_pairs(conn, 149, 0.02)
    report = killcheck.forward_clv_report(conn)
    assert report["verdict"] == "INSUFFICIENT_SAMPLE"
    assert report["n_resolved"] == 149
    for forbidden in ("mean_clv_prob", "ci95", "beat_close_rate",
                      "probability_nonpositive", "mean_point_move"):
        assert forbidden not in report, f"{forbidden} leaked below the sample gate"
    assert "trend" in report["detail"] or "not a direction" in report["detail"]


def test_emits_a_verdict_at_exactly_the_threshold(conn):
    _seed_pairs(conn, 150, 0.02)
    report = killcheck.forward_clv_report(conn, iterations=2000)
    assert report["n_resolved"] == 150
    assert report["verdict"] == "GO"
    assert report["mean_clv_prob"] == pytest.approx(0.02, abs=1e-6)
    assert report["beat_close_rate"] == 1.0


def test_negative_clv_at_threshold_is_no_go(conn):
    _seed_pairs(conn, 150, -0.01)
    report = killcheck.forward_clv_report(conn, iterations=2000)
    assert report["verdict"] == "NO_GO"
    assert "KILL CRITERION MET" in report["detail"]
    assert "stop staking" in report["detail"]


def test_unresolved_decisions_are_counted_but_make_no_claim(conn):
    _seed_pairs(conn, 10, 0.02)
    _decide(conn, player_id="P-UNRESOLVED", game_id="2026_01_NO_CLOSE")
    report = killcheck.forward_clv_report(conn)
    assert report["n_resolved"] == 10
    assert report["n_unresolved"] == 1
    assert report["verdict"] == "INSUFFICIENT_SAMPLE"


# --------------------------------------------------------------------------- #
# A.5 -- the bar cannot be moved silently
# --------------------------------------------------------------------------- #
PRECOMMITMENT_PATH = ROOT / "analysis" / "clv_killcheck_precommitment.json"
PROTOCOL_PATH = ROOT / "analysis" / "accuracy_protocol.json"

# Pinned so that relaxing the bar requires editing this line in the same diff.
PINNED_PRECOMMITMENT_SHA256 = (
    "38ee6b9b46157f86a0e8c453bf56d43c7b1696c6a68853d8ab6ad08e9e99f237"
)


def test_precommitment_matches_the_protocol_it_was_declared_against():
    bar = killcheck.load_precommitment()
    assert bar["min_sample"] == 150
    assert bar["min_mean_clv"] == 0.0
    assert bar["min_beat_close_rate"] == 0.52
    assert bar["horizon"]["abandon_if_unmet_by_season"] == 2026
    assert bar["horizon"]["abandon_if_unmet_by_week"] == 18
    assert "stop staking" in bar["horizon"]["action_on_failure"]


def test_precommitment_file_content_is_pinned():
    import hashlib
    actual = hashlib.sha256(PRECOMMITMENT_PATH.read_bytes()).hexdigest()
    assert actual == PINNED_PRECOMMITMENT_SHA256, (
        "The CLV kill-check precommitment changed. This is allowed only with a "
        "new precommitment_id and a sample counted from zero — update this pin "
        "in the same commit and say so in the accuracy ledger.")


def test_relaxing_the_protocol_threshold_breaks_the_precommitment(tmp_path):
    """Loosen the bar in a copy of the protocol; evaluation must refuse."""
    protocol = json.loads(PROTOCOL_PATH.read_text())
    protocol["forward_clv"]["minimum_resolved"] = 40
    relaxed = tmp_path / "accuracy_protocol.json"
    relaxed.write_text(json.dumps(protocol, indent=2))
    with pytest.raises(killcheck.PrecommitmentViolation, match="invalidates the sample"):
        killcheck.load_precommitment(protocol_path=str(relaxed))


def test_thresholds_are_not_restated_in_the_precommitment_file():
    """Two copies of a number drift; the precommitment must reference, not copy."""
    text = PRECOMMITMENT_PATH.read_text()
    payload = json.loads(text)
    assert payload["thresholds_source"].endswith("#/forward_clv")
    assert "minimum_beat_close_rate" not in text
    assert "0.52" not in text
