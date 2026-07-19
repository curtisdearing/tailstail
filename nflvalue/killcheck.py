"""The kill-check: after ~150 logged leans, does this thing actually work?

PROP_SHORTLISTER_SPEC.md §5 / PREMORTEM.md: if the forward leans don't beat
the market after ~150 logged props, the composite is not finding real prop
edges -- the honest response is to revert to "projection/entertainment tool"
and stop staking, not to explain the sample away. This module renders that
verdict mechanically so nobody (including future-us) can argue with it.

Verdicts:
  INSUFFICIENT_SAMPLE  n < min_sample: keep logging, draw no conclusion.
  GO                   avg CLV > 0 AND positive-CLV rate >= 52%: the leans
                       systematically beat the close -- consistent with edge.
  NO_GO                anything else at n >= min_sample: KILL CRITERION MET.

The naive baseline is built in: CLV measures our entry against the SAME
side at the SAME book-consensus close, i.e. exactly "take the model's side
at the posted number" -- a strategy with zero timing skill scores ~0 here
(minus noise), so beating 0 with a >=52% hit rate is the bar.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

from . import db as dbmod
from . import decision_ledger
from .clv import rolling_clv

PROTOCOL_PATH = os.path.join(dbmod.ROOT, "analysis", "accuracy_protocol.json")
PRECOMMITMENT_PATH = os.path.join(
    dbmod.ROOT, "analysis", "clv_killcheck_precommitment.json")


class PrecommitmentViolation(RuntimeError):
    """Raised when the declared bar no longer matches what was declared."""


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_precommitment(protocol_path: str = PROTOCOL_PATH,
                       precommitment_path: str = PRECOMMITMENT_PATH) -> Dict:
    """Load the bar the strategy was committed to BEFORE any result was seen.

    Thresholds live in ``accuracy_protocol.json`` (single source of truth) and
    the horizon lives in the precommitment file, which records the protocol
    hash it was declared against. If someone edits the thresholds after seeing
    results, that hash stops matching and every evaluation raises instead of
    quietly reporting against an easier bar. The check is not that the bar
    CANNOT move -- it is that it cannot move silently.
    """
    with open(precommitment_path) as handle:
        precommitment = json.loads(handle.read())
    actual = _sha256(protocol_path)
    declared = precommitment["protocol_sha256"]
    if actual != declared:
        raise PrecommitmentViolation(
            "accuracy_protocol.json has changed since the CLV kill check was "
            f"precommitted (declared {declared[:12]}..., found {actual[:12]}...). "
            "Relaxing a threshold after seeing results invalidates the sample: "
            "issue a new precommitment_id and start the count from zero.")
    with open(protocol_path) as handle:
        protocol = json.loads(handle.read())
    forward = protocol["forward_clv"]
    return {
        "precommitment_id": precommitment["precommitment_id"],
        "declared_at_utc": precommitment["declared_at_utc"],
        "protocol_sha256": actual,
        "min_sample": int(forward["minimum_resolved"]),
        "min_mean_clv": float(forward["minimum_mean_probability_clv"]),
        "min_beat_close_rate": float(forward["minimum_beat_close_rate"]),
        "horizon": precommitment["horizon"],
        "relaxation_policy": precommitment["relaxation_policy"],
    }


def _bar() -> Dict:
    try:
        return load_precommitment()
    except (FileNotFoundError, KeyError):  # pragma: no cover - legacy fallback
        return {"min_sample": 150, "min_beat_close_rate": 0.52, "min_mean_clv": 0.0}


DEFAULT_MIN_SAMPLE = _bar()["min_sample"]
POSITIVE_RATE_BAR = _bar()["min_beat_close_rate"]


def report(conn=None, min_sample: int = DEFAULT_MIN_SAMPLE, window: int = 50) -> Dict:
    conn = conn or dbmod.connect()
    stats = rolling_clv(conn, window=window)
    n = stats["n"]

    leans_n = int(dbmod.query_df(
        conn, "SELECT COUNT(*) AS n FROM leans WHERE status='active'").iloc[0]["n"])

    if n < min_sample:
        verdict = "INSUFFICIENT_SAMPLE"
        detail = (f"{n} leans with resolved CLV (of {leans_n} logged; "
                  f"{min_sample} needed). Keep logging; no conclusion yet — "
                  "and no staking conclusions either way.")
    elif (stats["lifetime_mean"] or 0) > 0 and (stats["positive_rate"] or 0) >= POSITIVE_RATE_BAR:
        verdict = "GO"
        detail = (f"Avg CLV {stats['lifetime_mean']:+.4f} prob-points over {n} leans, "
                  f"{stats['positive_rate']:.0%} beat the close (bar: {POSITIVE_RATE_BAR:.0%}). "
                  "Consistent with real edge. Staking still means quarter-to-half Kelly on a "
                  "SHRUNK edge, hard per-bet cap, fixed monthly loss limit (spec §8).")
    else:
        verdict = "NO_GO"
        detail = (f"KILL CRITERION MET: avg CLV {stats['lifetime_mean']:+.4f}, positive rate "
                  f"{(stats['positive_rate'] or 0):.0%} over {n} leans — the leans do not beat "
                  "the close. Revert to projection/entertainment tool; stop staking. "
                  "(PROP_SHORTLISTER_SPEC.md §5.3 — this outcome was pre-committed.)")

    return {**stats, "leans_logged": leans_n, "min_sample": min_sample,
            "verdict": verdict, "detail": detail}


# --------------------------------------------------------------------------- #
# Phase A verdict: decision-snapshot ledger, precommitted bar, week-block CI
# --------------------------------------------------------------------------- #
def forward_clv_report(conn=None, *, iterations: int = 20_000,
                       random_seed: int = 6102026,
                       precommitment: Optional[Dict] = None) -> Dict:
    """Verdict over matched (decision, close) pairs from ``decision_ledger``.

    Below the precommitted minimum this returns NO estimate of any kind. The
    keys simply are not there, so a dashboard cannot render a mean, a rate, or
    a direction from a sample that has not earned one. "Trending positive" at
    n=40 is not a weak claim; it is a claim the data cannot support at all.
    """
    conn = conn or dbmod.connect()
    bar = precommitment or load_precommitment()
    pairs = decision_ledger.resolved_pairs(conn)
    n_resolved = int(len(pairs))
    n_unresolved = decision_ledger.unresolved_count(conn)

    base = {
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "min_sample": bar["min_sample"],
        "precommitment_id": bar["precommitment_id"],
        "protocol_sha256": bar["protocol_sha256"],
        "horizon": bar["horizon"],
    }

    if n_resolved < bar["min_sample"]:
        return {**base, "verdict": "INSUFFICIENT_SAMPLE", "detail": (
            f"{n_resolved} resolved decision/close pairs of the "
            f"{bar['min_sample']} precommitted ({n_unresolved} unresolved). "
            "No CLV estimate is produced at this sample size — not a point "
            "estimate, not a direction, not a trend. Keep logging.")}

    boot = decision_ledger.paired_week_bootstrap_clv(
        pairs, iterations=iterations, random_seed=random_seed)
    beat_close_rate = float(pairs["beat_close"].mean())
    passes = (boot["mean_clv_prob"] > bar["min_mean_clv"]
              and beat_close_rate >= bar["min_beat_close_rate"])
    stats = {**base, "mean_clv_prob": round(boot["mean_clv_prob"], 5),
             "ci95": [round(v, 5) for v in boot["ci95"]],
             "probability_nonpositive": round(boot["probability_nonpositive"], 4),
             "beat_close_rate": round(beat_close_rate, 4),
             "mean_point_move": round(float(pairs["point_move"].mean()), 3),
             "weeks": boot["weeks"], "iterations": boot["iterations"],
             "random_seed": boot["random_seed"]}

    if passes:
        return {**stats, "verdict": "GO", "detail": (
            f"Mean probability CLV {boot['mean_clv_prob']:+.4f} "
            f"(week-block 95% CI {boot['ci95'][0]:+.4f} to {boot['ci95'][1]:+.4f}), "
            f"{beat_close_rate:.0%} beat the close over {n_resolved} pairs "
            f"(bar: {bar['min_beat_close_rate']:.0%}). Consistent with real edge. "
            "Staking still means quarter-to-half Kelly on a SHRUNK edge, a hard "
            "per-bet cap, and a fixed monthly loss limit (spec §8).")}
    return {**stats, "verdict": "NO_GO", "detail": (
        f"KILL CRITERION MET: mean CLV {boot['mean_clv_prob']:+.4f}, "
        f"{beat_close_rate:.0%} beat the close over {n_resolved} pairs. The "
        "decisions do not beat the close. Revert to projection/entertainment "
        f"tool; stop staking. Precommitted {bar['declared_at_utc']} as "
        f"{bar['precommitment_id']} — this outcome was declared in advance.")}


def main() -> None:  # pragma: no cover - thin CLI
    print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
