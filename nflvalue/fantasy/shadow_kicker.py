"""Kicker projections -- SHADOW / RESEARCH ONLY.

Separate lane by construction.  This module imports no part of the frozen
QB/RB/WR/TE path (no models, no simulation, no projection_snapshot), writes to
its own directory, and stamps every artifact `status: "shadow"`.  Nothing here
may reach an optimal-lineup decision until the promotion gates in
`docs/K_SHADOW_MODEL_CARD.md` have been passed and recorded.

Method, deliberately plain for a first shadow pass: per-bucket attempt rates
and make rates from the kicker's own history, each shrunk toward the league
rate so a kicker with three 50-yard tries does not get a bespoke long-range
number.  Attempts are Poisson, makes Binomial, and the resulting stat line is
scored through the LIVE league contract -- no point value is written here.

Two refusals are load-bearing:
  * a kicker with no history is `unavailable`, never the league average wearing
    his name;
  * history at or after the target week is dropped before fitting, so a pregame
    projection cannot see its own result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .espn_contract import FIELD_GOAL_BUCKETS, LeagueContract
from .special_scoring import score_kicker

MODEL_VERSION = "k-shadow-0.1.0"

#: ESPN statId for the league's 0-39 field-goal tier; tests mutate it to prove
#: no point value is hardcoded in this module.
FG_0_39_STAT_ID = 80
STATUS_SHADOW = "shadow"
UTC = timezone.utc

#: nflverse per-bucket columns -> the contract bucket they roll up into.
#: The league pays 0-39 as one tier; nflverse splits it three ways.
MADE_COLUMNS = {
    "fg_made_0_39": ("fg_made_0_19", "fg_made_20_29", "fg_made_30_39"),
    "fg_made_40_49": ("fg_made_40_49",),
    "fg_made_50_59": ("fg_made_50_59",),
    "fg_made_60_plus": ("fg_made_60_",),
}
MISSED_COLUMNS = {
    "fg_made_0_39": ("fg_missed_0_19", "fg_missed_20_29", "fg_missed_30_39"),
    "fg_made_40_49": ("fg_missed_40_49",),
    "fg_made_50_59": ("fg_missed_50_59",),
    "fg_made_60_plus": ("fg_missed_60_",),
}
BUCKETS = tuple(MADE_COLUMNS)

#: Shrinkage strength, in pseudo-attempts of league-average evidence. Long
#: buckets are shrunk harder because a season yields very few 50+ tries.
ATTEMPT_PRIOR_WEEKS = {"fg_made_0_39": 6.0, "fg_made_40_49": 8.0,
                       "fg_made_50_59": 12.0, "fg_made_60_plus": 24.0}
MAKE_PRIOR_ATTEMPTS = {"fg_made_0_39": 12.0, "fg_made_40_49": 16.0,
                       "fg_made_50_59": 24.0, "fg_made_60_plus": 40.0}
PAT_PRIOR_WEEKS = 6.0
PAT_PRIOR_ATTEMPTS = 12.0


class ShadowKickerError(RuntimeError):
    """The lane refuses rather than emitting a number it cannot justify."""


def bucket_for(distance: float) -> str:
    """Contract bucket key for a field-goal distance, inclusive bounds."""
    for low, high, key in FIELD_GOAL_BUCKETS:
        if low <= distance <= high:
            return key
    raise ShadowKickerError(f"no bucket covers {distance}")


def score_line(line: Mapping[str, Any], contract: LeagueContract) -> float:
    """Score one kicker stat line through the league contract."""
    return score_kicker(line, contract)


def _sum(frame: pd.DataFrame, columns: Sequence[str]) -> float:
    total = 0.0
    for column in columns:
        if column in frame.columns:
            total += float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
    return total


def past_only(history: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows strictly before (season, week). The leakage boundary."""
    if history.empty:
        return history
    s = pd.to_numeric(history["season"], errors="coerce")
    w = pd.to_numeric(history["week"], errors="coerce")
    return history[(s < season) | ((s == season) & (w < week))]


def league_rates(history: pd.DataFrame) -> dict[str, Any]:
    """League-wide per-week attempt rates and per-attempt make rates."""
    weeks = max(len(history), 1)
    attempts, makes = {}, {}
    for bucket in BUCKETS:
        made = _sum(history, MADE_COLUMNS[bucket])
        missed = _sum(history, MISSED_COLUMNS[bucket])
        att = made + missed
        attempts[bucket] = att / weeks
        makes[bucket] = (made / att) if att > 0 else 0.0
    pat_made = _sum(history, ("pat_made",))
    pat_att = _sum(history, ("pat_att",))
    return {"attempts": attempts, "makes": makes,
            "pat_attempts": pat_att / weeks,
            "pat_make": (pat_made / pat_att) if pat_att > 0 else 0.0}


def kicker_rates(history: pd.DataFrame, player_id: str) -> dict[str, Any] | None:
    """Shrunk rates for one kicker, or None when he has no usable history."""
    own = history[history["player_id"].astype(str) == str(player_id)]
    if own.empty:
        return None
    league = league_rates(history)
    weeks = float(len(own))
    attempts, makes = {}, {}
    for bucket in BUCKETS:
        made = _sum(own, MADE_COLUMNS[bucket])
        missed = _sum(own, MISSED_COLUMNS[bucket])
        att = made + missed
        w_a = ATTEMPT_PRIOR_WEEKS[bucket]
        attempts[bucket] = (att + league["attempts"][bucket] * w_a) / (weeks + w_a)
        w_m = MAKE_PRIOR_ATTEMPTS[bucket]
        makes[bucket] = ((made + league["makes"][bucket] * w_m) / (att + w_m))
    pat_made = _sum(own, ("pat_made",))
    pat_att = _sum(own, ("pat_att",))
    pat_attempts = ((pat_att + league["pat_attempts"] * PAT_PRIOR_WEEKS)
                    / (weeks + PAT_PRIOR_WEEKS))
    pat_make = ((pat_made + league["pat_make"] * PAT_PRIOR_ATTEMPTS)
                / (pat_att + PAT_PRIOR_ATTEMPTS))
    return {"attempts": attempts, "makes": makes, "pat_attempts": pat_attempts,
            "pat_make": pat_make, "weeks_observed": int(weeks)}


def _percentiles(points: np.ndarray) -> dict[str, float]:
    q = np.percentile(points, [5, 25, 50, 75, 95])
    return {
        "mean": round(float(points.mean()), 4),
        "sd": round(float(points.std(ddof=1)), 4),
        "p05": round(float(q[0]), 4), "p25": round(float(q[1]), 4),
        "p50": round(float(q[2]), 4), "p75": round(float(q[3]), 4),
        "p95": round(float(q[4]), 4),
        "p_zero": round(float((points <= 0).mean()), 4),
    }


def project(history: pd.DataFrame, player_id: str, contract: LeagueContract, *,
            season: int, week: int, simulations: int = 10_000, seed: int = 6102026,
            active: bool = True, player_name: str | None = None,
            team: str | None = None) -> dict[str, Any]:
    """One kicker's scored point distribution, or an explicit refusal."""
    row: dict[str, Any] = {
        "player_id": str(player_id), "player_name": player_name,
        "team": team, "position": "K", "season": int(season), "week": int(week),
        "status": "unavailable", "unavailable_reason": None,
        "replacement": False, "distribution": None, "components": None,
        "simulations": int(simulations), "seed": int(seed),
    }
    if not active:
        row["unavailable_reason"] = "kicker is not active for this game"
        return row

    usable = past_only(history, season, week)
    rates = kicker_rates(usable, player_id) if not usable.empty else None
    if rates is None:
        row["unavailable_reason"] = (
            "no pre-kickoff history for this kicker; refusing to substitute a "
            "league-average kicker under his name")
        return row

    rng = np.random.default_rng([int(seed), abs(hash(str(player_id))) % (2 ** 31)])
    points = np.zeros(simulations, dtype=float)
    totals = {f"{b}_attempts": 0.0 for b in BUCKETS}
    made_totals = {f"{b}_made": 0.0 for b in BUCKETS}
    misses = np.zeros(simulations, dtype=float)
    for bucket in BUCKETS:
        attempts = rng.poisson(max(rates["attempts"][bucket], 0.0), simulations)
        made = rng.binomial(attempts, min(max(rates["makes"][bucket], 0.0), 1.0))
        points += made * contract.points(bucket)
        misses += attempts - made
        totals[f"{bucket}_attempts"] += float(attempts.mean())
        made_totals[f"{bucket}_made"] += float(made.mean())
    points += misses * contract.points("fg_missed_total")

    pat_att = rng.poisson(max(rates["pat_attempts"], 0.0), simulations)
    pat_made = rng.binomial(pat_att, min(max(rates["pat_make"], 0.0), 1.0))
    points += pat_made * contract.points("pat_made")
    points += (pat_att - pat_made) * contract.points("pat_missed")

    row["status"] = "projected"
    row["distribution"] = _percentiles(points)
    row["components"] = {
        **{k: round(v, 4) for k, v in totals.items()},
        **{k: round(v, 4) for k, v in made_totals.items()},
        "fg_missed_mean": round(float(misses.mean()), 4),
        "pat_attempts_mean": round(float(pat_att.mean()), 4),
        "pat_made_mean": round(float(pat_made.mean()), 4),
        "weeks_observed": rates["weeks_observed"],
    }
    return row


def build_artifact(history: pd.DataFrame, player_ids: Iterable[str],
                   contract: LeagueContract, *, season: int, week: int,
                   simulations: int = 10_000, seed: int = 6102026,
                   out_path: str | Path | None = None,
                   active: Mapping[str, bool] | None = None,
                   provenance: Sequence[Mapping[str, Any]] = (),
                   information_as_of: str | None = None) -> dict[str, Any]:
    """Immutable shadow artifact. Never merged into the offensive snapshot."""
    active = dict(active or {})
    players = [
        project(history, pid, contract, season=season, week=week,
                simulations=simulations, seed=seed,
                active=active.get(str(pid), True))
        for pid in player_ids
    ]
    artifact = {
        "schema_version": 1,
        "status": STATUS_SHADOW,
        "promoted": False,
        "disclaimer": ("Research only. Kicker projections are a shadow lane and "
                       "are not used for lineup, waiver or trade decisions."),
        "model_version": MODEL_VERSION,
        "season": int(season), "week": int(week),
        "simulations": int(simulations), "seed": int(seed),
        "scoring_hash": contract.scoring_hash,
        "roster_slot_hash": contract.roster_slot_hash,
        "model_run_at": datetime.now(UTC).isoformat(),
        "information_as_of": information_as_of,
        "provenance": [dict(p) for p in provenance],
        "n_projected": sum(1 for p in players if p["status"] == "projected"),
        "n_unavailable": sum(1 for p in players if p["status"] == "unavailable"),
        "players": players,
    }
    body = json.dumps(artifact, sort_keys=True, separators=(",", ":"), default=str)
    artifact["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(out_path)
    return artifact
