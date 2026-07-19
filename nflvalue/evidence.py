"""Phase C: turn a computed selection into legible, falsifiable evidence.

This is a RENDERING layer. It computes no new opinion, re-weights nothing, and
must never change a selection. Every number it emits is either already present
on the lean, already present in a registry or replay artifact, or an exact
algebraic rearrangement of those (a log share of a product that already equals
the projected mean). If you find yourself wanting to add a coefficient here,
that belongs in the accuracy ledger and a promotion gate, not in the explainer.

Three things this module refuses to do, because doing them is how an
explanation layer turns into a second, unaudited model:

1. Invent a strength score. Evidence strength is read from the factor
   registries (``n_raw``, ``n_effective``, ``posterior``, ``multiplicity_q``,
   ``season_forward``) written by the factor studies, and from the shrinkage
   already applied in ``features.py``. Where no registered evidence exists, the
   driver says ``NO_REGISTERED_EVIDENCE`` rather than receiving a number.
2. Fill a calibration bucket by interpolation. A bucket with too little
   measured history returns ``UNMEASURED_BUCKET`` and no strength claim.
3. Omit counter-evidence. ``counter_evidence`` renders even when empty, so a
   silent section is visibly a claim ("nothing found") rather than an absence.

The decomposition inverts the identity the projection already uses:

    mean = volume x efficiency x opp_factor x (optional adjustment multipliers)

Taking logs makes the contributions additive and comparable across units, which
is the same trick ``prop_learning.attribute`` already uses retrospectively to
split a graded miss into volume and efficiency error. This is that machinery run
forward instead of backward.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence

from .features import SHRINK_K
from .freshness import DEFAULT_STALENESS_HOURS, parse_ts, utcnow

SCHEMA_VERSION = 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")

# A calibration bucket needs at least this many graded selections before it may
# make a strength claim. Mirrors the pre-registered matched-control minimum in
# analysis/accuracy_protocol.json (matched_control.minimum_exposed_n), rather
# than introducing a second, softer threshold for the display layer.
MIN_BUCKET_N = 100

# features.py shrinks rolling efficiency toward the role's league mean with this
# constant; the weight on the player's own history is n / (n + SHRINK_K).
EFFICIENCY_SHRINK_K = SHRINK_K

# The opponent factor is clipped in features.py. At the bound the true effect is
# unknown and possibly larger, which is a caveat, not a stronger signal.
OPP_FACTOR_CLIP = (0.6, 1.6)

# Optional multipliers stamped onto a candidate by the adjustment passes. Each
# is absent unless that pass fired, so every consumer must default to 1.0.
ADJUSTMENT_MULTIPLIERS = (
    ("realloc_mult", "vacated opportunity", "teammate absence reallocation"),
    ("realloc_eff_mult", "reallocation efficiency penalty",
     "measured second-order efficiency loss on redistributed volume"),
    ("backup_qb_adj", "backup quarterback", "passing efficiency under a backup QB"),
    ("absence_qb_mult", "skill-position absence", "QB output when a leader sits"),
    ("bias_mult", "learned market bias", "walk-forward per-market bias correction"),
    ("reliability_mult", "learned reliability", "shrunk trailing hit-rate multiplier"),
)

# Driver -> registered factor id. CURATED, and deliberately sparse.
#
# The obvious implementation -- match on the registry's ``component_node`` --
# is wrong, and wrong in a way that flatters the output: component_node is the
# OUTCOME a factor moves (d_rec_yards), not the MECHANISM the driver expresses.
# Matching on it attached fam_D91_crossbook_disagreement (a market-disagreement
# study, n=0) to the opponent-strength driver, printing an authoritative factor
# id next to a driver it has nothing to do with. An explanation layer that does
# that is worse than one that says nothing.
#
# So: an entry appears below only where the registered study measures the same
# mechanism the driver describes. Everything else reads NO_REGISTERED_EVIDENCE.
# Keyed by (driver_id, position) where position matters, else (driver_id, None).
DRIVER_TO_REGISTRY_IDS = {
    # "opponent allows more/less to this position" -- the A family measures exactly this.
    ("opp_factor", "QB"): ("fam_A01_opp_qb_points_soft",),
    ("opp_factor", "RB"): ("fam_A02_opp_rb_points_soft",),
    ("opp_factor", "WR"): ("fam_A03_opp_wr_points_soft",),
    ("opp_factor", "TE"): ("fam_A04_opp_te_points_soft",),
    # game script is expressed here through the implied team total.
    ("game_script", None): ("fam_C05_implied_high_all",),
    # implied total -> team pass attempts is the only registered VOLUME mechanism.
    ("volume", "QB"): ("fam_D01_implied_high_passatt_team",),
}

# Drivers whose fallback strength is the player's own trailing sample. An
# opponent or game-script driver has no player-sample interpretation, so it must
# not borrow roll_games as if it did.
PLAYER_HISTORY_DRIVERS = ("volume", "efficiency")

MARKET_UNITS = {
    "receiving_yards": ("targets", "yards per target"),
    "receptions": ("targets", "catch rate"),
    "rushing_yards": ("carries", "yards per carry"),
    "rush_attempts": ("carries", "attempts"),
    "passing_yards": ("pass attempts", "yards per attempt"),
    "pass_attempts": ("pass attempts", "attempts"),
    "anytime_td": ("opportunities", "touchdown rate"),
}


class EvidenceUnavailable(RuntimeError):
    """Raised when a selection cannot be decomposed honestly."""


# --------------------------------------------------------------------------- #
# Small statistics, matching conventions already used in the audits
# --------------------------------------------------------------------------- #
def jeffreys_interval(hits: int, n: int, alpha: float = 0.05) -> Optional[List[float]]:
    """Jeffreys Beta(0.5, 0.5) credible interval for a rate.

    Same prior ``analysis/all_data_factor_audit.beta_difference`` uses for the
    exposed/control rates, so a band interval here and a factor interval there
    mean the same thing.
    """
    if n <= 0:
        return None
    try:
        from scipy.stats import beta
    except ImportError:  # pragma: no cover - scipy is a hard dependency in CI
        return None
    lo = float(beta.ppf(alpha / 2.0, hits + 0.5, n - hits + 0.5)) if hits > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2.0, hits + 0.5, n - hits + 0.5)) if hits < n else 1.0
    return [round(lo, 4), round(hi, 4)]


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# C.1 -- contribution decomposition
# --------------------------------------------------------------------------- #
def _direction(delta: float, side: str) -> str:
    """Does this driver push the projection toward or away from our side?

    A zero contribution reads ``not_ranked``, not ``neutral``: for the level-only
    drivers the contribution is unknown, and calling that neutral would assert
    something the decomposition cannot support.
    """
    if abs(delta) < 1e-9:
        return "not_ranked"
    pushes_up = delta > 0
    if side == "under":
        return "supports" if not pushes_up else "opposes"
    return "supports" if pushes_up else "opposes"


def _driver(driver_id: str, label: str, statement: str, *, value: float,
            log_contribution: float, side: str, units: str = "",
            baseline: Optional[float] = None, delta: Optional[float] = None,
            as_of: Optional[str] = None, provenance: Optional[Dict] = None,
            notes: Optional[List[str]] = None) -> Dict:
    return {
        "id": driver_id,
        "label": label,
        "statement": statement,
        "direction": _direction(log_contribution, side),
        "units": units,
        "value": round(float(value), 4),
        "baseline": None if baseline is None else round(float(baseline), 4),
        "delta": None if delta is None else round(float(delta), 4),
        "log_contribution": round(float(log_contribution), 5),
        "as_of": as_of,
        "provenance": provenance or {},
        "notes": notes or [],
    }


def decompose(lean: Dict, *, player_week_row: Optional[Dict] = None,
              as_of: Optional[str] = None) -> Dict:
    """Rank the drivers of one selection by their share of the projection.

    Ranking is by ``|log contribution|``: the projection is a product, so logs
    make a 15% volume bump and a 15% opponent factor directly comparable
    without inventing weights for them.
    """
    components = lean.get("proj_components") or lean.get("components") or {}
    if not components or "volume" not in components:
        raise EvidenceUnavailable(
            f"{lean.get('player_id')}/{lean.get('market')}: no projection components on the "
            "lean; decomposition would have to recompute the model, which this layer "
            "must not do")

    market = lean.get("market", "")
    side = lean.get("side") or ("over" if (lean.get("p_over") or 0) >= 0.5 else "under")
    volume_units, efficiency_units = MARKET_UNITS.get(market, ("opportunities", "efficiency"))
    drivers: List[Dict] = []
    warnings: List[str] = []

    if market == "anytime_td":
        # projection.py sums the rush and receive TD pathways into one
        # components entry, so they cannot be separated after the fact.
        warnings.append(
            "anytime_td collapses the rushing and receiving touchdown pathways into a "
            "single volume and rate term; this decomposition cannot separate them")

    volume = float(components["volume"])
    efficiency = float(components.get("efficiency") or 1.0)
    opp_factor = float(components.get("opp_factor") or 1.0)
    game_script = float(components.get("game_script") or 1.0)

    # ---- volume, expressed against the player's own trailing baseline -------
    baseline_volume = None
    if player_week_row:
        for column in ("roll_targets", "roll_carries", "roll_pass_attempts"):
            if volume_units.startswith(column.split("_")[1][:4]) or (
                    column == "roll_targets" and volume_units == "targets") or (
                    column == "roll_carries" and volume_units == "carries") or (
                    column == "roll_pass_attempts" and volume_units == "pass attempts"):
                candidate = player_week_row.get(column)
                if candidate is not None and not (isinstance(candidate, float)
                                                  and math.isnan(candidate)):
                    baseline_volume = float(candidate)
                break
    delta_volume = None if baseline_volume is None else volume - baseline_volume
    if baseline_volume:
        statement = (f"projected {volume:.2f} {volume_units} vs {baseline_volume:.2f} "
                     f"trailing baseline ({delta_volume:+.2f})")
        log_volume = math.log(volume / baseline_volume) if baseline_volume > 0 and volume > 0 else 0.0
    else:
        statement = (f"projected {volume:.2f} {volume_units} (no trailing baseline available, "
                     "so this is a level, not a change)")
        log_volume = 0.0
        warnings.append("no trailing volume baseline available; volume is reported as a "
                        "level, not a delta, and is not ranked against the other drivers")
    drivers.append(_driver(
        "volume", "projected volume", statement, value=volume, baseline=baseline_volume,
        delta=delta_volume, log_contribution=log_volume, side=side, units=volume_units,
        as_of=as_of,
        provenance={"source": "player_week", "columns": ["roll_team_pass_att/roll_team_rush_att",
                                                         "roll_target_share/roll_carry_share"],
                    "computed_in": "projection.expected_volume"}))

    # ---- opponent factor ----------------------------------------------------
    at_clip = opp_factor <= OPP_FACTOR_CLIP[0] + 1e-9 or opp_factor >= OPP_FACTOR_CLIP[1] - 1e-9
    notes = []
    if at_clip:
        notes.append(f"factor is at its clip bound {OPP_FACTOR_CLIP}; the true effect may be "
                     "larger than shown and is censored, not confirmed")
    drivers.append(_driver(
        "opp_factor", "opponent vs position",
        f"opponent allows {(opp_factor - 1.0) * 100:+.1f}% to this position vs league average",
        value=opp_factor, baseline=1.0, delta=opp_factor - 1.0,
        log_contribution=math.log(opp_factor) if opp_factor > 0 else 0.0,
        side=side, units="multiplier", as_of=as_of, notes=notes,
        provenance={"source": "opp_pos_def", "column": "roll_yp*_allowed_factor",
                    "definition": "rolling yards-per-play allowed to this role / prior-week league mean"}))

    # ---- game script --------------------------------------------------------
    implied = implied_team_total(lean)
    script_statement = (f"game script multiplier {game_script:.4f} "
                        f"({(game_script - 1.0) * 100:+.1f}% vs neutral)")
    if implied is not None:
        script_statement += f"; team implied total {implied:.1f}"
    drivers.append(_driver(
        "game_script", "game script", script_statement, value=game_script, baseline=1.0,
        delta=game_script - 1.0,
        log_contribution=math.log(game_script) if game_script > 0 else 0.0,
        side=side, units="multiplier", as_of=as_of,
        provenance={"source": "schedules", "columns": ["spread_line", "total_line"],
                    "computed_in": "projection.game_script_multipliers",
                    "note": "already folded into the volume term; reported, not re-applied"}))

    # ---- efficiency ---------------------------------------------------------
    if efficiency != 1.0:
        drivers.append(_driver(
            "efficiency", "efficiency",
            f"{efficiency:.3f} {efficiency_units} (level, not a change: the unshrunk value "
            "features.py used is not persisted, so no baseline delta can be shown)",
            value=efficiency, log_contribution=0.0, side=side, units=efficiency_units,
            as_of=as_of, notes=["not ranked against the other drivers -- no baseline exists "
                                "to express it as a contribution"],
            provenance={"source": "player_week", "column": "roll_ypt/roll_ypc/roll_ypa/roll_catch_rate",
                        "shrinkage": f"already shrunk toward the role league mean, k={EFFICIENCY_SHRINK_K}"}))

    # ---- optional adjustment multipliers ------------------------------------
    for key, label, description in ADJUSTMENT_MULTIPLIERS:
        raw = lean.get(key)
        if raw is None:
            continue
        multiplier = float(raw)
        if abs(multiplier - 1.0) < 1e-9:
            continue
        drivers.append(_driver(
            key, label, f"{description}: {(multiplier - 1.0) * 100:+.1f}%",
            value=multiplier, baseline=1.0, delta=multiplier - 1.0,
            log_contribution=math.log(multiplier) if multiplier > 0 else 0.0,
            side=side, units="multiplier", as_of=as_of,
            provenance={"source": "candidates adjustment pass", "column": key}))

    ranked = sorted(drivers, key=lambda d: abs(d["log_contribution"]), reverse=True)
    # Rank is part of the payload because the ORDERING is a model output: the
    # translator is allowed to print "1." only because the payload says 1.
    for position, driver in enumerate(ranked, start=1):
        driver["rank"] = position

    # ---- reconstruct, and say so if it does not tie out --------------------- #
    reconstructed = volume * efficiency * opp_factor
    for key, _label, _description in ADJUSTMENT_MULTIPLIERS:
        raw = lean.get(key)
        if raw is not None:
            reconstructed *= float(raw)
    reported = float(lean.get("mean") or 0.0)
    error = reconstructed - reported
    if reported and abs(error) > max(0.01, 0.001 * abs(reported)):
        warnings.append(
            f"decomposition does not reconstruct the reported mean "
            f"({reconstructed:.4f} vs {reported:.4f}); an unrecorded adjustment was applied "
            "and this breakdown is incomplete")

    return {
        "identity": "mean = volume x efficiency x opp_factor x adjustment multipliers",
        "reported_mean": round(reported, 4),
        "reconstructed_mean": round(reconstructed, 4),
        "reconstruction_error": round(error, 6),
        "drivers": ranked,
        "warnings": warnings,
    }


def implied_team_total(lean: Dict) -> Optional[float]:
    """Standard implied total from the posted total and spread, team-perspective."""
    total = lean.get("total_line")
    spread = lean.get("spread_line")
    if total is None or spread is None:
        return None
    team_spread = float(spread) if not lean.get("home") else -float(spread)
    return float(total) / 2.0 - team_spread / 2.0


# --------------------------------------------------------------------------- #
# C.2 -- evidence strength, read from the registries
# --------------------------------------------------------------------------- #
def load_registries() -> List[Dict]:
    entries: List[Dict] = []
    for name in ("factor_registry_families.json", "factor_registry_chemistry.json",
                 "factor_registry_roleshock.json"):
        payload = _load_json(os.path.join(DATA_DIR, name), [])
        if isinstance(payload, list):
            for entry in payload:
                entries.append({**entry, "_registry": name})
    return entries


def _protocol_minimums() -> Dict:
    protocol = _load_json(os.path.join(ROOT, "analysis", "accuracy_protocol.json"), {})
    matched = protocol.get("matched_control", {})
    return {"min_exposed": matched.get("minimum_exposed_n", 100),
            "min_control": matched.get("minimum_control_n", 100),
            "multiple_testing": matched.get("multiple_testing", "Benjamini-Hochberg q < 0.05")}


def strength_for_driver(driver: Dict, registries: Sequence[Dict],
                        player_week_row: Optional[Dict] = None,
                        position: Optional[str] = None) -> Dict:
    """Attach measured strength. Never scores; only reports what was measured."""
    minimums = _protocol_minimums()
    ids = (DRIVER_TO_REGISTRY_IDS.get((driver["id"], position))
           or DRIVER_TO_REGISTRY_IDS.get((driver["id"], None)) or ())
    matches = [e for e in registries if e.get("id") in ids]
    # A "proposed" factor with an empty cohort is a plan, not evidence.
    matches = [e for e in matches if ((e.get("n_raw") or {}).get("exposed") or 0) > 0]

    if not matches:
        if driver["id"] in PLAYER_HISTORY_DRIVERS:
            own = _own_sample_strength(driver, player_week_row)
            return {**own, "protocol_minimums": minimums}
        return {"status": "NO_REGISTERED_EVIDENCE", "strength_label": "unmeasured",
                "detail": ("no registered factor study measures this driver's mechanism; "
                           "no strength claim is available for it"),
                "protocol_minimums": minimums}

    # Strongest available: prefer an entry that survived its own gate.
    def rank(entry):
        status_rank = {"admitted": 0, "shadow": 1, "proposed": 2,
                       "research_only": 3, "rejected": 4}.get(entry.get("status"), 5)
        return (status_rank, -(entry.get("n_effective") or 0))

    best = sorted(matches, key=rank)[0]
    n_raw = best.get("n_raw")
    exposed = n_raw.get("exposed") if isinstance(n_raw, dict) else n_raw
    control = n_raw.get("control") if isinstance(n_raw, dict) else None
    season_forward = best.get("season_forward") or {}
    posterior = best.get("posterior") or {}
    effect = best.get("effect") or {}

    meets_n = (exposed or 0) >= minimums["min_exposed"] and (control or 0) >= minimums["min_control"]
    q = best.get("multiplicity_q")
    ci = posterior.get("ci95") or effect.get("ci95")
    excludes_zero = bool(ci) and (ci[0] > 0 or ci[1] < 0)
    forward_holds = bool(season_forward.get("magnitude_order_holds")) and \
        str(season_forward.get("sign_agreement_2023_25", "")).startswith("3/")

    return {
        "status": "REGISTERED",
        "registry_id": best.get("id"),
        "registry_file": best.get("_registry"),
        "registry_status": best.get("status"),
        "registry_status_reason": best.get("status_reason"),
        "n_raw": {"exposed": exposed, "control": control},
        "n_effective": best.get("n_effective"),
        "effect": {"point": _round(effect.get("point")), "ci95": _round_list(effect.get("ci95")),
                   "p": _round(effect.get("p"), 6)},
        "posterior": {"mean": _round(posterior.get("mean")),
                      "ci95": _round_list(posterior.get("ci95"))},
        "multiplicity_q": _round(q, 6),
        "season_forward": season_forward,
        "gates": {"meets_protocol_n": meets_n,
                  "posterior_ci_excludes_zero": excludes_zero,
                  "q_below_0_05": (q is not None and q < 0.05),
                  "season_forward_replicates": forward_holds},
        "strength_label": _strength_label(best, meets_n, excludes_zero, q, forward_holds),
        "protocol_minimums": minimums,
        "live_scoring": False,
    }


def _own_sample_strength(driver: Dict, player_week_row: Optional[Dict]) -> Dict:
    """Strength of the player's own trailing estimate, using features.py shrinkage."""
    games = None
    if player_week_row:
        raw = player_week_row.get("roll_games")
        if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
            games = float(raw)
    if games is None:
        return {"status": "NO_REGISTERED_EVIDENCE", "strength_label": "unmeasured",
                "detail": "no registered factor study covers this driver and the player's "
                          "trailing sample size is unavailable"}
    weight = games / (games + EFFICIENCY_SHRINK_K)
    return {
        "status": "NO_REGISTERED_EVIDENCE",
        "detail": "no registered factor study covers this driver; strength shown is the "
                  "player's own trailing sample after the shrinkage features.py already applied",
        "n_raw": {"player_games": int(games)},
        "n_effective": round(games * weight, 2),
        "shrinkage": {"k": EFFICIENCY_SHRINK_K, "weight_on_own_history": round(weight, 4),
                      "toward": "role league mean"},
        "strength_label": ("thin" if games < 4 else "moderate" if games < 8 else "adequate"),
    }


def _strength_label(entry: Dict, meets_n: bool, excludes_zero: bool,
                    q: Optional[float], forward_holds: bool) -> str:
    """Deterministic label from gates that already exist. No new arithmetic."""
    if entry.get("status") == "rejected":
        return "rejected-by-gate"
    passed = sum([meets_n, excludes_zero, bool(q is not None and q < 0.05), forward_holds])
    if passed == 4 and entry.get("status") in ("admitted", "shadow"):
        return "strong"
    if passed >= 3:
        return "suggestive"
    if passed >= 1:
        return "weak"
    return "unsupported"


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _round_list(values, digits: int = 4):
    return None if not values else [round(float(v), digits) for v in values]


# --------------------------------------------------------------------------- #
# C.5 -- measured calibration band
# --------------------------------------------------------------------------- #
def load_calibration_bands(path: Optional[str] = None) -> Dict:
    payload = _load_json(path or os.path.join(DATA_DIR, "lean_replay_2025.json"), {})
    leans = payload.get("leans") or {}
    return {"bands": leans.get("by_composite_band") or {},
            "overall": leans.get("overall") or {},
            "baseline": (payload.get("baseline_all_candidates") or {}).get("overall") or {},
            "framing": payload.get("framing"),
            "season": payload.get("season"),
            "source": os.path.basename(path or "lean_replay_2025.json")}


def _band_for_composite(composite: Optional[float], bands: Dict) -> Optional[str]:
    if composite is None:
        return None
    edges = [("<35", 0.0, 35.0), ("35-45", 35.0, 45.0),
             ("45-55", 45.0, 55.0), ("55+", 55.0, 1e9)]
    for name, low, high in edges:
        if low <= composite < high and name in bands:
            return name
    return None


def calibration_band(lean: Dict, calibration: Optional[Dict] = None) -> Dict:
    """Map the selection into a MEASURED band, or refuse.

    A bucket below ``MIN_BUCKET_N`` graded selections yields
    ``UNMEASURED_BUCKET`` and carries no rate, no interval, and no strength
    claim -- an unqualified 48% on n=33 reads as a finding when it is noise.
    """
    calibration = calibration or load_calibration_bands()
    bands = calibration.get("bands") or {}
    composite = lean.get("composite")
    name = _band_for_composite(composite, bands)
    if name is None:
        return {"status": "UNMEASURED_BUCKET", "composite": composite,
                "reason": "no measured band covers this composite score",
                "minimum_n": MIN_BUCKET_N, "source": calibration.get("source")}

    band = bands[name]
    n = int(band.get("n") or 0)
    rate = band.get("hit_rate")
    if n < MIN_BUCKET_N or rate is None:
        return {"status": "UNMEASURED_BUCKET", "composite": composite, "band": name,
                "n": n, "minimum_n": MIN_BUCKET_N,
                "reason": (f"band {name} has n={n} graded selections, below the "
                           f"{MIN_BUCKET_N} required before a rate is reported"),
                "source": calibration.get("source"), "framing": calibration.get("framing")}

    hits = int(round(float(rate) * n))
    overall = calibration.get("overall") or {}
    baseline = calibration.get("baseline") or {}
    return {
        "status": "MEASURED",
        "composite": composite,
        "band": name,
        "n": n,
        "hit_rate": round(float(rate), 4),
        "ci95": jeffreys_interval(hits, n),
        "pooled_hit_rate": overall.get("hit_rate"),
        "pooled_n": overall.get("n"),
        "all_candidates_baseline": baseline.get("hit_rate"),
        "source": calibration.get("source"),
        "season": calibration.get("season"),
        "framing": calibration.get("framing"),
    }


# --------------------------------------------------------------------------- #
# C.3 -- counter-evidence (always rendered)
# --------------------------------------------------------------------------- #
def load_regime_errors(path: Optional[str] = None) -> Dict:
    payload = _load_json(path or os.path.join(REPORTS_DIR, "fantasy_monte_carlo_history.json"), {})
    return payload.get("regimes") or {}


def counter_evidence(lean: Dict, decomposition: Dict, band: Dict, *,
                     strengths: Optional[Dict] = None,
                     feeds: Optional[Sequence] = None,
                     regimes: Optional[Dict] = None,
                     as_of: Optional[str] = None) -> List[Dict]:
    """Everything that argues against the selection. Rendered even when empty."""
    out: List[Dict] = []
    strengths = strengths or {}

    # 1. drivers pointing the other way
    for driver in decomposition["drivers"]:
        if driver["direction"] == "opposes" and abs(driver["log_contribution"]) > 1e-9:
            out.append({"kind": "contradicting_driver", "driver": driver["id"],
                        "detail": f"{driver['label']} argues against this side: {driver['statement']}",
                        "magnitude_log": driver["log_contribution"]})

    # 2. registry regimes where the factor did not replicate
    for driver_id, strength in strengths.items():
        if strength.get("status") != "REGISTERED":
            continue
        forward = strength.get("season_forward") or {}
        agreement = str(forward.get("sign_agreement_2023_25", ""))
        if agreement and not agreement.startswith("3/"):
            out.append({"kind": "failed_replication", "driver": driver_id,
                        "detail": (f"registered factor {strength.get('registry_id')} held its sign "
                                   f"in only {agreement} season-forward seasons"),
                        "registry_status": strength.get("registry_status")})
        if strength.get("registry_status") == "rejected":
            out.append({"kind": "rejected_factor", "driver": driver_id,
                        "detail": (f"the registered study for this driver "
                                   f"({strength.get('registry_id')}) was REJECTED by its own "
                                   f"promotion gate: {strength.get('registry_status_reason')}"),
                        "registry_status": "rejected"})

    # 3. the model's measured error in this player's regime
    regimes = regimes if regimes is not None else load_regime_errors()
    if regimes:
        out.append(_regime_counter_evidence(lean, regimes))

    # 4. the band's own shape
    if band.get("status") == "MEASURED":
        pooled = band.get("pooled_hit_rate")
        if pooled is not None and band["hit_rate"] < pooled:
            out.append({"kind": "band_underperforms_pool",
                        "detail": (f"selections in band {band['band']} hit {band['hit_rate']:.4f}, "
                                   f"below the pooled {pooled:.4f} across n={band.get('pooled_n')}")})
        ci = band.get("ci95")
        if ci and ci[0] <= 0.5238:
            out.append({"kind": "band_ci_spans_breakeven",
                        "detail": (f"the 95% interval [{ci[0]:.4f}, {ci[1]:.4f}] includes the "
                                   "0.5238 breakeven at a -110 price, so this band is not "
                                   "measurably profitable")})
    else:
        out.append({"kind": "unmeasured_bucket",
                    "detail": band.get("reason", "this selection falls in an unmeasured bucket, "
                                                 "so no historical strength claim is available")})

    # 5. data freshness
    out.extend(_freshness_counter_evidence(feeds, as_of))

    # 6. decomposition self-doubt
    for warning in decomposition.get("warnings", []):
        out.append({"kind": "decomposition_gap", "detail": warning})

    return out


def _regime_counter_evidence(lean: Dict, regimes: Dict) -> Dict:
    """Role-shock cohorts are the known weak spot; report them, don't average them away."""
    worst_name, worst = None, None
    for name, methods in regimes.items():
        metrics = (methods or {}).get("calibrated_monte_carlo") or next(iter((methods or {}).values()), None)
        if not metrics or metrics.get("coverage80") is None:
            continue
        deviation = abs(float(metrics["coverage80"]) - 0.80)
        if worst is None or deviation > worst[0]:
            worst = (deviation, metrics)
            worst_name = name
    if worst is None:
        return {"kind": "regime_error_rate", "detail": "no measured per-regime error rates available"}
    _deviation, metrics = worst
    return {
        "kind": "regime_error_rate",
        "regime": worst_name,
        "detail": (f"this selection's role regime is not determined on the props path, and the "
                   f"model's worst measured cohort is {worst_name}: n={metrics.get('n')}, "
                   f"MAE {metrics.get('mae'):.4f}, 80% interval coverage "
                   f"{metrics.get('coverage80'):.4f} against a nominal 0.8"),
        "n": metrics.get("n"),
        "mae": _round(metrics.get("mae")),
        "coverage80": _round(metrics.get("coverage80")),
        "nominal_coverage": 0.8,
    }


def _freshness_counter_evidence(feeds: Optional[Sequence], as_of: Optional[str]) -> List[Dict]:
    if not feeds:
        return [{"kind": "freshness_unknown",
                 "detail": "no feed freshness was supplied with this selection, so staleness "
                           "of injuries, inactives, rosters, and lines is unverified"}]
    now = parse_ts(as_of) if as_of else utcnow()
    out = []
    for feed in feeds:
        age = feed.age_hours(now)
        limit = DEFAULT_STALENESS_HOURS.get(feed.name)
        if age is None:
            out.append({"kind": "freshness_gap", "feed": feed.name,
                        "detail": f"feed {feed.name} has no timestamp"})
        elif limit is not None and age > limit:
            out.append({"kind": "freshness_gap", "feed": feed.name,
                        "age_hours": round(age, 2), "limit_hours": limit,
                        "detail": (f"feed {feed.name} is {age:.2f}h old against a "
                                   f"{limit}h staleness limit")})
    return out


# --------------------------------------------------------------------------- #
# C.4 -- falsifiers
# --------------------------------------------------------------------------- #
def falsifiers(lean: Dict, decomposition: Dict, band: Dict) -> List[Dict]:
    """Plainly: what observable event would void this call."""
    out: List[Dict] = []
    driver_ids = {d["id"] for d in decomposition["drivers"]}

    if "realloc_mult" in driver_ids:
        out.append({"driver": "realloc_mult", "voids": "the vacated-opportunity driver",
                    "statement": ("if the absent teammate is activated, the vacated-opportunity "
                                  "boost is void and the projection reverts toward baseline"),
                    "observable_by": "inactives report ~90 minutes before kickoff",
                    "gap": ("the reallocation pass does not record WHICH teammate was out, so "
                            "this falsifier cannot name the player")})
    if "absence_qb_mult" in driver_ids:
        out.append({"driver": "absence_qb_mult", "voids": "the skill-position absence driver",
                    "statement": ("if the absent skill-position leader is activated, this "
                                  "downgrade is void"),
                    "observable_by": "inactives report ~90 minutes before kickoff",
                    "gap": "the adjustment stores only the compounded multiplier, not the player"})
    if "backup_qb_adj" in driver_ids:
        out.append({"driver": "backup_qb_adj", "voids": "the backup-quarterback efficiency penalty",
                    "statement": "if the starting quarterback plays, the 0.92 efficiency penalty is void",
                    "observable_by": "inactives report / pregame warmups"})

    opp = next((d for d in decomposition["drivers"] if d["id"] == "opp_factor"), None)
    if opp and opp["notes"]:
        out.append({"driver": "opp_factor", "voids": "the magnitude of the opponent driver",
                    "statement": ("the opponent factor is censored at its clip bound, so the "
                                  "true matchup effect is unknown in size; a single opponent "
                                  "performance inside the bound would move it"),
                    "observable_by": "next opponent game result"})

    script = next((d for d in decomposition["drivers"] if d["id"] == "game_script"), None)
    if script and abs(script["delta"] or 0) > 0.005:
        out.append({"driver": "game_script", "voids": "the game-script tilt",
                    "statement": ("the script tilt is derived from the posted spread; if the "
                                  "spread moves through pick'em before kickoff the tilt "
                                  "reverses sign"),
                    "observable_by": "closing spread"})

    if band.get("status") == "MEASURED":
        out.append({"driver": "calibration_band", "voids": "the strength claim",
                    "statement": (f"if band {band['band']} finishes the season below the "
                                  f"0.5238 breakeven, the band claim is falsified"),
                    "observable_by": "end-of-season grading of this band"})
    else:
        out.append({"driver": "calibration_band", "voids": "nothing yet",
                    "statement": ("this bucket has no measured history to falsify; it becomes "
                                  f"testable once n reaches {band.get('minimum_n', MIN_BUCKET_N)}"),
                    "observable_by": "accumulating graded selections in this bucket"})
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_evidence(lean: Dict, *, player_week_row: Optional[Dict] = None,
                   registries: Optional[Sequence[Dict]] = None,
                   calibration: Optional[Dict] = None,
                   regimes: Optional[Dict] = None,
                   feeds: Optional[Sequence] = None,
                   as_of: Optional[str] = None) -> Dict:
    """The full computed payload. Deterministic; no LLM anywhere near it."""
    registries = list(registries) if registries is not None else load_registries()
    decomposition = decompose(lean, player_week_row=player_week_row, as_of=as_of)
    position = lean.get("pos") or (player_week_row or {}).get("role")
    strengths = {d["id"]: strength_for_driver(d, registries, player_week_row, position)
                 for d in decomposition["drivers"]}
    for driver in decomposition["drivers"]:
        driver["evidence"] = strengths[driver["id"]]
    band = calibration_band(lean, calibration)
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "player_id": lean.get("player_id"), "name": lean.get("name"),
            "team": lean.get("team"), "opponent": lean.get("defteam"),
            "market": lean.get("market"), "side": lean.get("side"),
            "line": lean.get("line"), "projected_mean": lean.get("mean"),
            "sd": lean.get("sd"), "model_probability": lean.get("p_over") if
            (lean.get("side") or "over") == "over" else lean.get("p_under"),
            "composite": lean.get("composite"), "season": lean.get("season"),
            "week": lean.get("week"), "game_id": lean.get("game_id"),
            "line_source": lean.get("line_source"), "as_of": as_of,
        },
        "decomposition": decomposition,
        "calibration_band": band,
        "counter_evidence": counter_evidence(lean, decomposition, band, strengths=strengths,
                                             feeds=feeds, regimes=regimes, as_of=as_of),
        "falsifiers": falsifiers(lean, decomposition, band),
        "conventions": {
            # Stated in the payload because the prose quotes them, and the
            # translator may only use numbers the payload contains.
            "credible_interval_pct": 95,
            "breakeven_probability_at_minus_110": 0.5238,
            "nominal_interval_coverage": 0.8,
            "shrinkage_k_efficiency": EFFICIENCY_SHRINK_K,
            "minimum_bucket_n": MIN_BUCKET_N,
        },
        "provenance": {
            "registries": sorted({e.get("_registry") for e in registries}) if registries else [],
            "calibration_source": band.get("source"),
            "live_scoring_from_registries": False,
            "note": ("this layer re-weights nothing; every value is read from the selection, a "
                     "registry, or a replay artifact, or is an exact log rearrangement of the "
                     "projection identity"),
        },
    }
