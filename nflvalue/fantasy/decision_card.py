"""``decision-card/1`` — the short private page this team is actually run from.

``my_team/1.0.0`` is a *contract*: it states everything that is true about one
fantasy team right now, in eight sections, and refuses loudly wherever it
cannot.  That is the right shape for an audit and the wrong shape for a
Sunday morning, because it answers a question nobody asked ("what is the state
of every lane?") in place of the one everybody does ("what do I change?").

This module is the reading layer over that contract.  It states four things
and stops:

  1. the current legal lineup;
  2. only the seats that differ from what is already set;
  3. at most three actionable decisions;
  4. freshness and unresolved-identity alerts.

Three rules give the card its shape, and each exists because the alternative
has already gone wrong somewhere in this repository:

*No number is computed here.*  Every delta, interval and probability is read
from ``my_team``, which reads it from the lineup engine, which reads it from
the simulation.  A second implementation of a quantity is how three modules
came to publish three different "scoring hashes" for one league.

*Nothing model-internal is shown.*  The reader is deciding whether to bench a
running back, not reviewing a stack.  Composite scores, learner names and the
unqualified word "confidence" are refused by :func:`validate`, which runs on
every card before it is returned — a rule that is only enforced by review is
not a rule.

*Cited context never moves a number.*  Team and injury news arrives as
:class:`context items <dict>` with a source and an as-of timestamp, and it can
appear only as a driver, a risk or an invalidation trigger.  Ranking, deltas
and ordering are computed before context is looked at, so a plausible-sounding
note cannot promote a decision.  ``tests/test_decision_card.py`` asserts the
card is byte-identical with and without context except for those fields.

Fail-closed
-----------
A stale, partial or unprovenanced run renders ONE ``no_current_pick`` reason
and no decisions.  It never falls back to the previous week: the card carries
its own scoring period, and :func:`write` overwrites both artifacts every run,
so there is no path by which last week's page survives as this week's answer.

Private by construction
-----------------------
The card names a private league, its rosters and its manager.  ``visibility``
is ``"private"`` on every card and
:mod:`nflvalue.fantasy.private_boundary` refuses to let one into any public
payload.  Nothing here writes to ESPN; the platform is read-only upstream and
this layer only reads what that read produced.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA_VERSION = "decision-card/1"
SOURCE_CONTRACT = "my_team/1.0.0"

#: Every card is private.  The value is asserted, not inferred, so a public
#: renderer that ever receives one can refuse it on the object itself.
VISIBILITY = "private"

#: The page is a decision aid, not a report.  Four is already the number of
#: things a person will read before acting; three decisions is the budget.
MAX_ACTIONABLE_DECISIONS = 3

DECISION_STATUSES = ("start", "sit", "add", "drop", "hold", "shadow", "no_current_pick")
#: Statuses that consume the budget above.  ``shadow`` and ``no_current_pick``
#: are shown but are not actions, so they never crowd out an action.
ACTIONABLE_STATUSES = frozenset({"start", "sit", "add", "drop", "hold"})
USABLE_FRESHNESS = frozenset({"fresh", "aging"})

#: Seats whose projections have not passed the 2026 promotion gate.  They are
#: rendered, visibly, as having no pick — never omitted, because an omitted
#: seat reads as a settled one.
SHADOW_POSITIONS = ("K", "D/ST")

ESPN_USE = ("Recommendation only. ESPN is read once, for display; this card never "
            "changes a lineup, places a claim or proposes a trade on the platform.")

MAX_DRIVERS = 2


class CardRejected(ValueError):
    """The card violated its own contract and was not returned.

    Raised rather than repaired.  A card that has been quietly corrected on the
    way out is a card whose rules are advisory, and the whole point of the
    jargon and budget rules is that they hold on the day someone is in a hurry.
    """


class ContextRejected(ValueError):
    """A cited context item could not be used.  Recorded, never rendered."""


# --------------------------------------------------------------------------- #
# Vocabulary the card may not use
# --------------------------------------------------------------------------- #
# These are not stylistic preferences.  Every term below names something the
# reader cannot check: a composite has no units, a learner name is not evidence,
# and "confidence" without a qualifier is read as a calibrated probability by
# everyone who has ever seen a weather forecast.  The model-relative frequency
# this card *does* publish has never been graded against outcomes, so it is
# allowed to say so and nothing else.
FORBIDDEN_PROSE_TERMS: tuple[str, ...] = (
    "composite", "composite score", "blended score", "model score", "ml score",
    "ml_score", "ensemble", "stack weight", "stack_weight", "stacking",
    "conformal", "z-score", "zscore", "standardised score", "standardized score",
    "feature importance", "shap", "random forest", "histgb", "hist gradient",
    "bayesian ridge", "gradient boosting", "xgboost", "lightgbm", "neural net",
    "hyperparameter", "regressor", "estimator", "residual", "loss function",
    "edge", "expected value", "ev", "alpha", "sharpe", "vorp",
)

#: The only phrase in which the word "confidence" may appear: the one that
#: says the number is not one.
ALLOWED_CONFIDENCE_PHRASE = "not a calibrated confidence"

#: Keys whose values are identifiers, digests or machine codes rather than
#: prose.  A git sha is not a sentence and must not be word-scanned.
OPAQUE_KEYS = frozenset({
    "schema_version", "source_contract", "generated_at", "captured_at", "as_of",
    "evaluated_at", "model_version", "scoring_hash", "roster_slot_hash",
    "snapshot_hash", "player_id", "espn_player_id", "pro_team_id",
    "fantasy_team_id", "league_id", "code", "kind", "status", "state",
    "freshness_state", "visibility",
})

_WORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (term, re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE))
    for term in FORBIDDEN_PROSE_TERMS
)
_CONFIDENCE_RE = re.compile(r"(?<![a-z])confidence(?![a-z])", re.IGNORECASE)
_NUMERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def prose_violations(text: str) -> list[str]:
    """Forbidden vocabulary in one string, in declaration order."""
    found = [term for term, pattern in _WORD_PATTERNS if pattern.search(text)]
    stripped = text.lower().replace(ALLOWED_CONFIDENCE_PHRASE, " ")
    if _CONFIDENCE_RE.search(stripped):
        found.append("confidence (unqualified)")
    return found


def _walk_prose(node: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in OPAQUE_KEYS:
                continue
            yield from _walk_prose(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk_prose(value, f"{path}[{index}]")
    elif isinstance(node, str):
        yield path, node


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _round(value: Any, digits: int = 2) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def _reason(code: str, text: str) -> dict:
    """A machine code plus the sentence a person reads.

    Upstream reasons are deliberately not forwarded.  ``my_team`` explains
    itself in module paths ("nflvalue.fantasy.waivers.plan() produces one"),
    which is the correct register for a contract and the wrong one for a page,
    and forwarding it is how model-internal vocabulary would arrive on the card
    through a door the validator does not watch.
    """
    return {"code": str(code), "text": str(text)}


def _identity(entry: Mapping[str, Any], *, slot: str | None, team_id: Any) -> dict:
    """Stable identifiers for one player, in one place.

    ``espn_player_id`` is the platform's stable key and ``player_id`` the
    model's; both travel, because the two sides of every decision on this page
    are joined on one and graded on the other.
    """
    return {
        "player_id": entry.get("player_id"),
        "espn_player_id": entry.get("espn_player_id"),
        "name": entry.get("name"),
        "position": entry.get("position"),
        "pro_team_id": entry.get("pro_team_id"),
        "fantasy_team_id": team_id,
        "slot": slot,
    }


# --------------------------------------------------------------------------- #
# Provenance — required, never partially claimed
# --------------------------------------------------------------------------- #
def provenance(my_team: Mapping[str, Any], *, model_version: str | None) -> dict:
    """Model version, scoring hash, snapshot hash and freshness, or the gap.

    ``complete`` is false the moment any one of them is missing.  A decision
    that cannot say which model produced it, under which scoring rules, from
    which capture, is not a decision anyone can check next week — and an
    uncheckable recommendation is exactly the artifact this project exists to
    not produce.
    """
    fresh = dict(my_team.get("freshness") or {})
    source = (my_team.get("sources") or [{}])[0]
    snapshot_hash = source.get("roster_hash")
    fields = {
        "model_version": (str(model_version) if model_version else None),
        "scoring_hash": my_team.get("scoring_hash"),
        "roster_slot_hash": my_team.get("roster_slot_hash"),
        "snapshot_hash": snapshot_hash,
        "freshness_state": fresh.get("state"),
    }
    missing = sorted(name for name, value in fields.items() if not value)
    return {
        **fields,
        "source_contract": my_team.get("schema_version"),
        "snapshot_retrieved_at": source.get("retrieved_at"),
        "complete": not missing,
        "missing": missing,
    }


def _decision_provenance(prov: Mapping[str, Any]) -> dict:
    """The four stamps every decision carries, copied so a row travels alone."""
    return {
        "model_version": prov.get("model_version"),
        "scoring_hash": prov.get("scoring_hash"),
        "snapshot_hash": prov.get("snapshot_hash"),
        "freshness_state": prov.get("freshness_state"),
    }


# --------------------------------------------------------------------------- #
# Cited context — drivers, risks, invalidation.  Never a number.
# --------------------------------------------------------------------------- #
def _clean_context(items: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split supplied context into usable items and stated refusals.

    An item without a source or an as-of timestamp is not context, it is a
    rumour with good posture, and it is dropped with its reason recorded so the
    page can say how many notes it declined rather than silently thinning them.
    """
    usable: list[dict] = []
    refused: list[dict] = []
    # A refusal records its ordinal and its reason and NOT the note itself.
    # Echoing the rejected text is how banned vocabulary would reach the page
    # through the one door the validator cannot close, wearing an apology.
    for position, raw in enumerate(items or ()):
        if not isinstance(raw, Mapping):
            refused.append({"item": position, "reason": "context item is not structured"})
            continue
        text = str(raw.get("text") or "").strip()
        source = str(raw.get("source") or "").strip()
        as_of = str(raw.get("as_of") or "").strip()
        if not text:
            refused.append({"item": position, "reason": "context item carries no text"})
            continue
        if not source or not as_of:
            refused.append({"item": position,
                            "reason": "context item carries no source and as-of timestamp"})
            continue
        offending = prose_violations(text) + prose_violations(source)
        if offending:
            refused.append({"item": position,
                            "reason": "context item is written in wording this page does not show"})
            continue
        usable.append({
            "text": text,
            "source": source,
            "as_of": as_of,
            "counter_evidence": bool(raw.get("counter_evidence")),
            "player_ids": {str(v) for v in (raw.get("player_ids") or []) if v is not None},
            "espn_player_ids": {int(v) for v in (raw.get("espn_player_ids") or [])
                                if isinstance(v, int)},
            "pro_team_ids": {int(v) for v in (raw.get("pro_team_ids") or [])
                             if isinstance(v, int)},
        })
    return usable, refused


def _matches(item: Mapping[str, Any], *subjects: Mapping[str, Any]) -> bool:
    for subject in subjects:
        if not subject:
            continue
        if str(subject.get("player_id")) in item["player_ids"]:
            return True
        if subject.get("espn_player_id") in item["espn_player_ids"]:
            return True
        if subject.get("pro_team_id") is not None and \
                subject["pro_team_id"] in item["pro_team_ids"]:
            return True
    return False


def _cited(item: Mapping[str, Any]) -> dict:
    return {"text": item["text"], "source": item["source"], "as_of": item["as_of"]}


def _drivers(context: Sequence[Mapping[str, Any]], *subjects: Mapping[str, Any]) -> list[dict]:
    """Up to two cited notes about these players.  Deterministically chosen.

    Sorted newest-first by the item's own as-of string, then by text, so the
    same inputs always produce the same two.  A driver is context for a
    decision the numbers already made; it is never the reason the decision
    exists.
    """
    matched = [item for item in context
               if not item["counter_evidence"] and _matches(item, *subjects)]
    matched.sort(key=lambda item: (item["as_of"], item["text"]), reverse=True)
    return [_cited(item) for item in matched[:MAX_DRIVERS]]


# --------------------------------------------------------------------------- #
# Lineup
# --------------------------------------------------------------------------- #
def _current_slots(my_team: Mapping[str, Any]) -> dict[Any, str]:
    return {entry.get("espn_player_id"): str(entry.get("lineup_slot") or "BE")
            for entry in (my_team.get("roster") or [])}


def _current_lineup(my_team: Mapping[str, Any], *, team_id: Any) -> dict:
    """The legal lineup to run, with each seat marked already-set or not."""
    section = my_team.get("optimal_lineup") or {}
    shadow = list(section.get("shadow_slots") or [])
    if section.get("status") != "ok":
        violations = [str(v) for v in (section.get("violations") or [])]
        draft_state = str((my_team.get("draft") or {}).get("state") or "")
        if not (my_team.get("roster") or []):
            reason = _reason(
                "no_roster",
                "This team holds no players that could be tied to a projection, so there is no "
                "lineup to set."
                + (" The draft has not happened yet." if draft_state == "pre_draft" else ""))
        else:
            reason = _reason(
                "lineup_undecidable",
                "No legal lineup can be stated from this snapshot."
                + (f" Unfilled or illegal: {'; '.join(violations)}." if violations else ""))
        return {
            "status": "no_current_pick",
            "reason": reason,
            "legal": False,
            "violations": violations,
            "slots": [],
            "projected_points_total": None,
            "shadow_slots": shadow,
        }
    set_now = _current_slots(my_team)
    slots = []
    for entry in section.get("starters") or []:
        espn_id = entry.get("espn_player_id")
        slots.append({
            **_identity(entry, slot=entry.get("slot"), team_id=team_id),
            "projected_points": _round(entry.get("projected_mean")),
            "already_set": set_now.get(espn_id) == entry.get("slot"),
        })
    return {
        "status": "ok",
        "reason": None,
        "legal": True,
        "violations": [],
        "slots": slots,
        "projected_points_total": _round(section.get("projected_total")),
        "shadow_slots": shadow,
    }


def _interval(measured: Mapping[str, Any] | None) -> dict:
    """The move's own distribution, or a stated absence.

    *measured* is already normalised to ``p10``/``p90``/``simulations``/
    ``basis``; it is None when the producer did not measure the move. Every
    field is required: a partly-filled interval is worse than a missing one,
    because ``status: ok`` beside two empty numbers is a claim the page then
    renders as an em dash and the validator waves through.
    """
    required = ("p10", "p90", "simulations")
    if not measured or any(measured.get(key) is None for key in required):
        return {
            "status": "unavailable",
            "reason": _reason("no_joint_simulation",
                              "This move was not measured over paired simulated weeks, so it has "
                              "no range."),
            "basis": None, "simulations": None, "p10": None, "p90": None,
        }
    return {
        "status": "ok",
        "reason": None,
        "basis": measured.get("basis") or "paired joint simulation rows",
        "simulations": int(measured["simulations"]),
        "p10": _round(measured["p10"]),
        "p90": _round(measured["p90"]),
    }


def _swap_measurement(uncertainty: Mapping[str, Any]) -> dict | None:
    """`my_team`'s start/sit uncertainty block, in this module's field names.

    The rows are paired: the same simulated week supplies both players, so the
    difference is read off one row rather than assembled from two players'
    separate percentiles, which is the worst case under a dependence the
    simulation never draws.
    """
    if (uncertainty or {}).get("status") != "ok":
        return None
    return {
        "p10": uncertainty.get("p10_delta"),
        "p90": uncertainty.get("p90_delta"),
        "simulations": uncertainty.get("simulations"),
        "basis": uncertainty.get("basis"),
        "probability": uncertainty.get("model_relative_prob_start_scores_more"),
    }


def _probability(measured: Mapping[str, Any] | None) -> dict | None:
    if not measured:
        return None
    value = measured.get("probability")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return {
        "value": round(float(value), 4),
        "label": "share of simulated weeks in which this choice scores more",
        "qualifier": (f"model-relative over the model's own draws — {ALLOWED_CONFIDENCE_PHRASE} "
                      "and not graded against what happened"),
        "basis": "paired joint simulation rows",
        "simulations": measured.get("simulations"),
    }


def _lineup_change_rows(my_team: Mapping[str, Any], *, team_id: Any) -> list[dict]:
    """Seats that differ from what is set, read from the contract's own pairing.

    ``my_team.start_sit`` already pairs the player the legal optimum seats with
    the player it displaces, at the same slot, from the same rows.  Re-deriving
    the pairing here would be a second implementation of the one quantity the
    page is about.
    """
    section = my_team.get("start_sit") or {}
    if section.get("status") != "ok":
        return []
    # A player the lineup engine excluded is not a choice that was made, he is
    # a seat that emptied.  The distinction decides whether this row is a
    # judgement (compare the two) or a consequence (say who cannot play), and
    # the difference is visible on the page.
    unavailable = {entry.get("espn_player_id"): str(entry.get("reason") or "cannot play")
                   for entry in ((my_team.get("optimal_lineup") or {}).get("excluded") or [])}
    rows = []
    for decision in section.get("decisions") or []:
        start = decision.get("start") or {}
        sit = decision.get("sit")
        uncertainty = decision.get("uncertainty") or {}
        blocker = unavailable.get((sit or {}).get("espn_player_id")) if sit else None
        rows.append({
            "slot": decision.get("slot"),
            "start": _identity(start, slot=decision.get("slot"), team_id=team_id),
            "sit": (_identity(sit, slot=decision.get("slot"), team_id=team_id)
                    if sit else None),
            "mean_delta": _round(decision.get("projected_delta")),
            "median_delta": _round(uncertainty.get("median_delta")),
            "uncertainty": uncertainty,
            "forced": sit is None or blocker is not None,
            "blocker": blocker,
        })
    rows.sort(key=lambda row: (not row["forced"], -(row["mean_delta"] or 0.0), str(row["slot"]),
                               row["start"]["espn_player_id"] or 0))
    return rows


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
def _risk_from_interval(interval: Mapping[str, Any], probability: Mapping[str, Any] | None,
                        *, subject: str, alternative: str) -> dict:
    """One visible counter-case, taken from the same rows as the recommendation.

    Every decision carries one.  A recommendation with no stated way of being
    wrong reads as a fact, and the p10 of a swap this page is confident about
    is routinely negative — that is the honest headline, not a footnote.
    """
    low = interval.get("p10")
    if isinstance(low, (int, float)) and low < 0:
        share = ""
        if probability and isinstance(probability.get("value"), float):
            share = f" {round((1.0 - probability['value']) * 100)}% of simulated weeks favour "\
                    f"{alternative}."
        return {
            "text": (f"The worst tenth of simulated weeks has {subject} scoring {abs(low)} points "
                     f"below {alternative}.{share}"),
            "source": "paired joint simulation rows",
            "as_of": None,
        }
    return {
        "text": (f"This is a projection gap, not a guarantee: {alternative} outscoring "
                 f"{subject} is an ordinary week, not an upset."),
        "source": "paired joint simulation rows",
        "as_of": None,
    }


def _counter_evidence(context: Sequence[Mapping[str, Any]],
                      *subjects: Mapping[str, Any]) -> dict | None:
    matched = [item for item in context if item["counter_evidence"] and _matches(item, *subjects)]
    matched.sort(key=lambda item: (item["as_of"], item["text"]), reverse=True)
    return _cited(matched[0]) if matched else None


def _start_decision(row: Mapping[str, Any], *, context: Sequence[Mapping[str, Any]],
                    prov: Mapping[str, Any]) -> dict:
    """One seat that differs from what is set, as a judgement or a consequence.

    A *forced* row is a seat whose occupant cannot play — a bye, an out
    designation, an empty slot.  It has no comparison to measure, so it carries
    no delta and no probability, and it says who cannot play instead of
    pretending a distribution was consulted.  Reporting it as a swap worth
    ``-2.2`` points, which is what a straight difference of two projections
    produces when the player being replaced is on bye, is arithmetic about a
    week that will not happen.
    """
    start, sit = row["start"], row["sit"]
    subject_name = str(start.get("name") or "this player")
    alt_name = str((sit or {}).get("name") or "the empty seat")
    base = {
        "kind": "lineup",
        "slot": row["slot"],
        "subject": start,
        "alternative": sit,
        "forced": bool(row["forced"]),
        "provenance": _decision_provenance(prov),
    }

    if row["forced"]:
        blocker = row.get("blocker")
        if sit is None:
            headline = f"Start {subject_name} at {row['slot']} — the seat is empty"
            why = (f"Nobody is set at {row['slot']}. {subject_name} is the best remaining player "
                   "eligible for the seat.")
            wrong_if = (f"{subject_name} is downgraded to out or inactive, or another player "
                        f"eligible for {row['slot']} becomes available.")
        else:
            headline = f"Start {subject_name} at {row['slot']} — {alt_name} cannot play"
            why = (f"{alt_name} is out of this lineup ({blocker or 'cannot play'}), so "
                   f"{row['slot']} has to be filled. {subject_name} is the best remaining player "
                   "eligible for the seat.")
            wrong_if = (f"{alt_name} is cleared to play before kickoff, or {subject_name} is "
                        "downgraded to out or inactive.")
        return {
            **base,
            "status": "start",
            "headline": headline,
            "reason": _reason("forced_change", why),
            "mean_delta": None,
            "median_delta": None,
            "interval": {
                "status": "not_applicable",
                "reason": _reason("forced_change",
                                  "There is no comparison to measure: the player being replaced "
                                  "cannot play."),
                "basis": None, "simulations": None, "p10": None, "p90": None,
            },
            "model_relative_probability": None,
            "drivers": _drivers(context, start, sit),
            "risk": _counter_evidence(context, start, sit) or {
                "text": (f"{subject_name} is the best option left, which is not the same as a good "
                         f"one; this seat is weaker this week than it looks."),
                "source": "current legal lineup",
                "as_of": None,
            },
            "invalidation_trigger": wrong_if,
        }

    measured = _swap_measurement(row["uncertainty"])
    interval = _interval(measured)
    probability = _probability(measured)
    base |= {
        "mean_delta": row["mean_delta"],
        "median_delta": row["median_delta"],
        "interval": interval,
        "model_relative_probability": probability,
    }
    if not isinstance(row["mean_delta"], float) or row["mean_delta"] <= 0:
        return {
            **base,
            "status": "no_current_pick",
            "headline": f"No pick at {row['slot']}",
            "reason": _reason("no_measured_gain",
                              f"Seating {subject_name} at {row['slot']} in place of {alt_name} "
                              "does not project to gain anything, so there is nothing to "
                              "recommend."),
            "drivers": [], "risk": None, "invalidation_trigger": None,
        }
    if interval["status"] != "ok":
        return {
            **base,
            "status": "no_current_pick",
            "headline": f"No pick at {row['slot']}",
            "reason": _reason("no_joint_simulation",
                              f"{subject_name} projects {row['mean_delta']} points above "
                              f"{alt_name}, but the two were not drawn together, so the range of "
                              "that gap is unmeasured and the swap is not recommended on the "
                              "mean alone."),
            "drivers": [], "risk": None, "invalidation_trigger": None,
        }
    risk = _counter_evidence(context, start, sit) or _risk_from_interval(
        interval, probability, subject=subject_name, alternative=alt_name)
    return {
        **base,
        "status": "start",
        "headline": f"Start {subject_name} at {row['slot']}, sit {alt_name}",
        "reason": _reason("lineup_change",
                          f"{subject_name} projects {row['mean_delta']} points above {alt_name} "
                          f"at {row['slot']}."),
        "drivers": _drivers(context, start, sit),
        "risk": risk,
        "invalidation_trigger": (
            f"{subject_name} or {alt_name} is downgraded to out, inactive or on bye, or a "
            f"projection revision closes the {row['mean_delta']}-point gap before kickoff."),
    }


def _waiver_rows(my_team: Mapping[str, Any], *, team_id: Any,
                 context: Sequence[Mapping[str, Any]], prov: Mapping[str, Any]) -> list[dict]:
    """Waiver adds, and only once the planner's own gate has passed.

    ``my_team.waivers`` is ``ok`` only when the planner produced a legal add,
    an identified legal drop and a computed joint delta that cleared the gate.
    Anything short of that arrives here as a refusal, and a refusal is not
    rendered as an empty table: the section is simply absent from the card and
    its reason is recorded under ``withheld``.
    """
    section = my_team.get("waivers") or {}
    if section.get("status") != "ok":
        return []
    rows = []
    for target in section.get("targets") or []:
        add = target.get("add") or {}
        drop = target.get("drop") or {}
        delta = target.get("lineup_delta") or {}
        mean_delta = _round(delta.get("own_optimal_lineup_delta"))
        # The planner publishes `median`, `p10`, `p90`, `simulations` and
        # `model_relative_prob_improves` on the same block. Reading them under
        # any other name silently produces an interval that claims to be `ok`
        # with nothing in it.
        measured = ({"p10": delta.get("p10"), "p90": delta.get("p90"),
                     "simulations": delta.get("simulations"), "basis": delta.get("basis"),
                     "probability": delta.get("model_relative_prob_improves")}
                    if target.get("lineup_delta_status") == "ok" else None)
        add_id = _identity(
            {"player_id": None, "espn_player_id": add.get("espn_player_id"),
             "name": add.get("name"), "position": add.get("position")},
            slot=None, team_id=team_id)
        drop_id = _identity(
            {"player_id": None, "espn_player_id": drop.get("espn_player_id"),
             "name": drop.get("name"), "position": None},
            slot=None, team_id=team_id) if drop.get("espn_player_id") is not None else None
        add_name = str(add.get("name") or "this player")
        drop_name = str((drop or {}).get("name") or "an unnamed drop")
        interval = _interval(measured)
        risk = _counter_evidence(context, add_id, drop_id) or {
            "text": (f"Adding {add_name} costs {drop_name} permanently; if the role behind the "
                     "add does not hold, the roster is worse by whatever the dropped player "
                     "would have been."),
            "source": "waiver planner legality pass",
            "as_of": None,
        }
        base = {
            "kind": "waiver",
            "forced": False,
            "slot": add.get("position"),
            "subject": add_id,
            "alternative": drop_id,
            "mean_delta": mean_delta,
            "median_delta": _round(delta.get("median")),
            "interval": interval,
            "model_relative_probability": _probability(measured),
            "provenance": _decision_provenance(prov),
        }
        if interval["status"] != "ok":
            rows.append({
                **base,
                "status": "no_current_pick",
                "headline": f"No claim for {add_name}",
                "reason": _reason("no_joint_simulation",
                                  f"Adding {add_name} was not measured over paired simulated "
                                  "weeks, so the size of the gain is unknown and the claim is "
                                  "not recommended."),
                "drivers": [], "risk": None, "invalidation_trigger": None,
            })
            continue
        rows.append({
            **base,
            "status": "add",
            "headline": f"Claim {add_name}, drop {drop_name}",
            "reason": _reason("waiver_add",
                              f"{add_name} improves the legal lineup by {mean_delta} points once "
                              f"{drop_name} is dropped."),
            "drivers": _drivers(context, add_id, drop_id),
            "risk": risk,
            "invalidation_trigger": (f"The claim is processed, {add_name} is rostered by another "
                                     "team, or the transaction window closes."),
        })
    rows.sort(key=lambda row: (-(row["mean_delta"] or 0.0),
                               row["subject"]["espn_player_id"] or 0))
    return rows


def _hold_decision(lineup: Mapping[str, Any], *, context: Sequence[Mapping[str, Any]],
                   prov: Mapping[str, Any]) -> dict:
    seats = list(lineup.get("slots") or [])
    margin = None
    points = sorted((s["projected_points"] for s in seats
                     if isinstance(s.get("projected_points"), float)), reverse=True)
    if len(points) >= 2:
        margin = round(points[-2] - points[-1], 2)
    subjects = [{"player_id": s.get("player_id"), "espn_player_id": s.get("espn_player_id"),
                 "pro_team_id": s.get("pro_team_id")} for s in seats]
    return {
        "kind": "lineup",
        "status": "hold",
        "forced": False,
        "slot": None,
        "subject": None,
        "alternative": None,
        "headline": "Hold — the lineup already set is the legal optimum",
        "reason": _reason("no_change",
                          "Every seat the model would fill is already filled by the player it "
                          "would choose, so there is nothing to change."),
        "mean_delta": 0.0,
        "median_delta": 0.0,
        "interval": {"status": "not_applicable",
                     "reason": _reason("no_swap", "There is no swap to measure."),
                     "basis": None, "simulations": None, "p10": None, "p90": None},
        "model_relative_probability": None,
        "drivers": _drivers(context, *subjects),
        "risk": {
            "text": ("Holding is only right while these projections hold; a late inactive can "
                     "make a change correct after this card was written."),
            "source": "current legal lineup",
            "as_of": None,
        },
        "invalidation_trigger": (
            "Any starter is downgraded to out or inactive"
            + (f", or a projection revision larger than {margin} points reorders two seats."
               if margin is not None else ".")),
        "provenance": _decision_provenance(prov),
    }


def _shadow_seats(my_team: Mapping[str, Any]) -> list[dict]:
    """K and D/ST, stated as unpromoted rather than omitted."""
    seats = []
    for position, key in (("K", "kicker_shadow"), ("D/ST", "dst_shadow")):
        section = my_team.get(key) or {}
        seats.append({
            "position": position,
            "status": "shadow",
            "promoted": bool(section.get("promoted")),
            "seats": section.get("seats"),
            "headline": f"{position} — no pick",
            "reason": _reason(
                "shadow_not_promoted",
                f"{position} projections have not passed their own season-forward check, so this "
                "seat has no recommendation. The offensive lineup above does not depend on it."),
            "invalidation_trigger": (f"A passing season-forward check promotes {position} out of "
                                     "shadow."),
        })
    return seats


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
def _alerts(my_team: Mapping[str, Any], *, refused_context: Sequence[Mapping[str, Any]],
            withheld: Sequence[Mapping[str, Any]]) -> list[dict]:
    alerts: list[dict] = []
    fresh = my_team.get("freshness") or {}
    state = str(fresh.get("state") or "missing")
    if state != "fresh":
        age = fresh.get("age_hours")
        alerts.append({
            "kind": "freshness",
            "severity": "blocking" if state not in USABLE_FRESHNESS else "warning",
            "text": (f"The league was last read {age} hours ago." if age is not None
                     else "The league capture carries no readable timestamp."),
            "state": state,
            "captured_at": fresh.get("captured_at"),
        })
    unresolved = my_team.get("unresolved_identities") or {}
    count = int(unresolved.get("count") or 0)
    if count:
        names = [str(entry.get("name") or entry.get("espn_player_id"))
                 for entry in (unresolved.get("entries") or [])]
        alerts.append({
            "kind": "identity",
            "severity": "warning",
            "text": (f"{count} player(s) on this roster could not be tied to a projection and "
                     "were left out of the lineup rather than counted as zero."),
            "players": names,
        })
    lineup = my_team.get("optimal_lineup") or {}
    if lineup.get("status") != "ok" and lineup.get("violations"):
        alerts.append({
            "kind": "legality",
            "severity": "blocking",
            "text": "The roster cannot field a legal lineup.",
            "violations": [str(v) for v in lineup["violations"]],
        })
    if refused_context:
        alerts.append({
            "kind": "context",
            "severity": "warning",
            "text": (f"{len(refused_context)} supplied note(s) were not used because they carried "
                     "no source and timestamp, or used wording this page does not show."),
            "refused": [dict(item) for item in refused_context],
        })
    if withheld:
        alerts.append({
            "kind": "budget",
            "severity": "info",
            "text": (f"{len(withheld)} further action(s) were held back to keep this page to "
                     f"{MAX_ACTIONABLE_DECISIONS}."),
            "held_back": [dict(item) for item in withheld],
        })
    return alerts


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _blocked_card(my_team: Mapping[str, Any], *, now: str, prov: Mapping[str, Any],
                  reason: Mapping[str, Any], refused_context: Sequence[Mapping[str, Any]],
                  league: Mapping[str, Any]) -> dict:
    """One reason, no decisions, no reuse of anything earlier."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_contract": SOURCE_CONTRACT,
        "visibility": VISIBILITY,
        "generated_at": now,
        "state": "no_current_pick",
        "reason": dict(reason),
        "league": dict(league),
        "provenance": dict(prov),
        "current_lineup": {
            "status": "no_current_pick",
            "reason": dict(reason),
            "legal": False, "violations": [], "slots": [],
            "projected_points_total": None,
            "shadow_slots": list((my_team.get("optimal_lineup") or {}).get("shadow_slots") or []),
        },
        "lineup_changes": [],
        "decisions": [],
        "withheld": [],
        "shadow_seats": _shadow_seats(my_team),
        "alerts": _alerts(my_team, refused_context=refused_context, withheld=()),
        "espn_use": ESPN_USE,
        "prose_rewrite": {"applied": False,
                          "reason": "the default path composes every sentence in code"},
    }


def build(my_team: Mapping[str, Any], *, now: str, model_version: str | None = None,
          context: Sequence[Mapping[str, Any]] = ()) -> dict:
    """Assemble a ``decision-card/1`` from one ``my_team`` contract.

    *context* is cited team or injury news.  It may add a driver, a risk or an
    invalidation trigger to a decision the numbers already made, and it can do
    nothing else — no item is consulted until ranking is finished.
    """
    if not isinstance(my_team, Mapping):
        raise CardRejected(f"contract is not a mapping ({type(my_team).__name__})")
    version = my_team.get("schema_version")
    if version != SOURCE_CONTRACT:
        raise CardRejected(
            f"unsupported contract {version!r}; this card reads {SOURCE_CONTRACT}. Rendering a "
            "card from a shape this reader does not know would produce a confident page about a "
            "team nobody checked.")

    raw_league = my_team.get("league") or {}
    league = {
        "league_id": raw_league.get("league_id"),
        "league_name": raw_league.get("league_name"),
        "season": raw_league.get("season"),
        "scoring_period": raw_league.get("scoring_period"),
        "team_id": raw_league.get("team_id"),
        "team_name": raw_league.get("team_name"),
    }
    team_id = raw_league.get("team_id")
    prov = provenance(my_team, model_version=model_version)
    usable_context, refused_context = _clean_context(context)

    fresh_state = str((my_team.get("freshness") or {}).get("state") or "missing")
    if fresh_state not in USABLE_FRESHNESS:
        described = {
            "stale": "too old to describe the team as it is now",
            "missing": "carrying no readable capture time, so its age cannot be established",
            "future": "dated ahead of the clock, which is a clock disagreement or an edited "
                      "file rather than the freshest possible reading",
        }.get(fresh_state, f"in state {fresh_state}")
        card = _blocked_card(
            my_team, now=now, prov=prov, league=league, refused_context=refused_context,
            reason=_reason("snapshot_not_current",
                           f"The league snapshot is {described}. No decision is offered, and "
                           "last week's card is not reused in its place."))
        validate(card)
        return card
    if not prov["complete"]:
        card = _blocked_card(
            my_team, now=now, prov=prov, league=league, refused_context=refused_context,
            reason=_reason("provenance_incomplete",
                           "This run cannot say which model, scoring rules or capture produced "
                           f"it ({', '.join(prov['missing'])} missing), so no recommendation is "
                           "offered that could not be checked later."))
        validate(card)
        return card

    lineup = _current_lineup(my_team, team_id=team_id)
    changes = _lineup_change_rows(my_team, team_id=team_id)

    ranked: list[dict] = [
        _start_decision(row, context=usable_context, prov=prov) for row in changes
    ]
    ranked.extend(_waiver_rows(my_team, team_id=team_id, context=usable_context, prov=prov))
    # "Hold" is a claim that nothing needs changing, so it may only be made when
    # nothing was found to change.  A pending swap whose spread could not be
    # measured is an open question, not a settled lineup, and printing both
    # would tell the reader two different things on one page.
    if lineup["status"] == "ok" and not ranked:
        ranked.append(_hold_decision(lineup, context=usable_context, prov=prov))

    kind_rank = {"lineup": 0, "waiver": 1, "trade": 2}
    ranked.sort(key=lambda d: (
        0 if d["status"] in ACTIONABLE_STATUSES else 1,
        -(d.get("mean_delta") or 0.0),
        kind_rank.get(d.get("kind"), 9),
        str(d.get("slot") or ""),
        (d.get("subject") or {}).get("espn_player_id") or 0,
    ))

    kept: list[dict] = []
    withheld: list[dict] = []
    actionable = 0
    for decision in ranked:
        if decision["status"] in ACTIONABLE_STATUSES:
            if actionable >= MAX_ACTIONABLE_DECISIONS:
                withheld.append({"headline": decision["headline"],
                                 "mean_delta": decision.get("mean_delta"),
                                 "reason": "beyond the three-decision budget"})
                continue
            actionable += 1
        kept.append(decision)

    card = {
        "schema_version": SCHEMA_VERSION,
        "source_contract": SOURCE_CONTRACT,
        "visibility": VISIBILITY,
        "generated_at": now,
        "state": "ok" if lineup["status"] == "ok" else "no_current_pick",
        "reason": None if lineup["status"] == "ok" else dict(lineup["reason"]),
        "league": league,
        "provenance": prov,
        "current_lineup": lineup,
        "lineup_changes": [
            {"slot": row["slot"], "start": row["start"], "sit": row["sit"],
             "mean_delta": row["mean_delta"], "median_delta": row["median_delta"]}
            for row in changes
        ],
        "decisions": kept,
        "withheld": withheld,
        "shadow_seats": _shadow_seats(my_team),
        "alerts": _alerts(my_team, refused_context=refused_context, withheld=withheld),
        "espn_use": ESPN_USE,
        "prose_rewrite": {"applied": False,
                          "reason": "the default path composes every sentence in code"},
    }
    validate(card)
    return card


# --------------------------------------------------------------------------- #
# Validation — runs on every card, including one an LLM has touched
# --------------------------------------------------------------------------- #
REQUIRED_DECISION_KEYS = (
    "status", "kind", "slot", "subject", "alternative", "forced", "headline", "reason",
    "mean_delta", "median_delta", "interval", "model_relative_probability",
    "drivers", "risk", "invalidation_trigger", "provenance",
)
REQUIRED_PROVENANCE_KEYS = ("model_version", "scoring_hash", "snapshot_hash", "freshness_state")


def validate(card: Mapping[str, Any]) -> None:
    """Refuse a card that breaks its own contract.  Never repairs one."""
    if card.get("schema_version") != SCHEMA_VERSION:
        raise CardRejected(f"card claims schema {card.get('schema_version')!r}")
    if card.get("visibility") != VISIBILITY:
        raise CardRejected("a decision card is private; visibility may not be changed")

    decisions = card.get("decisions") or []
    actionable = [d for d in decisions if d.get("status") in ACTIONABLE_STATUSES]
    if len(actionable) > MAX_ACTIONABLE_DECISIONS:
        raise CardRejected(f"{len(actionable)} actionable decisions exceeds the budget of "
                           f"{MAX_ACTIONABLE_DECISIONS}")

    for index, decision in enumerate(decisions):
        where = f"decisions[{index}]"
        missing = [key for key in REQUIRED_DECISION_KEYS if key not in decision]
        if missing:
            raise CardRejected(f"{where} is missing {missing}")
        if decision["status"] not in DECISION_STATUSES:
            raise CardRejected(f"{where} has unknown status {decision['status']!r}")
        if len(decision.get("drivers") or []) > MAX_DRIVERS:
            raise CardRejected(f"{where} carries more than {MAX_DRIVERS} drivers")
        for driver in decision.get("drivers") or []:
            if not driver.get("source") or not driver.get("as_of"):
                raise CardRejected(f"{where} has a driver without a source and as-of timestamp")
        prov = decision.get("provenance") or {}
        absent = [key for key in REQUIRED_PROVENANCE_KEYS if not prov.get(key)]
        if absent:
            raise CardRejected(f"{where} cannot state {absent}")
        if decision["status"] in ACTIONABLE_STATUSES:
            if not decision.get("risk") or not str(decision["risk"].get("text") or "").strip():
                raise CardRejected(f"{where} is actionable with no stated risk")
            if not str(decision.get("invalidation_trigger") or "").strip():
                raise CardRejected(f"{where} is actionable with no invalidation trigger")
            interval_state = (decision.get("interval") or {}).get("status")
            # An action may skip the interval only where there is nothing to
            # measure: holding a lineup, or filling a seat whose occupant
            # cannot play. Everywhere else a recommendation without a range is
            # a mean wearing a decision's clothes.
            excused = decision["status"] == "hold" or decision.get("forced")
            if interval_state != "ok" and not (excused and interval_state == "not_applicable"):
                raise CardRejected(f"{where} is actionable without a joint-simulation interval")

    for path, text in _walk_prose(card):
        offending = prose_violations(text)
        if offending:
            raise CardRejected(f"{path} uses vocabulary this card does not show: {offending}")


# --------------------------------------------------------------------------- #
# Optional prose rewrite — off by default, and fail-closed when on
# --------------------------------------------------------------------------- #
PROSE_FIELDS = ("headline", "invalidation_trigger")


def _protected_tokens(decision: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for side in ("subject", "alternative"):
        entry = decision.get(side) or {}
        for key in ("name", "position", "slot"):
            value = entry.get(key)
            if value:
                tokens.add(str(value))
    if decision.get("slot"):
        tokens.add(str(decision["slot"]))
    return tokens


def apply_prose_rewrite(card: Mapping[str, Any], rewriter: Callable[[str], str]) -> dict:
    """Let a language model restate sentences after every number is fixed.

    The rewrite may change words and nothing else.  A candidate is accepted
    only if it carries the same multiset of numerals, still names every player,
    slot and position the original named, introduces no forbidden vocabulary,
    and does not grow beyond twice the original.  Any failure keeps the
    original sentence and is recorded — the card never degrades into whatever
    the model happened to say, and the whole card is re-validated afterwards.
    """
    import copy

    out = copy.deepcopy(dict(card))
    rejected: list[dict] = []
    rewritten = 0
    for index, decision in enumerate(out.get("decisions") or []):
        protected = _protected_tokens(decision)
        for field in PROSE_FIELDS:
            original = decision.get(field)
            if not isinstance(original, str) or not original.strip():
                continue
            where = f"decisions[{index}].{field}"
            try:
                candidate = rewriter(original)
            except Exception as exc:  # a failed rewrite is never fatal
                rejected.append({"path": where, "reason": f"rewriter raised {type(exc).__name__}"})
                continue
            problem = _rewrite_problem(original, candidate, protected)
            if problem:
                rejected.append({"path": where, "reason": problem})
                continue
            decision[field] = candidate
            rewritten += 1
    out["prose_rewrite"] = {
        "applied": True,
        "reason": "sentences restated after the numbers and the order were fixed",
        "rewritten": rewritten,
        "rejected": rejected,
    }
    validate(out)
    return out


def _rewrite_problem(original: str, candidate: Any, protected: set[str]) -> str | None:
    if not isinstance(candidate, str) or not candidate.strip():
        return "rewrite is empty"
    if len(candidate) > 2 * len(original) + 40:
        return "rewrite is more than twice the original length"
    if sorted(_NUMERAL_RE.findall(original)) != sorted(_NUMERAL_RE.findall(candidate)):
        return "rewrite changed the numbers"
    missing = sorted(token for token in protected
                     if token in original and token not in candidate)
    if missing:
        return f"rewrite dropped identifiers {missing}"
    # Named generically on purpose: this string is recorded on the card, and a
    # rejection that quotes the term it rejected would carry the banned word
    # onto the page through the audit trail.
    if prose_violations(candidate):
        return "rewrite introduced wording this card does not show"
    return None
