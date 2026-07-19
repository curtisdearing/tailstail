"""C.6: the plain-language translator. Last step, and strictly downstream.

The translator receives ONLY the computed evidence payload. It may not compute,
look up, infer, or round anything into existence, and it may not reorder the
drivers -- the ranking is a model output, and a prose layer that reorders it is
quietly re-weighting the model.

Two guards make that enforceable rather than aspirational:

* ``assert_numerals_grounded`` -- every numeral in the prose must appear in the
  payload. A model that writes "hit 58% of the time" when the payload says
  0.5743 is inventing a number, and this rejects it.
* ``assert_order_preserved`` -- driver labels must appear in prose in payload
  order.

Following ``synthesis.RuleBasedMockLLM``, the default renderer is deterministic
and needs no network. A real client can be injected, but its output goes
through the same guards and is REJECTED on violation rather than patched --
silently repairing a translator that fabricates numbers just hides that it does.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Set

NUMERAL = re.compile(r"-?\d+(?:\.\d+)?")


class TranslationRejected(RuntimeError):
    """The prose failed a grounding guard and must not be shown."""


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def _numerals(text: str) -> List[str]:
    return NUMERAL.findall(text or "")


def _normalize(token: str) -> str:
    """Compare numerals by value, so 0.5743 and .5743 and 0.57430 agree."""
    try:
        value = float(token)
    except ValueError:
        return token
    if value == int(value):
        return str(int(value))
    return repr(round(value, 10))


def payload_numerals(payload) -> Set[str]:
    """Every numeral the prose is allowed to contain.

    Structure-aware on purpose. A blind walk that admits ``value * 100`` for
    every number is far too permissive: a volume delta of 0.733 TARGETS then
    licenses the string "73", and a translator writing "hits 73% of the time"
    sails through. Percentage and multiplier renderings are therefore admitted
    only for values that are actually rates or multipliers, which the payload
    already tells us via each driver's ``units`` and via known rate-valued keys.

    Known limit, stated rather than papered over: this proves every numeral is
    DERIVABLE from some payload value. It cannot prove the translator attached
    it to the right subject. Numeral grounding plus order preservation are
    necessary conditions, not a correctness proof of the prose.
    """
    found: Set[str] = set()

    def add_plain(value: float) -> None:
        found.add(_normalize(str(value)))
        for digits in (0, 1, 2, 3, 4):
            found.add(_normalize(f"{value:.{digits}f}"))
            found.add(_normalize(f"{abs(value):.{digits}f}"))

    def add_rate(value: float) -> None:
        for digits in (0, 1, 2, 3):
            found.add(_normalize(f"{value * 100:.{digits}f}"))
            found.add(_normalize(f"{abs(value) * 100:.{digits}f}"))

    def add_multiplier(value: float) -> None:
        for digits in (0, 1, 2, 3):
            found.add(_normalize(f"{(value - 1.0) * 100:.{digits}f}"))
            found.add(_normalize(f"{abs(value - 1.0) * 100:.{digits}f}"))

    def walk(node, key: str = "", units: str = "") -> None:
        if isinstance(node, dict):
            local_units = str(node.get("units") or units)
            for child_key, value in node.items():
                walk(value, child_key, local_units)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value, key, units)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            value = float(node)
            add_plain(value)
            if key in RATE_KEYS or (key in ("value", "baseline", "delta", "ci95")
                                    and units == "multiplier"):
                add_rate(value)
            if units == "multiplier" and key in ("value", "baseline"):
                add_multiplier(value)
        elif isinstance(node, str):
            for token in _numerals(node):
                found.add(_normalize(token))
                try:
                    add_plain(float(token))
                except ValueError:
                    continue

    walk(payload)
    return found


# Keys whose values are genuinely rates or probabilities, so a percentage
# rendering of them is a faithful restatement rather than a new number.
RATE_KEYS = frozenset({
    "hit_rate", "pooled_hit_rate", "all_candidates_baseline", "coverage80",
    "nominal_coverage", "model_probability", "p_over", "p_under", "p",
    "multiplicity_q", "weight_on_own_history", "ci95", "exposed_rate",
    "control_rate", "sign_agreement", "share_of_projection_pct",
})


def assert_numerals_grounded(prose: str, payload: Dict) -> None:
    allowed = payload_numerals(payload)
    ungrounded = [t for t in _numerals(prose) if _normalize(t) not in allowed]
    if ungrounded:
        raise TranslationRejected(
            f"prose contains numeral(s) absent from the evidence payload: "
            f"{sorted(set(ungrounded))}. The translator may not introduce numbers.")


def assert_order_preserved(prose: str, payload: Dict) -> None:
    labels = [d["label"] for d in payload["decomposition"]["drivers"]]
    positions = []
    for label in labels:
        index = prose.find(label)
        if index >= 0:
            positions.append((label, index))
    ordered = [label for label, _ in positions]
    expected = [label for label in labels if label in ordered]
    if ordered != expected or positions != sorted(positions, key=lambda p: p[1]):
        raise TranslationRejected(
            f"prose reorders the drivers. Payload order: {expected}; prose order: {ordered}. "
            "Driver ranking is a model output and the translator may not change it.")


# --------------------------------------------------------------------------- #
# Deterministic renderer
# --------------------------------------------------------------------------- #
def _strength_phrase(evidence: Dict) -> str:
    if evidence.get("status") == "REGISTERED":
        n_raw = evidence.get("n_raw") or {}
        posterior = evidence.get("posterior") or {}
        ci = posterior.get("ci95")
        bits = [f"registered as {evidence.get('registry_id')} ({evidence.get('strength_label')})"]
        if n_raw.get("exposed") is not None:
            bits.append(f"n={n_raw.get('exposed')} exposed vs {n_raw.get('control')} control")
        if evidence.get("n_effective") is not None:
            bits.append(f"effective n {evidence['n_effective']}")
        if posterior.get("mean") is not None and ci:
            bits.append(f"posterior {posterior['mean']} (95% CI {ci[0]} to {ci[1]})")
        if evidence.get("multiplicity_q") is not None:
            bits.append(f"q={evidence['multiplicity_q']}")
        return "; ".join(bits)
    n_raw = evidence.get("n_raw") or {}
    if "player_games" in n_raw:
        shrink = evidence.get("shrinkage") or {}
        return (f"no registered study; the player's own history is {n_raw['player_games']} games, "
                f"effective n {evidence.get('n_effective')} after shrinkage "
                f"(weight {shrink.get('weight_on_own_history')} on his own data) "
                f"-- {evidence.get('strength_label')}")
    return f"no registered study and no usable sample -- {evidence.get('strength_label')}"


def render_rules(payload: Dict) -> str:
    """Deterministic prose. Every numeral is copied, never recomputed."""
    selection = payload["selection"]
    decomposition = payload["decomposition"]
    band = payload["calibration_band"]

    lines: List[str] = []
    head = (f"{selection.get('name')} {selection.get('market','').replace('_',' ')} "
            f"{selection.get('side')} {selection.get('line')} "
            f"({selection.get('team')} vs {selection.get('opponent')}, "
            f"{selection.get('season')} week {selection.get('week')}). "
            f"The model projects {selection.get('projected_mean')}.")
    lines.append(head)

    lines.append("")
    lines.append("What is driving it, strongest first:")
    for driver in decomposition["drivers"]:
        lines.append(f"{driver['rank']}. {driver['label']} — {driver['statement']} "
                     f"[{driver['direction']} this side]. "
                     f"Evidence: {_strength_phrase(driver.get('evidence') or {})}.")
        for note in driver.get("notes") or []:
            lines.append(f"   Caveat: {note}")

    lines.append("")
    if band.get("status") == "MEASURED":
        ci = band.get("ci95") or []
        interval = f", 95% CI {ci[0]} to {ci[1]}" if len(ci) == 2 else ""
        lines.append(f"Calibration: selections in composite band {band['band']} hit "
                     f"{band['hit_rate']} over n={band['n']}{interval}. "
                     f"Pooled rate is {band.get('pooled_hit_rate')} over n={band.get('pooled_n')}.")
    else:
        lines.append(f"Calibration: UNMEASURED_BUCKET — {band.get('reason')}. "
                     "No strength claim is made for this selection.")

    lines.append("")
    lines.append("What argues against it:")
    if payload["counter_evidence"]:
        for item in payload["counter_evidence"]:
            lines.append(f"- {item['detail']}")
    else:
        lines.append("- Nothing found by the checks that ran. That is a claim about the "
                     "checks, not a clean bill of health.")

    lines.append("")
    lines.append("What would flip it:")
    for item in payload["falsifiers"]:
        entry = f"- {item['statement']} (observable by: {item['observable_by']})"
        if item.get("gap"):
            entry += f" [gap: {item['gap']}]"
        lines.append(entry)

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def translate(payload: Dict, client: Optional[Callable[[str], str]] = None, *,
              prompt_builder: Optional[Callable[[Dict], str]] = None) -> Dict:
    """Render ``payload`` as prose and verify it against the payload.

    ``client`` is any callable taking a prompt and returning prose. It sees the
    payload and nothing else. Its output is verified, not trusted; a violation
    raises rather than falling back silently, so a fabricating translator
    surfaces as a failure instead of degrading into the rule-based text and
    looking fine.
    """
    if client is None:
        prose = render_rules(payload)
        source = "rule_based"
    else:
        prompt = (prompt_builder or _default_prompt)(payload)
        prose = client(prompt)
        source = "client"
    assert_numerals_grounded(prose, payload)
    assert_order_preserved(prose, payload)
    return {"prose": prose, "source": source, "schema_version": payload.get("schema_version"),
            "verified": {"numerals_grounded": True, "driver_order_preserved": True}}


def _default_prompt(payload: Dict) -> str:
    import json
    return (
        "Rewrite the following computed evidence as plain prose for a reader who is "
        "not a modeller.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Use ONLY numbers that appear in the JSON. Do not compute, round into a new "
        "value, convert units, or estimate any figure.\n"
        "- Keep the drivers in exactly the order given. Do not re-rank them.\n"
        "- Preserve every driver label verbatim so ordering can be verified.\n"
        "- Render the counter-evidence and falsifiers sections even if they are short.\n"
        "- Do not add a recommendation, a confidence word, or a bet size.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )
