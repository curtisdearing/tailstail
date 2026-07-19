"""D.6: refuse to render language the protocol has already forbidden.

There was no literal "banned phrase list" in the standing invariants when this
was written. What DOES exist, pre-registered and frozen, is
``analysis/accuracy_protocol.json -> synthetic_lines.forbidden_claims``:

    ["profit", "ROI", "market edge", "closing-line value"]

with the surrounding rule that synthetic-line results "support trend and
regression tests only; no profit, ROI, market-edge, or CLV claim may be derived
from them." Every graded number this interface currently has comes from
synthetic-line replay, so those four claims are exactly the ones the UI must not
make. That list is the authoritative source here and is READ from the protocol
at scan time -- if the protocol is edited, the guard changes with it rather than
drifting from it.

On top of that, this module adds certainty and inevitability language. Those are
not in the protocol, and they are marked as a local extension rather than
smuggled in as if pre-registered. The justification is PREMORTEM.md's F8: the
documented failure mode is acting on an overestimated edge, and copy like
"lock" or "guaranteed" is how an overestimate gets communicated.

The guard is deliberately dumb: substring and word-boundary matching over
rendered output, run in CI. A cleverer semantic check would be easier to argue
with, and the point of a language gate is that it is not arguable.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL_PATH = os.path.join(ROOT, "analysis", "accuracy_protocol.json")

# Local extension: certainty/inevitability language. NOT pre-registered.
CERTAINTY_PHRASES: Sequence[str] = (
    "lock", "locks", "guaranteed", "guarantee", "can't lose", "cannot lose",
    "sure thing", "free money", "easy money", "no-brainer", "slam dunk",
    "must bet", "hammer", "max bet", "sharp play", "insider",
    "riskless", "risk-free", "printing money", "auto-bet",
)

# Claims that require a measured, price-beating result this tool does not have.
UNEARNED_CLAIM_PATTERNS: Sequence[str] = (
    r"\bbeats?\s+the\s+(?:closing\s+)?(?:line|market)\b",
    r"\bproven\s+edge\b",
    r"\bexpected\s+(?:profit|return)\b",
    r"\b\+?EV\b",
    r"\bunits?\s+won\b",
    r"\bbankroll\s+growth\b",
)

# A forbidden TERM is not a forbidden CLAIM. The protocol bars deriving a
# profit / ROI / market-edge / CLV claim from synthetic lines; it does not bar
# naming the metric in order to say it is unproven. A guard that cannot tell
# "no closing-line edge is established" from "closing-line edge" forces the UI
# to stop saying the honest thing, which inverts the intent.
#
# So a hit is suppressed when a negation or refusal cue sits near it. This is a
# lint, not a proof: an author determined to evade it could park a "not" nearby.
# It is calibrated to catch careless copy, which is the realistic failure.
NEGATION_CUES: Sequence[str] = (
    "no ", "not ", "never", "without", "cannot", "can't", "unproven",
    "unestablished", "insufficient_sample", "insufficient sample", "refuses",
    "refused", "makes no", "is not", "does not", "denies", "unmeasured",
    "no claim", "un-established",
)
CUE_WINDOW_BEFORE = 90
CUE_WINDOW_AFTER = 70


@dataclass
class Violation:
    phrase: str
    kind: str
    context: str
    source: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.phrase!r} ({self.source}) in: ...{self.context}..."


def protocol_forbidden_claims(path: Optional[str] = None) -> List[str]:
    """Read the pre-registered forbidden-claim list from the frozen protocol."""
    target = path or PROTOCOL_PATH
    if not os.path.exists(target):
        return []
    with open(target) as handle:
        protocol = json.load(handle)
    return list((protocol.get("synthetic_lines") or {}).get("forbidden_claims") or [])


def strip_markup(text: str) -> str:
    """Rendered HTML -> visible text. A banned word hidden in a class name is
    not a user-facing claim; one inside a <p> is."""
    without_blocks = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text,
                            flags=re.DOTALL | re.IGNORECASE)
    without_tags = re.sub(r"<[^>]+>", " ", without_blocks)
    return html.unescape(re.sub(r"\s+", " ", without_tags))


def _is_negated(haystack: str, index: int, length: int = 0) -> bool:
    start = max(0, index - CUE_WINDOW_BEFORE)
    end = index + length + CUE_WINDOW_AFTER
    context = haystack[start:end]
    return any(cue in context for cue in NEGATION_CUES)


def scan_text(text: str, *, source: str = "text",
              protocol_path: Optional[str] = None) -> List[Violation]:
    """Return every banned-language violation in visible copy."""
    lowered = text.lower()
    violations: List[Violation] = []

    def record(phrase: str, kind: str, index: int) -> None:
        if _is_negated(lowered, index, len(phrase)):
            return
        start = max(0, index - 45)
        violations.append(Violation(phrase=phrase, kind=kind, source=source,
                                    context=text[start:index + 55].strip()))

    for claim in protocol_forbidden_claims(protocol_path):
        for match in re.finditer(re.escape(claim.lower()), lowered):
            record(claim, "protocol_forbidden_claim", match.start())

    for phrase in CERTAINTY_PHRASES:
        pattern = r"\b" + re.escape(phrase.lower()).replace(r"\ ", r"\s+") + r"\b"
        for match in re.finditer(pattern, lowered):
            record(phrase, "certainty_language", match.start())

    for pattern in UNEARNED_CLAIM_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            record(match.group(0), "unearned_claim", match.start())

    return violations


def scan_html(markup: str, *, source: str = "html",
              protocol_path: Optional[str] = None) -> List[Violation]:
    return scan_text(strip_markup(markup), source=source, protocol_path=protocol_path)


def assert_clean(text: str, *, source: str = "output", is_html: bool = False,
                 protocol_path: Optional[str] = None) -> None:
    scanner = scan_html if is_html else scan_text
    violations = scanner(text, source=source, protocol_path=protocol_path)
    if violations:
        joined = "\n  ".join(str(v) for v in violations)
        raise AssertionError(
            f"{len(violations)} banned-language violation(s) in {source}:\n  {joined}\n"
            "Protocol-forbidden claims come from analysis/accuracy_protocol.json "
            "(synthetic_lines.forbidden_claims); certainty language is a local extension "
            "justified by PREMORTEM.md F8.")


def describe_ruleset(protocol_path: Optional[str] = None) -> Dict:
    return {
        "protocol_forbidden_claims": protocol_forbidden_claims(protocol_path),
        "protocol_source": "analysis/accuracy_protocol.json#/synthetic_lines/forbidden_claims",
        "certainty_phrases": list(CERTAINTY_PHRASES),
        "unearned_claim_patterns": list(UNEARNED_CLAIM_PATTERNS),
        "negation_cues": list(NEGATION_CUES),
        "cue_window": {"before": CUE_WINDOW_BEFORE, "after": CUE_WINDOW_AFTER},
        "note": ("certainty and unearned-claim rules are a local extension, not "
                 "pre-registered; the forbidden-claim list is read from the frozen protocol"),
    }
