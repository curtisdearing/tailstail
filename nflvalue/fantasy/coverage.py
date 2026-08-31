"""Does the producer actually emit every event this league prices?

"Exact custom scoring" is a claim about two things at once: that the contract
reproduces the league's rules, and that whatever is being scored *contains the
events those rules price*. The first has a test. The second did not, and the
gap was invisible because `score_components` reads its inputs with
`components.get(key, 0.0)` — an event nobody modelled and an event that did
not happen are the same number.

That is fine for a category the league scores at zero and wrong for every
other one. This league pays 4.0 for a receiving two-point conversion; the
weekly simulator emits no two-point-conversion component at all, so every
simulated season silently assumes nobody ever converts one. The projection is
not wrong by much. It is wrong by an amount nobody can state, which is worse,
because the output still says "exact".

So coverage is audited explicitly, and a producer that cannot back a priced
category does not get the label. Nothing here changes a number: this module
computes no points and is not on the scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: Scoring-contract category key -> the component name a producer must emit
#: for that category to be scored from real modelled events.
#:
#: Only the offensive skill-position categories appear here. Kicking, defence,
#: points-allowed and yards-allowed are scored from separate shadow artifacts
#: with their own provenance (`shadow_kicker`, `special_scoring`), so they are
#: not claims this audit can settle and are reported as `shadow`, not as
#: covered.
CATEGORY_COMPONENTS: Mapping[str, str] = {
    "passing_yards": "passing_yards",
    "passing_td": "passing_tds",
    "interception_thrown": "passing_interceptions",
    "rushing_yards": "rushing_yards",
    "rushing_td": "rushing_tds",
    "receiving_yards": "receiving_yards",
    "receiving_td": "receiving_tds",
    "reception": "receptions",
    "fumble_lost": "fumbles_lost",
    "passing_2pt": "passing_2pt_conversions",
    "rushing_2pt": "rushing_2pt_conversions",
    "receiving_2pt": "receiving_2pt_conversions",
}

#: Groups scored outside the offensive simulation entirely.
SHADOW_GROUPS = frozenset({"kicking", "defense", "points_allowed", "yards_allowed", "return"})

#: Categories with no modelled event anywhere in this project. `team_win`
#: (statId 155) is parsed and hashed by the contract and consumed by nothing:
#: there is no win-probability term on the fantasy path, so a league that pays
#: for a Team Win is not being scored exactly, whatever the components say.
NO_MODELLED_EVENT: Mapping[str, str] = {
    "team_win": "no win-probability term exists on the fantasy scoring path",
    "fumble_recovered_td": (
        "an offensive player recovering a fumble for a touchdown is not sampled by "
        "the weekly simulator and has no historical component"),
}


@dataclass(frozen=True)
class CoverageReport:
    """What a producer can and cannot back, and whether "exact" is honest."""

    exact: bool
    covered: tuple[str, ...]
    unsupported: tuple[str, ...]
    shadow: tuple[str, ...]
    zero_valued: tuple[str, ...]
    reasons: Mapping[str, str]

    def label(self) -> str:
        return "exact_custom" if self.exact else "custom_with_unsupported_categories"


def required_components(contract: Any) -> frozenset[str]:
    """Every component a producer must emit to back this contract exactly."""
    return frozenset(
        CATEGORY_COMPONENTS[key]
        for key, category in _priced(contract).items()
        if key in CATEGORY_COMPONENTS
    )


def _priced(contract: Any) -> dict[str, Any]:
    """Categories this league actually pays for. A 0.0 category is not a claim."""
    priced = {}
    for key, category in (getattr(contract, "categories", {}) or {}).items():
        points = float(getattr(category, "points", 0.0) or 0.0)
        overrides = getattr(category, "position_overrides", {}) or {}
        if points != 0.0 or any(float(v) != 0.0 for v in overrides.values()):
            priced[key] = category
    return priced


def audit(contract: Any, *, emitted: Iterable[str]) -> CoverageReport:
    """Compare what the league prices against what the producer emits."""
    produced = frozenset(emitted)
    categories = getattr(contract, "categories", {}) or {}
    priced = _priced(contract)

    covered: list[str] = []
    unsupported: list[str] = []
    shadow: list[str] = []
    reasons: dict[str, str] = {}

    for key in sorted(categories):
        category = categories[key]
        group = str(getattr(category, "group", ""))
        if key not in priced:
            continue
        if key in NO_MODELLED_EVENT:
            unsupported.append(key)
            reasons[key] = NO_MODELLED_EVENT[key]
            continue
        if group in SHADOW_GROUPS:
            shadow.append(key)
            reasons[key] = f"scored by the {group} shadow artifact, not this producer"
            continue
        component = CATEGORY_COMPONENTS.get(key)
        if component is None:
            unsupported.append(key)
            reasons[key] = "no component is mapped for this category"
        elif component not in produced:
            unsupported.append(key)
            reasons[key] = (
                f"the league prices {key} but the producer emits no {component!r}; "
                "scoring it would read a missing component as zero")
        else:
            covered.append(key)

    zero_valued = tuple(sorted(set(categories) - set(priced)))
    return CoverageReport(
        exact=not unsupported,
        covered=tuple(covered),
        unsupported=tuple(sorted(unsupported)),
        shadow=tuple(sorted(shadow)),
        zero_valued=zero_valued,
        reasons=reasons,
    )
