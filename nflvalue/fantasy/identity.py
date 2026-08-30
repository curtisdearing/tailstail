"""One ESPN-id -> model-id crosswalk, reused rather than reinvented.

The ESPN comparison path already proved a crosswalk: nflverse weekly rosters
carry both `espn_id` and `gsis_id`, `espn_compare.build_identity_map` turns
that into a table, and `espn_compare.match_players` reports every failure mode
instead of dropping rows. That is the implementation; this module is the door
everything else uses so a second one never gets written.

The rule that matters is about names. A name is not an identity: two players
share one, a player changes one, and a suffix moves. So a name match is a
*reported fallback* — it can tell a human "this is probably Josh Allen" — and
it may never become an id. Anything that resolves by name arrives labelled,
and anything that resolves by nothing at all arrives in `unresolved` rather
than quietly disappearing or being scored as a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


class IdentityError(RuntimeError):
    """The crosswalk cannot be built, so no identity claim can be made."""


@dataclass(frozen=True)
class Resolution:
    """Who was matched, how, and who was not."""

    matched: Mapping[int, str]
    name_candidates: Mapping[int, str]
    unresolved: tuple[Mapping[str, Any], ...] = field(default=())

    @property
    def method(self) -> str:
        return "espn_id" if not self.name_candidates else "espn_id+reported_name_fallback"


def normalize_name(value: Any) -> str:
    """Lowercase, punctuation-free, suffix-free — for *reporting* only."""
    text = str(value or "").lower().strip()
    for junk in (".", ",", "'", "-"):
        text = text.replace(junk, " ")
    parts = [p for p in text.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def build_crosswalk(rosters: Any, season: int) -> dict[int, str]:
    """`{espn_id: gsis_id}` from the nflverse weekly rosters.

    Delegates to `espn_compare.build_identity_map`, which fails closed when the
    nflverse schema loses either column — a crosswalk that silently returns
    nothing would look exactly like a league of unknown players.
    """
    from .espn_compare import build_identity_map

    identity = build_identity_map(rosters, season)
    return {int(espn): str(gsis) for espn, gsis in zip(identity["espn_id"], identity["gsis_id"])}


def resolve(entries: Sequence[Mapping[str, Any]], crosswalk: Mapping[int, str] | None, *,
            known_names: Mapping[str, str] | None = None,
            id_key: str = "player_id", name_key: str = "full_name") -> Resolution:
    """Map roster/pool entries onto model ids, reporting every miss.

    `known_names` is an optional `{normalized name: model id}` index. When an
    ESPN id has no crosswalk row, a name hit is recorded in `name_candidates`
    — never in `matched`, and never used as an id by a caller that has not
    looked at it.
    """
    crosswalk = crosswalk or {}
    names = known_names or {}
    matched: dict[int, str] = {}
    candidates: dict[int, str] = {}
    unresolved: list[dict[str, Any]] = []

    for entry in entries:
        raw_id = entry.get(id_key)
        if not isinstance(raw_id, int):
            unresolved.append({
                "espn_player_id": raw_id, "name": entry.get(name_key),
                "reason": "entry carries no integer ESPN player id",
            })
            continue
        model_id = crosswalk.get(int(raw_id))
        if model_id is not None:
            matched[int(raw_id)] = str(model_id)
            continue
        guess = names.get(normalize_name(entry.get(name_key)))
        if guess is not None:
            candidates[int(raw_id)] = str(guess)
            unresolved.append({
                "espn_player_id": int(raw_id), "name": entry.get(name_key),
                "name_candidate": str(guess),
                "reason": ("no ESPN-id crosswalk row; a name match is reported for a human to "
                           "confirm and is never used as an identity"),
            })
            continue
        unresolved.append({
            "espn_player_id": int(raw_id), "name": entry.get(name_key),
            "reason": ("no identity crosswalk to a model player_id; excluded rather than "
                       "scored as zero"),
        })
    return Resolution(matched=matched, name_candidates=candidates,
                      unresolved=tuple(unresolved))


def index_names(model_ids: Iterable[str], names: Mapping[str, str]) -> dict[str, str]:
    """`{normalized name: model id}` for the reported-fallback path only."""
    index: dict[str, str] = {}
    for model_id in model_ids:
        key = normalize_name(names.get(model_id))
        if not key or key in index:
            index[key] = ""      # ambiguous: a shared name resolves to nobody
            continue
        index[key] = str(model_id)
    return {k: v for k, v in index.items() if v}
