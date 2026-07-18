"""Typed failures for every external boundary.

A model that cannot tell "the market offered nothing" from "we failed to ask"
will happily publish the second as the first. Every silent fallback in this
codebase was one of those two lies: an empty list from a 500, a `{"dome":
False}` from a DNS failure, a stale roster frame from a timeout.

The rule Phase B imposes: an external call either returns data or raises one
of these. A caller that WANTS to continue degraded must say so explicitly and
record the degradation, so the freshness gate and the published artifact both
know the run was incomplete.

``SourceFetchError`` carries the attempt log, so a failure report says which
endpoint, how many tries, and what each attempt died of -- not just "failed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class NflValueError(Exception):
    """Base for every error this package raises deliberately."""


class ConfigError(NflValueError):
    """Configuration exists but is unusable. NEVER silently defaulted."""


class SchemaViolation(NflValueError):
    """A frame crossing a stage boundary does not satisfy its contract."""


@dataclass
class Attempt:
    number: int
    error: str
    status: Optional[int] = None
    retry_after: Optional[float] = None

    def __str__(self) -> str:
        status = f" status={self.status}" if self.status is not None else ""
        return f"#{self.number}{status}: {self.error}"


class SourceFetchError(NflValueError):
    """An external source could not be read after the configured attempts."""

    def __init__(self, source: str, url: str, attempts: List[Attempt],
                 detail: str = "") -> None:
        self.source = source
        self.url = url
        self.attempts = attempts
        trail = "; ".join(str(a) for a in attempts)
        suffix = f" -- {detail}" if detail else ""
        super().__init__(
            f"{source}: {len(attempts)} attempt(s) failed for {url} [{trail}]{suffix}")


class SourceTimeout(SourceFetchError):
    """Every attempt timed out."""


class SourceUnavailable(SourceFetchError):
    """The source answered, but not with usable data (5xx, 429, bad payload)."""


class SourceRejected(SourceFetchError):
    """The source refused us (4xx other than 429). Retrying will not help."""


@dataclass
class Degradation:
    """A recorded, deliberate decision to continue without a source.

    Existence of one of these is the difference between a degraded run and a
    run that merely looks clean. ``load_bearing`` marks a source whose absence
    should block publication rather than quietly widen the error bars.
    """

    source: str
    reason: str
    load_bearing: bool = False
    detail: str = ""

    def as_dict(self) -> dict:
        return {"source": self.source, "reason": self.reason,
                "load_bearing": self.load_bearing, "detail": self.detail}


@dataclass
class DegradationLog:
    """Collects degradations across a run so the artifact can carry them."""

    entries: List[Degradation] = field(default_factory=list)

    def record(self, source: str, reason: str, *, load_bearing: bool = False,
               detail: str = "") -> Degradation:
        entry = Degradation(source=source, reason=reason,
                            load_bearing=load_bearing, detail=detail)
        self.entries.append(entry)
        return entry

    @property
    def degraded(self) -> bool:
        return bool(self.entries)

    @property
    def blocking(self) -> List[Degradation]:
        return [e for e in self.entries if e.load_bearing]

    def as_list(self) -> List[dict]:
        return [e.as_dict() for e in self.entries]
