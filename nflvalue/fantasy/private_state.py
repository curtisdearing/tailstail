"""The only path between this worktree and the private raw-state repository.

Two files have to survive between weekly runs and may never be published: the
row-level ESPN comparison ledger, and the immutable pre-kickoff ESPN captures.
Both are ESPN's per-player projections, fetched under terms recorded on every
capture that grant no redistribution right.

They used to travel in an ``actions/cache`` entry. That is not a security
boundary: GitHub documents that a pull-request workflow can restore caches
created on the default branch, and says in terms not to store sensitive
material in one. A separate private repository, reached with a write-scoped
deploy key that only trusted production runs are given, is the boundary.

This module is the whole of the copy. It exists so that "which files cross"
is one short allow-list in one place rather than a ``cp`` in a workflow step
that grows a wildcard the day somebody is in a hurry.

What it refuses
---------------
* Anything not on the allow-list: not copied, and named in the report.
* A symlink anywhere on the path, or a path that resolves outside its root:
  refused outright. Those are how a store somebody else can write turns into
  a write anywhere on the runner.
* A file that carries a league snapshot, a personalised card, roster or member
  data, or anything credential-shaped: refused outright. The raw captures are
  player projections; if one of them is carrying league identity, something
  upstream is wrong and copying it into a repository is the wrong response.

What it never does
------------------
Print file contents, or the value that tripped a check. Every message names a
path and the *kind* of problem. A guard that quotes what it rejected is a guard
that leaks, and everything on the other side of this boundary is private.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: Single files that may cross, by exact repository-relative path.
ALLOWED_FILES: tuple[str, ...] = ("data/espn_comparison_ledger.json",)
#: Directories whose ``*.json`` contents may cross.
ALLOWED_TREES: tuple[str, ...] = ("data/espn_snapshots",)
#: Suffix a file inside an allow-listed tree must have.
ALLOWED_TREE_SUFFIX = ".json"

#: Directories never walked when classifying a store.
SKIP_DIRS = frozenset({".git", "__pycache__", ".github"})


#: Contract markers that identify something which is not a raw ESPN capture.
#: Reused from the public boundary so there is one list of what a private
#: contract looks like.
_LEAGUE_MARKER = "espn-league/"
_PERSONAL_MARKERS = ("decision-card/", "my_team/")
#: Credential shapes. ESPN uses the SWID cookie value as a member id, so a
#: braced GUID in a capture is a credential, not an identifier.
_CREDENTIAL_RES = (
    re.compile(r"espn[_-]?s2", re.IGNORECASE),
    re.compile(r"\bswid\b", re.IGNORECASE),
    re.compile(r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
               r"[0-9A-Fa-f]{12}\}"),
)
_ROSTER_RES = (re.compile(r'"rosters"\s*:'), re.compile(r'"members"\s*:'))


class PrivateStateRejected(RuntimeError):
    """A copy was refused on safety grounds. Never worked around."""


@dataclass(frozen=True)
class Report:
    """What crossed, what did not, and why. Paths only -- never contents."""

    action: str
    available: bool
    copied: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        if not self.available:
            return f"{self.action}: private state unavailable; no raw state moved"
        return (f"{self.action}: {len(self.copied)} file(s) copied, "
                f"{len(self.skipped)} not on the allow-list")


def _is_allowed(relative: str) -> bool:
    if relative in ALLOWED_FILES:
        return True
    return any(relative.startswith(f"{tree}/") and relative.endswith(ALLOWED_TREE_SUFFIX)
               for tree in ALLOWED_TREES)


def _is_prefix_of_allowed(relative: str) -> bool:
    """True when *relative* is, or contains, something the allow-list names."""
    targets = ALLOWED_FILES + ALLOWED_TREES
    return any(target == relative or target.startswith(f"{relative}/") for target in targets)


def _resolve_inside(root: Path, relative: str) -> Path:
    """The absolute path of *relative* under *root*, or a refusal.

    ``Path.resolve`` collapses ``..``, so a relative path that climbs out of the
    root is visible here as a resolved path that is not under it. Checked on
    every crossing rather than trusted from the caller.
    """
    root_real = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root_real and root_real not in candidate.parents:
        raise PrivateStateRejected(
            f"refusing {relative!r}: it resolves outside {root_real}")
    return candidate


def _walk(root: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """Candidate files, looking ONLY where the allow-list points.

    The save direction runs with `--worktree .` on a CI runner, so walking the
    whole tree meant enumerating every file in the repository -- and reporting
    each one by name into a world-readable Actions log. Filenames are data: a
    league snapshot is named `espn-league-<league id>-<season>-<stamp>.json`,
    and that league id is exactly the string the public boundary refuses. So the
    walk starts at the allow-listed roots and never sees anything else.

    A symlink on an allow-listed path is a refusal, not a skip: a store another
    process can write is otherwise a write to anywhere the runner can reach.
    """
    files: list[tuple[str, Path]] = []
    skipped: list[tuple[str, str]] = []

    def check_symlink(relative: str, entry: Path) -> None:
        if entry.is_symlink():
            raise PrivateStateRejected(
                f"refusing {relative!r}: it is a symlink, and raw state is copied "
                "only from real files")

    for relative in ALLOWED_FILES:
        entry = root / relative
        if not entry.exists() and not entry.is_symlink():
            continue
        check_symlink(relative, entry)
        if entry.is_file():
            files.append((relative, entry))
        else:
            skipped.append((relative, "not a regular file"))

    for tree in ALLOWED_TREES:
        directory = root / tree
        if not directory.exists() and not directory.is_symlink():
            continue
        check_symlink(tree, directory)
        if not directory.is_dir():
            skipped.append((tree, "not a directory"))
            continue
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = entry.relative_to(root).as_posix()
            check_symlink(relative, entry)
            if entry.is_file() and relative.endswith(ALLOWED_TREE_SUFFIX):
                files.append((relative, entry))
            else:
                skipped.append((relative, "not an allow-listed capture"))

    return files, skipped


def _inspect(relative: str, path: Path) -> None:
    """Refuse a file whose content is not a raw ESPN capture.

    The captures are player projections; they legitimately name players. What
    they must never carry is league identity, a personalised contract, roster or
    member blocks, or anything credential-shaped -- if one does, something
    upstream put the wrong file here and copying it into a repository is the
    wrong response to that.

    Messages name the path and the kind of problem. Never the match.
    """
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise PrivateStateRejected(
            f"refusing {relative!r}: it is not readable as text ({type(exc).__name__})") from exc

    if _LEAGUE_MARKER in text:
        raise PrivateStateRejected(
            f"refusing {relative!r}: it carries a league snapshot marker, and league "
            "snapshots are not raw ESPN comparison state")
    for marker in _PERSONAL_MARKERS:
        if marker in text:
            raise PrivateStateRejected(
                f"refusing {relative!r}: it carries a personalised contract marker")
    for pattern in _CREDENTIAL_RES:
        if pattern.search(text):
            raise PrivateStateRejected(
                f"refusing {relative!r}: it carries something credential-shaped")
    for pattern in _ROSTER_RES:
        if pattern.search(text):
            raise PrivateStateRejected(
                f"refusing {relative!r}: it carries roster or member data")

    if relative in ALLOWED_FILES:
        from . import espn_compare

        try:
            ledger = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PrivateStateRejected(
                f"refusing {relative!r}: it is not readable JSON") from exc
        try:
            espn_compare.verify_ledger_integrity(ledger)
        except ValueError as exc:
            raise PrivateStateRejected(f"refusing {relative!r}: {exc}") from exc


def copy_one(source_root: Path, target_root: Path, relative: str) -> Path:
    """Copy one allow-listed file, checking the path on both sides.

    The symlink test runs on the UNRESOLVED path. `Path.resolve()` dereferences,
    so testing `is_symlink()` after resolving is always False -- the check reads
    like a second line of defence and is dead code.
    """
    raw_source = Path(source_root) / relative
    for part in (raw_source, *raw_source.parents):
        if part == Path(source_root):
            break
        if part.is_symlink():
            raise PrivateStateRejected(
                f"refusing {relative!r}: a symlink is on its path, and raw state is copied "
                "only from real files")
    source = _resolve_inside(Path(source_root), relative)
    destination = _resolve_inside(Path(target_root), relative)
    if not _is_allowed(relative):
        raise PrivateStateRejected(f"refusing {relative!r}: it is not on the allow-list")
    if not source.is_file():
        raise PrivateStateRejected(f"refusing {relative!r}: it is not a real file")
    _inspect(relative, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _copy(source_root: Path, target_root: Path, *, action: str) -> Report:
    files, skipped = _walk(source_root)
    copied: list[str] = []
    for relative, _path in files:
        if not _is_allowed(relative):
            skipped.append((relative, "not on the allow-list"))
            continue
        copy_one(source_root, target_root, relative)
        copied.append(relative)
    return Report(action=action, available=True,
                  copied=tuple(copied), skipped=tuple(sorted(skipped)))


def restore(store: str | Path, worktree: str | Path) -> Report:
    """Copy allow-listed raw state from the private store into the worktree."""
    store, worktree = Path(store), Path(worktree)
    if not store.is_dir():
        return Report(action="restore", available=False, warnings=(
            f"private state store {store} is not available; no raw ESPN state was "
            "restored this run",))
    return _copy(store, worktree, action="restore")


def save(worktree: str | Path, store: str | Path) -> Report:
    """Copy allow-listed raw state from the worktree into the private store."""
    worktree, store = Path(worktree), Path(store)
    if not store.is_dir():
        return Report(action="save", available=False, warnings=(
            f"private state store {store} is not available; raw ESPN state was not "
            "saved this run",))
    return _copy(worktree, store, action="save")


# --------------------------------------------------------------------------- #
# CLI — what the workflow calls
# --------------------------------------------------------------------------- #
# Exit codes carry the distinction the workflow has to act on:
#
#   0  the copy happened, or the store was not there and the run continues
#      without raw state (a `::warning::`, never a fallback);
#   2  a copy was REFUSED on safety grounds (`::error::`). That is a defect in
#      whatever wrote the store, and the right response is a red run, not a
#      quieter path to the same data.
EXIT_OK = 0
EXIT_REFUSED = 2


def _emit(report: Report) -> None:
    """Name what crossed; count what did not.

    A skipped path is still a filename, and this runs on a public repository
    whose Actions logs anyone can read.
    """
    for warning in report.warnings:
        print(f"::warning::{warning}")
    print(f"[private-state] {report.summary()}")
    for relative in report.copied:
        print(f"[private-state] copied {relative}")
    if report.skipped:
        print(f"[private-state] {len(report.skipped)} path(s) were not on the allow-list "
              "and are not named here")


def main(argv: list[str] | None = None) -> int:
    """Copy raw state in one direction. Prints paths and reasons, never content."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("restore", "save"))
    parser.add_argument("--store", required=True,
                        help="checkout of the private state repository")
    parser.add_argument("--worktree", default=".", help="this repository")
    args = parser.parse_args(argv)

    try:
        report = (restore(args.store, args.worktree) if args.action == "restore"
                  else save(args.worktree, args.store))
    except PrivateStateRejected as exc:
        # The exception text names a path and a kind of problem, never a value.
        print(f"::error::private state {args.action} refused: {exc}")
        return EXIT_REFUSED
    _emit(report)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
