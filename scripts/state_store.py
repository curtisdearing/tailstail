#!/usr/bin/env python3
"""Create and restore checksummed, versioned production-state archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STATE_PROFILES = {
    "prop": (
        "data/*.db",
        "data/*.joblib",
        "data/ml_frame.parquet",
        "data/history.json",
        "data/latest.json",
        "data/weekly.json",
        "data/weekly_props.json",
        "data/weights.json",
    ),
    # A release asset on a public repository is published material. This
    # profile therefore carries only material that may be published.
    #
    # `data/espn_comparison_history.json` is the aggregate grading history
    # (`espn-comparison-history/1`): one immutable row per graded week, counts
    # and MAEs and a sha256, and nothing that could be a player. It lives here
    # because the published season series has to survive between runs on its
    # own -- if it could only be rebuilt from the private row-level ledger, a
    # run that could not reach that ledger would publish an empty season.
    #
    # The row-level ledger and the pre-kickoff ESPN captures are NOT here and
    # must never be. They are ESPN's per-player projections, and the terms
    # recorded on every capture
    # (nflvalue/sources/espn_projections.REDISTRIBUTION_RIGHTS) grant no
    # redistribution right and say raw snapshots are "retained for audit, not
    # republication". They travel between runs through the private state
    # repository instead (nflvalue/fantasy/private_state.py); they are not put
    # in an Actions cache either, because a cache is restorable from
    # pull-request workflows against the default branch and GitHub documents it
    # as unsuitable for sensitive material.
    "fantasy": (
        "data/fantasy_model.joblib",
        "data/player_projection_snapshot.json",
        "data/espn_comparison_history.json",
        # Prior seasons, archived by `espn_compare.load_history` at the first
        # run of a new one. Same contract, same guarantees; they travel here
        # because this file is gitignored and the release is its only copy.
        "data/espn_comparison_history.*.json",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_files(root: Path = ROOT, profile: str = "prop") -> list[Path]:
    if profile not in STATE_PROFILES:
        raise ValueError(f"unknown state profile: {profile}")
    files: set[Path] = set()
    for pattern in STATE_PROFILES[profile]:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def pack(archive: Path, root: Path = ROOT, profile: str = "prop") -> dict[str, object]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = state_files(root, profile)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        os.replace(tmp, archive)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "profile": profile,
        "sha256": sha256(archive),
        "files": [path.relative_to(root).as_posix() for path in files],
    }


def _safe_member(member: tarfile.TarInfo, profile: str = "prop") -> PurePosixPath:
    path = PurePosixPath(member.name)
    if member.issym() or member.islnk() or not member.isfile():
        raise ValueError(f"state archive contains unsupported member: {member.name}")
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ValueError(f"state archive contains unsafe member: {member.name}")
    if profile not in STATE_PROFILES:
        raise ValueError(f"unknown state profile: {profile}")
    allowed = any(Path(path.as_posix()).match(pattern) for pattern in STATE_PROFILES[profile])
    if not allowed:
        raise ValueError(f"state archive contains undeclared state: {member.name}")
    return path


def restore(
    archive: Path, expected_sha: str, root: Path = ROOT, profile: str = "prop"
) -> list[str]:
    actual = sha256(archive)
    if actual != expected_sha:
        raise ValueError(f"state archive checksum mismatch: expected {expected_sha}, got {actual}")
    root.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    with tempfile.TemporaryDirectory(prefix="state-restore-", dir=root) as tmp_name:
        staging = Path(tmp_name)
        with tarfile.open(archive, "r:gz") as tar:
            members = [(member, _safe_member(member, profile)) for member in tar.getmembers()]
            for member, relative in members:
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read state member: {member.name}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
            for _, relative in members:
                source = staging / relative
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                restored.append(relative.as_posix())
    return restored


def write_pointer(
    archive: Path, asset: str, output: Path, profile: str = "prop"
) -> dict[str, object]:
    pointer = {
        "schema_version": 1, "profile": profile,
        "asset": asset, "sha256": sha256(archive),
    }
    output.write_text(json.dumps(pointer, indent=2) + "\n")
    return pointer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--archive", type=Path, required=True)
    p_pack.add_argument("--profile", choices=sorted(STATE_PROFILES), default="prop")
    p_restore = sub.add_parser("restore")
    p_restore.add_argument("--archive", type=Path, required=True)
    p_restore.add_argument("--sha", required=True)
    p_restore.add_argument("--profile", choices=sorted(STATE_PROFILES), default="prop")
    p_pointer = sub.add_parser("pointer")
    p_pointer.add_argument("--archive", type=Path, required=True)
    p_pointer.add_argument("--asset", required=True)
    p_pointer.add_argument("--output", type=Path, required=True)
    p_pointer.add_argument("--profile", choices=sorted(STATE_PROFILES), default="prop")
    args = parser.parse_args()
    if args.command == "pack":
        print(json.dumps(pack(args.archive, profile=args.profile), sort_keys=True))
    elif args.command == "restore":
        print(json.dumps({"restored": restore(
            args.archive, args.sha, profile=args.profile
        )}, sort_keys=True))
    else:
        print(json.dumps(write_pointer(
            args.archive, args.asset, args.output, profile=args.profile
        ), sort_keys=True))


if __name__ == "__main__":
    main()
