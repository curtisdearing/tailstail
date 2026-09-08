"""Mutation tests for the prospective-capture release (2026-09-08).

Each mutation edits a copy of the worktree, runs the guarding tests, and
reports whether the suite KILLED it (some test failed) or it SURVIVED.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/private/tmp/claude-501/-Users-curtisdearing/511e13c8-4b81-4dd2-9217-b6428c9ee2ae/scratchpad/tt-capture")
WORK = Path("/private/tmp/claude-501/-Users-curtisdearing/511e13c8-4b81-4dd2-9217-b6428c9ee2ae/scratchpad/mut")
PY = "/private/tmp/claude-501/-Users-curtisdearing/511e13c8-4b81-4dd2-9217-b6428c9ee2ae/scratchpad/venv/bin/python"

MUTATIONS = [
    # (id, description, file, old, new, tests)
    ("M1", "accept a snapshot generated AT the deadline (>= becomes >)",
     "nflvalue/fantasy/prospective_archive.py",
     "        if generated_at >= kickoff:\n",
     "        if generated_at > kickoff:\n",
     ["tests/test_prospective_archive.py"]),
    ("M1b", "ignore information_as_of when judging a deadline",
     "nflvalue/fantasy/prospective_archive.py",
     "        elif information_as_of >= kickoff:\n",
     "        elif False:\n",
     ["tests/test_prospective_archive.py"]),
    ("M1c", "ESPN ledger accepts a post-kickoff snapshot (< becomes <=)",
     "nflvalue/fantasy/espn_compare.py",
     "    return snap < kick\n",
     "    return snap <= kick\n",
     ["tests/test_espn_compare.py"]),
    ("M2", "overwrite an existing immutable archive entry",
     "nflvalue/fantasy/prospective_archive.py",
     "    if path.exists():\n        raise FileExistsError(\n            f\"refusing to overwrite immutable archive entry {path}; a new forecast \"\n            \"must produce a new timestamped entry\")\n",
     "    pass\n",
     ["tests/test_prospective_archive.py"]),
    ("M2b", "index silently accepts lost history",
     "nflvalue/fantasy/prospective_archive.py",
     "    if any(existing not in known for existing in on_disk):\n",
     "    if False:\n",
     ["tests/test_prospective_archive.py"]),
    ("M3", "expose a private field in the archive entry",
     "nflvalue/fantasy/prospective_archive.py",
     "        \"players\": players,\n    }\n    entry[\"payload_sha256\"]",
     "        \"players\": players,\n        \"league_name\": \"leak\", \"rosters\": [],\n    }\n    entry[\"payload_sha256\"]",
     ["tests/test_prospective_archive.py"]),
    ("M3b", "public payload allow-list leaks my_team",
     "nflvalue/fantasy/private_boundary.py",
     "    \"prospective_capture\",\n)",
     "    \"prospective_capture\", \"my_team\",\n)",
     ["tests/test_my_team_contract.py", "tests/test_prospective_capture_boundary.py",
      "tests/test_workflow_trust_boundary.py"]),
    ("M4", "enable the role mixture by default",
     "nflvalue/fantasy/config.py",
     "    role_scenario_mixture: bool = False\n",
     "    role_scenario_mixture: bool = True\n",
     ["tests/test_prospective_capture_boundary.py", "tests/test_role_state.py"]),
    ("M5", "fetch bypass downloads nothing for the lagging season (silently drops it)",
     "nflvalue/fantasy/data.py",
     "            if current_season is not None and latest > current_season:\n",
     "            if False:\n",
     ["tests/test_fantasy_fetch_seasons.py"]),
    ("M6", "kickoff defaulted to 13:00 when missing (the old behaviour)",
     "nflvalue/fantasy/espn_compare.py",
     "        if not gametime or gametime.lower() in {\"nan\", \"none\"}:\n            continue\n",
     "        if not gametime or gametime.lower() in {\"nan\", \"none\"}:\n            gametime = \"13:00\"\n",
     ["tests/test_espn_compare.py"]),
]


def run(mutation):
    mid, desc, rel, old, new, tests = mutation
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns(".git", "__pycache__", "historical"))
    target = WORK / rel
    text = target.read_text()
    if old not in text:
        return mid, desc, "NOT-APPLIED (anchor missing)"
    target.write_text(text.replace(old, new, 1))
    proc = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                          cwd=WORK, capture_output=True, text=True)
    status = "KILLED" if proc.returncode != 0 else "SURVIVED"
    summary = [line for line in proc.stdout.splitlines() if "passed" in line or "failed" in line]
    return mid, desc, f"{status} ({summary[-1] if summary else proc.stdout[-200:]})"


if __name__ == "__main__":
    results = [run(m) for m in MUTATIONS]
    for mid, desc, status in results:
        print(f"{mid:4s} {status:70s} {desc}")
    killed = sum(1 for _, _, s in results if s.startswith("KILLED"))
    print(f"\n{killed}/{len(results)} mutations killed")
    sys.exit(0 if killed == len(results) else 1)
