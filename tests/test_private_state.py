"""The private raw-state boundary: an allow-list, in both directions.

The ESPN row-level ledger and the pre-kickoff captures have to survive between
weekly runs, and they may not be published. They previously travelled in an
Actions cache, which is wrong as a security boundary: GitHub documents that
pull-request workflows can restore default-branch caches and says not to store
sensitive material there.

They travel through a separate private repository instead, and this module is
the only thing that copies between it and the worktree. Everything here is a
property of that copy: which paths it will touch, what it refuses outright, and
what it does when the store is not there. The last one matters most — an
unavailable private store must degrade to "no raw state this run", never to a
cache, a release, an artifact or Pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import private_state  # noqa: E402

LEDGER = "data/espn_comparison_ledger.json"
SNAPSHOT = "data/espn_snapshots/2026-w01.json"


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def ledger_text() -> str:
    from nflvalue.fantasy import espn_compare

    ledger = espn_compare.new_ledger(2026)
    ledger["weeks"]["1"] = {
        "rows": [{"player_id": "00-0011", "player_name": "R. Bell", "position": "RB",
                  "espn_pts": 12.0, "model_pts": 13.0}],
        "sources": [], "grading": None,
    }
    ledger["weeks"]["1"]["projections_sha256"] = espn_compare._rows_sha256(
        ledger["weeks"]["1"]["rows"])
    return json.dumps(ledger, indent=2, sort_keys=True)


def snapshot_text() -> str:
    return json.dumps({"players": [{"espn_id": 1, "full_name": "R. Bell", "points": 12.0}],
                       "retrieved_at": "2026-09-09T12:00:00+00:00",
                       "players_sha256": "a" * 64}, sort_keys=True)


def populated_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    write(store, LEDGER, ledger_text())
    write(store, SNAPSHOT, snapshot_text())
    return store


# --------------------------------------------------------------------------- #
# B1. Only the allow-listed paths move, in either direction
# --------------------------------------------------------------------------- #
def test_restore_copies_only_the_allow_listed_paths(tmp_path):
    store, worktree = populated_store(tmp_path), tmp_path / "wt"
    write(store, "README.md", "the private state repository")
    write(store, "data/espn_league/1111111111.json", '{"schema_version": "espn-league/1"}')
    write(store, "private/my_team.html", "<html></html>")
    write(store, "secrets.txt", "nothing to see")

    report = private_state.restore(store, worktree)

    assert report.available is True
    assert sorted(report.copied) == sorted([LEDGER, SNAPSHOT])
    assert (worktree / LEDGER).read_text() == (store / LEDGER).read_text()
    assert not (worktree / "README.md").exists()
    assert not (worktree / "data" / "espn_league").exists()
    assert not (worktree / "private").exists()
    assert not (worktree / "secrets.txt").exists()
    # Nothing outside the allow-list is copied -- and nothing outside it is
    # NAMED either. A skipped path is a filename, this runs on a public
    # repository, and `espn-league-<league id>-…json` is a filename that leaks
    # the league id all by itself.
    named = {path for path, _ in report.skipped} | set(report.copied)
    assert not any(path.startswith(("README", "private/", "secrets", "data/espn_league"))
                   for path in named), named


def test_save_copies_only_the_allow_listed_paths(tmp_path):
    worktree, store = populated_store(tmp_path), tmp_path / "store-out"
    store.mkdir()
    write(worktree, "data/fantasy_latest.json", '{"my_team": {}}')
    write(worktree, "private/my_team_latest.json", '{"schema_version": "decision-card/1"}')
    write(worktree, "fantasy.html", "<html></html>")

    report = private_state.save(worktree, store)

    assert sorted(report.copied) == sorted([LEDGER, SNAPSHOT])
    assert not (store / "data" / "fantasy_latest.json").exists()
    assert not (store / "private").exists()
    assert not (store / "fantasy.html").exists()


def test_the_allow_list_is_exactly_the_two_raw_paths():
    assert private_state.ALLOWED_FILES == ("data/espn_comparison_ledger.json",)
    assert private_state.ALLOWED_TREES == ("data/espn_snapshots",)


# --------------------------------------------------------------------------- #
# B2. Symlinks and traversal are refused outright, not skipped
# --------------------------------------------------------------------------- #
def test_a_symlinked_allow_listed_file_is_refused(tmp_path):
    store, worktree = tmp_path / "store", tmp_path / "wt"
    write(store, SNAPSHOT, snapshot_text())
    outside = write(tmp_path, "elsewhere/secret.json", '{"secret": true}')
    (store / "data").mkdir(parents=True, exist_ok=True)
    (store / LEDGER).parent.mkdir(parents=True, exist_ok=True)
    (store / LEDGER).symlink_to(outside)

    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.restore(store, worktree)
    assert "symlink" in str(caught.value)
    assert not (worktree / LEDGER).exists()


def test_a_symlinked_directory_on_the_path_is_refused(tmp_path):
    store, worktree = tmp_path / "store", tmp_path / "wt"
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "2026-w01.json").write_text(snapshot_text())
    (store / "data").mkdir(parents=True)
    (store / "data" / "espn_snapshots").symlink_to(real, target_is_directory=True)

    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.restore(store, worktree)
    assert "symlink" in str(caught.value)


def test_a_path_that_escapes_its_root_is_refused(tmp_path):
    store, worktree = populated_store(tmp_path), tmp_path / "wt"
    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.copy_one(store, worktree, "data/../../escape.json")
    assert "outside" in str(caught.value)
    assert not (tmp_path / "escape.json").exists()


def test_a_traversal_relative_path_is_not_allow_listed():
    assert not private_state._is_allowed("data/espn_snapshots/../../../etc/passwd")
    assert not private_state._is_allowed("../data/espn_comparison_ledger.json")
    assert not private_state._is_allowed("/data/espn_comparison_ledger.json")


# --------------------------------------------------------------------------- #
# B3. Content refusals, hash integrity, and a guard that does not itself leak
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload, expected", [
    ('{"schema_version": "espn-league/1", "rosters": {}}', "league snapshot"),
    ('{"schema_version": "decision-card/1"}', "personalised"),
    ('{"schema_version": "my_team/1.0.0"}', "personalised"),
    ('{"members": [{"id": "{AAAAAAAA-1111-4000-8000-000000000001}"}]}', "credential"),
    ('{"cookie": "espn_s2=ABCDEF; SWID={AAAAAAAA-1111-4000-8000-000000000001}"}',
     "credential"),
])
def test_a_capture_carrying_private_material_is_refused(tmp_path, payload, expected):
    store, worktree = tmp_path / "store", tmp_path / "wt"
    write(store, SNAPSHOT, payload)
    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.restore(store, worktree)
    assert expected in str(caught.value)
    assert not (worktree / SNAPSHOT).exists()


def test_a_refusal_names_the_path_and_never_the_value(tmp_path):
    """The values on the private side of this boundary do not reach a log."""
    store, worktree = tmp_path / "store", tmp_path / "wt"
    secret = "espn_s2=SUPERSECRETCOOKIEVALUE"
    write(store, SNAPSHOT, json.dumps({"cookie": secret}))
    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.restore(store, worktree)
    message = str(caught.value)
    assert SNAPSHOT in message
    assert "SUPERSECRETCOOKIEVALUE" not in message


def test_an_ordinary_capture_is_not_refused(tmp_path):
    """The guard must not reject the real thing: captures name players."""
    store, worktree = populated_store(tmp_path), tmp_path / "wt"
    report = private_state.restore(store, worktree)
    assert SNAPSHOT in report.copied


def test_a_restored_ledger_is_verified_and_a_malformed_one_fails_closed(tmp_path):
    """A row edited in the private store must not be graded against."""
    store, worktree = populated_store(tmp_path), tmp_path / "wt"
    tampered = json.loads((store / LEDGER).read_text())
    tampered["weeks"]["1"]["rows"][0]["espn_pts"] = 99.0
    (store / LEDGER).write_text(json.dumps(tampered, indent=2, sort_keys=True))

    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.restore(store, worktree)
    assert "hash" in str(caught.value).lower()


def test_a_ledger_that_is_not_json_fails_closed(tmp_path):
    store, worktree = tmp_path / "store", tmp_path / "wt"
    write(store, LEDGER, "{not json")
    with pytest.raises(private_state.PrivateStateRejected):
        private_state.restore(store, worktree)


def test_the_module_never_prints(tmp_path, capsys):
    store, worktree = populated_store(tmp_path), tmp_path / "wt"
    private_state.restore(store, worktree)
    private_state.save(worktree, store)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


# --------------------------------------------------------------------------- #
# B4. An unavailable store degrades. It never falls back to a public place.
# --------------------------------------------------------------------------- #
def test_a_missing_store_degrades_rather_than_raising(tmp_path):
    report = private_state.restore(tmp_path / "absent", tmp_path / "wt")
    assert report.available is False
    assert report.copied == ()
    assert report.warnings and "not available" in report.warnings[0]
    assert not (tmp_path / "wt").exists()


def test_a_missing_store_on_save_degrades_rather_than_raising(tmp_path):
    worktree = populated_store(tmp_path)
    report = private_state.save(worktree, tmp_path / "absent")
    assert report.available is False
    assert report.copied == ()
    assert report.warnings


def test_a_failed_restore_leaves_the_projection_inputs_untouched(tmp_path):
    """No public fallback: the run continues with no raw state, not with a copy."""
    worktree = tmp_path / "wt"
    write(worktree, "data/fantasy_public.json", '{"players": []}')
    write(worktree, "fantasy.html", "<html></html>")
    before = {path.name: path.read_text() for path in worktree.rglob("*") if path.is_file()}

    report = private_state.restore(tmp_path / "absent", worktree)

    assert report.available is False
    after = {path.name: path.read_text() for path in worktree.rglob("*") if path.is_file()}
    assert after == before
    assert not (worktree / LEDGER).exists()


def test_the_boundary_knows_nothing_about_any_public_destination():
    """No public destination is reachable from this module's executable code.

    Scanned over the AST rather than the raw text, because the docstrings here
    deliberately explain *why* the Actions cache is not the boundary. A test
    that forbade naming it would forbid documenting the reason. What must not
    exist is a path literal a fallback could be written against.
    """
    import ast

    tree = ast.parse((ROOT / "nflvalue" / "fantasy" / "private_state.py").read_text())
    # Identity, not equality: `ast.get_docstring` cleans and dedents, so the
    # cleaned text never equals the raw Constant it came from.
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstring_nodes.add(id(body[0].value))
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstring_nodes]
    for text in literals:
        for public in ("actions/cache", "upload-artifact", "_site", "gh release",
                       "fantasy_public", "fantasy.html", "pages"):
            assert public not in text.lower(), (public, text)


# --------------------------------------------------------------------------- #
# B4b. The CLI the workflow calls
# --------------------------------------------------------------------------- #
def test_cli_restore_of_a_missing_store_warns_and_succeeds(tmp_path, capsys):
    code = private_state.main(["restore", "--store", str(tmp_path / "absent"),
                               "--worktree", str(tmp_path / "wt")])
    out = capsys.readouterr().out
    assert code == 0
    assert "::warning::" in out
    assert "::error::" not in out


def test_cli_restore_copies_and_reports_paths_only(tmp_path, capsys):
    store, worktree = populated_store(tmp_path), tmp_path / "wt"
    code = private_state.main(["restore", "--store", str(store), "--worktree", str(worktree)])
    out = capsys.readouterr().out
    assert code == 0
    assert LEDGER in out
    assert "R. Bell" not in out and "espn_pts" not in out


def test_cli_fails_loudly_on_a_security_refusal(tmp_path, capsys):
    store, worktree = tmp_path / "store", tmp_path / "wt"
    write(store, SNAPSHOT, '{"schema_version": "espn-league/1"}')
    code = private_state.main(["restore", "--store", str(store), "--worktree", str(worktree)])
    out = capsys.readouterr().out
    assert code != 0
    assert "::error::" in out
    assert not (worktree / SNAPSHOT).exists()


def test_cli_save_of_a_missing_store_warns_and_succeeds(tmp_path, capsys):
    worktree = populated_store(tmp_path)
    code = private_state.main(["save", "--store", str(tmp_path / "absent"),
                               "--worktree", str(worktree)])
    assert code == 0
    assert "::warning::" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# B5. The log is world-readable; the paths in it are the leak
# --------------------------------------------------------------------------- #
def test_the_cli_never_prints_a_path_outside_the_allow_list(tmp_path, capsys):
    """`--worktree .` enumerates the whole runner workspace on a public repo.

    Filenames are data: a league snapshot is named `espn-league-<league id>-…`,
    and the league id is the exact needle the public boundary refuses. Counting
    them is enough; naming them publishes them.
    """
    worktree, store = populated_store(tmp_path), tmp_path / "store-out"
    store.mkdir()
    write(worktree, "data/espn_league/espn-league-1111111111-2026-0101.json", "{}")
    write(worktree, "private/my_team_latest.json", "{}")
    write(worktree, "notes/league-roster-notes.md", "x")

    code = private_state.main(["save", "--store", str(store), "--worktree", str(worktree)])
    out = capsys.readouterr().out

    assert code == 0
    assert "1111111111" not in out
    assert "my_team_latest" not in out
    assert "league-roster-notes" not in out
    assert LEDGER in out            # what DID cross is still named


def test_the_walk_does_not_enumerate_the_whole_worktree(tmp_path):
    worktree, store = populated_store(tmp_path), tmp_path / "store-out"
    store.mkdir()
    for index in range(20):
        write(worktree, f"unrelated/file{index}.txt", "x")
    report = private_state.save(worktree, store)
    assert report.copied == (LEDGER, SNAPSHOT) or sorted(report.copied) == sorted(
        [LEDGER, SNAPSHOT])
    assert all(not path.startswith("unrelated/") for path, _ in report.skipped)


def test_copy_one_refuses_a_symlink_before_resolving_it(tmp_path):
    """`Path.resolve()` dereferences, so the later is_symlink() check is dead."""
    store, worktree = tmp_path / "store", tmp_path / "wt"
    outside = write(tmp_path, "elsewhere/secret.json", '{"secret": true}')
    (store / "data").mkdir(parents=True)
    (store / LEDGER).symlink_to(outside)

    with pytest.raises(private_state.PrivateStateRejected) as caught:
        private_state.copy_one(store, worktree, LEDGER)
    assert "symlink" in str(caught.value)
    assert not (worktree / LEDGER).exists()
