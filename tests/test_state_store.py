import io
import tarfile

import pytest

from scripts import state_store


def test_state_round_trip_is_checksummed_and_declared(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "model.db").write_bytes(b"database-v1")
    (root / "data" / "latest.json").write_text('{"week": 3}')
    (root / "data" / "untracked.txt").write_text("do not archive")
    archive = tmp_path / "state.tar.gz"
    manifest = state_store.pack(archive, root)

    (root / "data" / "model.db").write_bytes(b"corrupt-local")
    restored = state_store.restore(archive, manifest["sha256"], root)
    assert (root / "data" / "model.db").read_bytes() == b"database-v1"
    assert set(restored) == {"data/latest.json", "data/model.db"}
    assert "data/untracked.txt" not in manifest["files"]


def test_restore_rejects_checksum_mismatch(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "model.db").write_bytes(b"db")
    archive = tmp_path / "state.tar.gz"
    state_store.pack(archive, root)
    with pytest.raises(ValueError, match="checksum mismatch"):
        state_store.restore(archive, "0" * 64, root)


def test_restore_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    payload = b"owned"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../outside.db")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe member"):
        state_store.restore(archive, state_store.sha256(archive), tmp_path / "repo")


def test_fantasy_state_profile_excludes_prop_state(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "fantasy_model.joblib").write_bytes(b"fantasy")
    (root / "data" / "player_projection_snapshot.json").write_text('{"schema_version": 1}')
    (root / "data" / "nfl_props.db").write_bytes(b"props")
    (root / "data" / "latest.json").write_text('{"betting": true}')
    archive = tmp_path / "fantasy-state.tar.gz"
    manifest = state_store.pack(archive, root, profile="fantasy")
    assert manifest["profile"] == "fantasy"
    assert manifest["files"] == [
        "data/fantasy_model.joblib",
        "data/player_projection_snapshot.json",
    ]
    restored_root = tmp_path / "restored"
    restored = state_store.restore(
        archive, manifest["sha256"], restored_root, profile="fantasy"
    )
    assert set(restored) == set(manifest["files"])
    assert not (restored_root / "data" / "nfl_props.db").exists()


def test_fantasy_state_profile_carries_public_grading_history_only(tmp_path):
    """The checksummed public archive is how the season series survives a run.

    It must carry the aggregate history — that is the whole reason the public
    series no longer depends on private state — and it must not carry the
    row-level ESPN ledger or the raw pre-kickoff captures, which are ESPN's
    per-player projections under terms that grant no redistribution right.
    A release asset on a public repository is published material.
    """
    root = tmp_path / "repo"
    (root / "data" / "espn_snapshots").mkdir(parents=True)
    (root / "data" / "fantasy_model.joblib").write_bytes(b"fantasy")
    (root / "data" / "player_projection_snapshot.json").write_text('{"schema_version": 1}')
    (root / "data" / "espn_comparison_history.json").write_text('{"kind": "history"}')
    (root / "data" / "espn_comparison_ledger.json").write_text('{"kind": "ledger"}')
    (root / "data" / "espn_snapshots" / "2026-01.json").write_text('{"players": []}')

    archive = tmp_path / "fantasy-state.tar.gz"
    manifest = state_store.pack(archive, root, profile="fantasy")

    assert "data/espn_comparison_history.json" in manifest["files"]
    assert "data/espn_comparison_ledger.json" not in manifest["files"]
    assert not any(name.startswith("data/espn_snapshots/") for name in manifest["files"])

    restored_root = tmp_path / "restored"
    state_store.restore(archive, manifest["sha256"], restored_root, profile="fantasy")
    assert (restored_root / "data" / "espn_comparison_history.json").exists()
    assert not (restored_root / "data" / "espn_comparison_ledger.json").exists()


def test_no_state_profile_names_a_raw_espn_path():
    for profile, patterns in state_store.STATE_PROFILES.items():
        for pattern in patterns:
            assert "espn_snapshots" not in pattern, (profile, pattern)
            assert "espn_comparison_ledger" not in pattern, (profile, pattern)
