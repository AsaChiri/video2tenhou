# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Resume and interruption boundaries for real sparse readings and voted evidence."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from video2tenhou import observe as stage
from video2tenhou.calm import Interval
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY, EvidencePolicy
from video2tenhou.train.data import CLASSES

if TYPE_CHECKING:
    from typing import TextIO


def _write_reads(path: "Path", tile: str = "1m") -> None:
    posterior = [0.0] * len(CLASSES)
    posterior[CLASSES.index(tile)] = 1.0
    rows = [
        {
            "t": t,
            "region": "pond:TL",
            "size": [400, 700],
            "boxes": [
                {
                    "xyxy": [10, 10, 50, 70],
                    "conf": 0.9,
                    "sideways": False,
                    "p": posterior,
                }
            ],
        }
        for t in (0.0, 1.0, 2.0)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def cache(tmp_path: "Path") -> "SimpleNamespace":
    """Build a complete sparse-reading and voted-observation cache fixture."""
    hdir = tmp_path / "reads/00"
    hdir.mkdir(parents=True)
    reads = hdir / "pond_TL.jsonl"
    _write_reads(reads)
    done = hdir / "done.json"
    done.write_text('{"identity":"original"}')
    ivs = [Interval("pond:TL", 0.0, 2.0, 3, calm=True, motion=0.0, skin=0.0)]
    (tmp_path / "calm.jsonl").write_text(json.dumps(ivs[0].to_dict()) + "\n")
    hands = [{"hand": 0, "t_start": 0.0, "t_end": 2.0}]
    result = SimpleNamespace(
        work=tmp_path,
        hands=hands,
        ivs=ivs,
        reads=reads,
        done=done,
        output=tmp_path / "obs/00.json",
        proof=tmp_path / "obs/provenance/00.json",
    )
    assert stage.run_observe(tmp_path, hands, ivs, log=lambda _: None) == {
        "observations": 1,
        "hands": 1,
        "changed_hands": [0],
    }
    stage.validate_observation_cache(tmp_path, hands)
    return result


def _run(
    cache: "SimpleNamespace",
    *,
    force: bool = False,
    touched: set[int] | None = None,
    policy: EvidencePolicy | dict | None = None,
) -> dict:
    policy = policy if policy is not None else getattr(cache, "policy", None)
    return stage.run_observe(
        cache.work,
        cache.hands,
        cache.ivs,
        log=lambda _: None,
        options=stage.ObservationOptions(force=force, touched=touched, policy=policy),
    )


def _votes(cache: "SimpleNamespace") -> "list[dict]":
    return stage.load_obs(cache.work, 0)["pond:TL"]


def test_resume_reuses_intact_votes_but_detects_new_reads_with_empty_touched(
    cache: "SimpleNamespace", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify resume reuses intact votes but detects new reads with empty touched."""
    original = cache.output.read_bytes()
    original_vote = stage.observe_interval

    def unexpected(*_: object) -> None:
        msg = "An intact cache should not vote again"
        raise AssertionError(msg)

    monkeypatch.setattr(stage, "observe_interval", unexpected)
    assert _run(cache, touched=set()) == {"changed_hands": []}
    assert cache.output.read_bytes() == original
    monkeypatch.setattr(stage, "observe_interval", original_vote)
    _write_reads(
        cache.reads, "2p"
    )  # Simulate prior process stopping after sparse-read publication.
    with pytest.raises(ValueError, match="Analyze recording"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache, touched=set())["changed_hands"] == [0]
    assert _votes(cache)[0]["slots"][0]["tile"] == "2p"
    stage.validate_observation_cache(cache.work, cache.hands)


@pytest.mark.parametrize(
    "change", ["interval", "partial", "settings", "completion", "window"]
)
def test_current_inputs_control_reuse_without_touched_hint(
    cache: "SimpleNamespace", change: str
) -> None:
    """Verify current inputs control reuse without touched hint."""
    before = cache.proof.read_bytes()
    if change == "interval":
        cache.ivs = [replace(cache.ivs[0], t0=1.0)]
    elif change == "partial":
        cache.ivs = [replace(cache.ivs[0], partial=True)]
    elif change == "settings":
        cache.policy = DEFAULT_POLICY.to_dict()
        cache.policy["sparse"]["pond"] = 0.95
    elif change == "completion":
        cache.done.write_text('{"identity":"new recognition runtime"}')
    else:
        cache.hands = [{"hand": 0, "t_start": 0.5, "t_end": 2.0}]
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(
            cache.work, cache.hands, cache.ivs, policy=getattr(cache, "policy", None)
        )
    assert _run(cache, touched=set())["hands"] == 1
    assert cache.proof.read_bytes() != before
    stage.validate_observation_cache(
        cache.work, cache.hands, cache.ivs, policy=getattr(cache, "policy", None)
    )
    if change == "interval":
        assert _votes(cache)[0]["n_readings"] == 2
        assert _votes(cache)[0]["t0"] == 1.0
    if change == "partial":
        assert _votes(cache)[0]["partial"]
    if change == "settings":
        assert _votes(cache)[0]["slots"] == []


def test_unrelated_calm_intervals_do_not_invalidate_hand(
    cache: "SimpleNamespace",
) -> None:
    """Verify unrelated calm intervals do not invalidate hand."""
    cache.ivs += [
        Interval("pond:TL", 20.0, 21.0, 2, calm=True, motion=0.0, skin=0.0),
        Interval("pond:TL", 0.0, 2.0, 3, calm=False, motion=9.0, skin=0.0),
    ]
    assert _run(cache) == {"changed_hands": []}
    stage.validate_observation_cache(
        cache.work, cache.hands, cache.ivs, policy=getattr(cache, "policy", None)
    )


@pytest.mark.parametrize(
    "damage", ["legacy", "corrupt_output", "missing_output", "corrupt_proof"]
)
def test_unproven_or_corrupted_output_is_recomputed(
    cache: "SimpleNamespace", damage: str
) -> None:
    """Verify unproven or corrupted output is recomputed."""
    if damage == "legacy":
        cache.proof.unlink()
    elif damage == "corrupt_output":
        cache.output.write_text('{"pond:TL":')
    elif damage == "missing_output":
        cache.output.unlink()
    else:
        cache.proof.write_text("[]")
    with pytest.raises(ValueError, match="Analyze recording"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache, touched=set())["hands"] == 1
    assert _votes(cache)[0]["slots"][0]["tile"] == "1m"
    stage.validate_observation_cache(cache.work, cache.hands)


@pytest.mark.parametrize("publication", ["output", "manifest"])
def test_failed_atomic_publication_never_authenticates_changed_evidence(
    cache: "SimpleNamespace", monkeypatch: "pytest.MonkeyPatch", publication: str
) -> None:
    """Verify failed atomic publication never authenticates changed evidence."""
    before_output, before_proof = cache.output.read_bytes(), cache.proof.read_bytes()
    _write_reads(cache.reads, "2p")
    target = cache.output if publication == "output" else cache.proof
    original_replace = Path.replace

    def fail_publication(path: "Path", destination: "Path") -> "Path":
        if Path(destination) == target:
            msg = "simulated publication failure"
            raise OSError(msg)
        return original_replace(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_publication)
        with pytest.raises(OSError, match="publication failure"):
            _run(cache)
    assert cache.proof.read_bytes() == before_proof
    if publication == "output":
        assert cache.output.read_bytes() == before_output
    else:
        assert _votes(cache)[0]["slots"][0]["tile"] == "2p"
    assert not list((cache.work / "obs").rglob(".*.tmp"))
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache, touched=set())["hands"] == 1
    stage.validate_observation_cache(cache.work, cache.hands)


def test_failed_forced_output_write_preserves_previous_usable_cache(
    cache: "SimpleNamespace", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify failed forced output write preserves previous usable cache."""
    original = cache.output.read_bytes(), cache.proof.read_bytes()
    real_dump = json.dump

    def fail_write(
        value: "object", stream: "TextIO", **_unused_options: object
    ) -> None:
        stream.write('{"interrupted":')
        msg = "disk full"
        raise OSError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(stage.json, "dump", fail_write)
        with pytest.raises(OSError, match="disk full"):
            _run(cache, force=True)
    assert stage.json.dump is real_dump
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == original
    assert not list((cache.work / "obs").rglob(".*.tmp"))
    stage.validate_observation_cache(cache.work, cache.hands)


def test_inputs_changing_during_vote_abort_before_publication(
    cache: "SimpleNamespace", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify inputs changing during vote abort before publication."""
    original = cache.output.read_bytes(), cache.proof.read_bytes()
    vote = stage.observe_interval

    def changed_input(
        region: str,
        readings: list[dict],
        iv: Interval,
        *,
        policy: EvidencePolicy | dict | None = None,
    ) -> list[stage.Observation]:
        result = vote(region, readings, iv, policy=policy)
        _write_reads(cache.reads, "2p")
        return result

    monkeypatch.setattr(stage, "observe_interval", changed_input)
    with pytest.raises(ValueError, match="inputs changed"):
        _run(cache, force=True)
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == original
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands)


def test_default_validation_rejects_unreadable_saved_calm_without_modifying_votes(
    cache: "SimpleNamespace",
) -> None:
    """Verify unreadable calm data fails without changing saved votes."""
    before = cache.output.read_bytes(), cache.proof.read_bytes()
    (cache.work / "calm.jsonl").write_text("not JSON")
    with pytest.raises(ValueError, match="cannot be verified"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == before


def test_validation_does_not_misreport_invalid_caller_data_as_stale_cache(
    cache: "SimpleNamespace",
) -> None:
    """Verify validation does not misreport invalid caller data as stale cache."""
    before = cache.output.read_bytes(), cache.proof.read_bytes()
    with pytest.raises(KeyError, match="t_start"):
        stage.validate_observation_cache(cache.work, [{"hand": 0}], cache.ivs)
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == before


def test_dense_only_policy_change_reuses_sparse_observations(
    cache: "SimpleNamespace",
) -> None:
    """Verify dense only policy change reuses sparse observations."""
    policy = DEFAULT_POLICY.to_dict()
    policy["dense"]["hand"] = 0.1
    before = (
        cache.reads.read_bytes(),
        cache.output.read_bytes(),
        cache.proof.read_bytes(),
    )
    assert _run(cache, policy=policy) == {"changed_hands": []}
    stage.validate_observation_cache(cache.work, cache.hands, policy=policy)
    assert (
        cache.reads.read_bytes(),
        cache.output.read_bytes(),
        cache.proof.read_bytes(),
    ) == before


def test_sparse_policy_change_revotes_saved_bytes_and_roundtrips(
    cache: "SimpleNamespace",
) -> None:
    """Verify sparse policy change revotes saved bytes and roundtrips."""
    original = cache.reads.read_bytes(), cache.output.read_bytes()
    policy = DEFAULT_POLICY.to_dict()
    policy["sparse"]["pond"] = 0.95
    assert _run(cache, policy=policy)["changed_hands"] == [0]
    assert _votes(cache)[0]["slots"] == []
    assert cache.reads.read_bytes() == original[0]
    stage.validate_observation_cache(cache.work, cache.hands, policy=policy)
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache)["changed_hands"] == [0]
    assert cache.output.read_bytes() == original[1]
    assert cache.reads.read_bytes() == original[0]
