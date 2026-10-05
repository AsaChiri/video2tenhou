# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Review rebuilds cannot mix stale sparse evidence with current dense models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

import pytest

from tests.recognition import RecognitionStub
from video2tenhou import observe, read
from video2tenhou.engine import decode
from video2tenhou.files import sha256_text
from video2tenhou.layout import Calibration
from video2tenhou.perception import classifier, detector, evidence_policy
from video2tenhou.perception.evidence_policy import EvidencePolicy
from video2tenhou.record import Game, HandResult


def test_engine_imports_load_no_inference_or_network_stack() -> None:
    """Review rebuild children import torch only if a hand rereads the video."""
    code = (
        "import sys, video2tenhou.cli, video2tenhou.engine.decode, "
        "video2tenhou.export; "
        "print(sorted({'torch', 'libreyolo', 'httpx'} & set(sys.modules)))"
    )
    result = subprocess.run(  # noqa: S603  fixed interpreter and source
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


@pytest.fixture(autouse=True)
def _isolated_policy_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evidence_policy, "load_policy", lambda: evidence_policy.DEFAULT_POLICY
    )


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"unchanged source")
    work = tmp_path / "work"
    det = RecognitionStub(
        id="detector:weights-and-runtime",
        evidence_policy=evidence_policy.DEFAULT_POLICY,
    )
    clf = RecognitionStub(id="classifier:weights-and-runtime", classes=["1m"], T=1.0)
    cal = Calibration.load("pml")
    hands: list[dict] = [
        {
            "hand": i,
            "t_start": i * 10,
            "t_end": (i + 1) * 10,
            "game": 0,
            "site_index": i,
            "kyoku": i,
            "honba": 0,
            "corner_wind": {"TL": "E"},
        }
        for i in range(2)
    ]
    for h in hands:
        directory = work / "reads" / f"{h['hand']:02d}"
        directory.mkdir(parents=True)
        manifest = {
            "detector": det.id,
            "classifier": clf.id,
            "identity": read._read_identity(source, clf),
            "geometry": {r: read.region_key(cal, r) for r in read.REGIONS},
            "window": [h["t_start"], h["t_end"] + 0.5],
        }
        (directory / "done.json").write_text(json.dumps(manifest))
        (work / "obs").mkdir(exist_ok=True)
        (work / "obs" / f"{h['hand']:02d}.json").write_text("{}")
        (work / "decode").mkdir(exist_ok=True)
        (work / "decode" / f"{h['hand']:02d}.json").write_text('{"previous":true}')
    monkeypatch.setattr(detector, "LazyDetector", lambda: det)
    monkeypatch.setattr(classifier, "LazyClassifier", lambda: clf)
    monkeypatch.setattr(decode, "load_facts", lambda _path: [])
    return source, work, cal, hands, det, clf


def _vote_empty_readings(
    work: Path, hands: list[dict], policy: EvidencePolicy | None = None
) -> None:
    """Publish an empty reading of every region and vote it, as analysis does."""
    (work / "calm.jsonl").write_text("")
    for hand in hands:
        directory = work / "reads" / f"{hand['hand']:02d}"
        for region in read.REGIONS:
            (directory / f"{region.replace(':', '_')}.jsonl").write_text("")
    observe.run_observe(work, hands, [], force=True, policy=policy)


@pytest.mark.parametrize(
    "change", ["model_runtime", "replaced_source", "geometry", "legacy"]
)
def test_rebuild_rejects_stale_evidence_before_any_hand_is_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands)
    path = work / "reads" / "01" / "done.json"
    manifest = json.loads(path.read_text())
    if change == "model_runtime":
        manifest["detector"] = "same-weights-previous-runtime"
    elif change == "geometry":
        manifest["geometry"]["hand:BR"] = "previous-crop"
    elif change == "legacy":
        manifest.pop("identity")
    else:
        stat = source.stat()
        replacement = source.with_suffix(".new")
        replacement.write_bytes(b"replacement data")  # same byte count and timestamp
        assert replacement.stat().st_size == stat.st_size
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(source)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        decode,
        "decode_hand",
        lambda *_unused_a, **_unused_k: pytest.fail("stale batch reached decoding"),
    )
    with pytest.raises(ValueError, match="Analyze recording"):
        decode.run_decode(
            work,
            hands,
            [],
            force=True,
            video_path=source,
            cal=cal,
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_read_identity_guard_checks_only_selected_hands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / "reads" / "01" / "done.json").unlink()
    read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands[:1])
    with pytest.raises(ValueError, match=r"hand\(s\) 1"):
        read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands)


@pytest.mark.parametrize("component", ["LazyDetector", "LazyClassifier"])
def test_model_loading_failure_preserves_existing_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    source, work, cal, hands, _, _ = _inputs(tmp_path, monkeypatch)

    def broken() -> None:
        raise RuntimeError("model loading failed")

    monkeypatch.setattr(
        detector if component == "LazyDetector" else classifier, component, broken
    )
    monkeypatch.setattr(
        decode,
        "decode_hand",
        lambda *_unused_a, **_unused_k: pytest.fail(
            "failed model became offline decoding"
        ),
    )
    with pytest.raises(RuntimeError, match="model loading failed"):
        decode.run_decode(
            work,
            hands,
            [],
            force=True,
            video_path=source,
            cal=cal,
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_explicit_missing_video_cannot_become_offline_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, cal, hands, _, _ = _inputs(tmp_path, monkeypatch)
    source.unlink()
    with pytest.raises(FileNotFoundError):
        decode.run_decode(work, hands, [], video_path=source, cal=cal)


def test_decode_cache_io_failure_is_not_a_cache_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, cal, hands, _, _ = _inputs(tmp_path, monkeypatch)
    read_text = Path.read_text

    def denied(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == work / "decode" / "00.json":
            raise PermissionError("decode is unreadable")
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="decode is unreadable"):
        decode.run_decode(work, hands, [], video_path=source, cal=cal)


def test_rebuild_rejects_unproven_observations_despite_matching_read_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / "calm.jsonl").write_text("")
    read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands)
    monkeypatch.setattr(
        decode,
        "decode_hand",
        lambda *_unused_a, **_unused_k: pytest.fail(
            "unproven observation reached decoding"
        ),
    )
    with pytest.raises(ValueError, match="Analyze recording"):
        decode.run_decode(
            work,
            hands,
            [],
            force=True,
            video_path=source,
            cal=cal,
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_rebuild_accepts_current_observations_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    _vote_empty_readings(work, hands)
    seen = []

    def rebuild(
        entry: dict,
        obs: dict,
        result: HandResult,
        facts: dict,
        *,
        options: decode.DecodeOptions,
        **_unused_kwargs: object,
    ) -> dict:
        assert options.models is not None
        assert options.models[:2] == (det, clf)
        seen.append(entry["hand"])
        return {
            "stats": {"turns": 0, "calls": 0},
            "dora": [],
            "solver": {"status": "ok", "objective": 0, "low_margin": []},
            "score": None,
            "result": {"han": 0, "fu": 0},
        }

    monkeypatch.setattr(decode, "decode_hand", rebuild)
    monkeypatch.setattr(decode, "facts_for_hand", lambda *_unused_args: {})
    result = decode.run_decode(
        work,
        hands,
        [Game(1, {}, {}, [HandResult(i, 0, 0, {}, "draw") for i in range(2)])],
        force=True,
        video_path=source,
        cal=cal,
    )
    assert seen == [0, 1]
    assert len(result) == 2


def test_decode_resume_tracks_observation_and_provenance_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, hands, games, seen = _decode_resume_inputs(tmp_path, monkeypatch)
    # offline reconstruction remains supported
    first = decode.run_decode(work, hands, games)[0]
    assert len(seen) == 1
    assert decode.run_decode(work, hands, games)[0] == first
    assert len(seen) == 1

    # A completed observation publication followed by a restart returns no
    # changed hands. Identical votes of new readings still bind a new record.
    path = work / "reads" / "00" / "done.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "detector": "re-read by a new model"}))
    observe.run_observe(work, hands, [])
    assert observe.run_observe(work, hands, [])["changed_hands"] == []
    second = decode.run_decode(work, hands, games)[0]
    assert len(seen) == 2
    assert (
        first["decode_inputs"]["observations"]
        != second["decode_inputs"]["observations"]
    )
    assert json.loads((work / "obs/provenance/00.json").read_text())[
        "output_sha256"
    ] == sha256_text((work / "obs/00.json").read_text())

    (work / "obs/provenance/00.json").unlink()  # an interrupted vote publication
    decode.run_decode(work, hands, games)
    assert len(seen) == 3  # unproven votes never reuse a bound result
    observe.run_observe(work, hands, [])
    decode.run_decode(work, hands, games)
    assert len(seen) == 4
    (work / "decode" / "00.json").write_text("{truncated")
    decode.run_decode(work, hands, games)
    assert len(seen) == 5  # incomplete output is a cache miss


def test_decode_rebuilds_for_changed_answers_records_and_seats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Human answers, site records and seat maps bind a decode; dismissals don't."""
    work, hands, games, seen = _decode_resume_inputs(tmp_path, monkeypatch)
    decode.run_decode(work, hands, games)
    facts = [{"game": 0, "kyoku": 0, "honba": 0, "kind": "draw", "tile": "1m"}]
    monkeypatch.setattr(decode, "load_facts", lambda _path: facts)
    seen.clear()
    decode.run_decode(work, hands, games)
    assert seen == [1]  # new human answers invalidate a completed decode
    seen.clear()
    facts[0]["tile"] = "2m"
    decode.run_decode(work, hands, games)
    assert seen == [1]
    seen.clear()
    facts.clear()
    decode.run_decode(work, hands, games)
    assert seen == [1]  # deleting an answer also changes the constraints
    seen.clear()
    games[0].hands[0].sticks = 1
    decode.run_decode(work, hands, games)
    assert seen == [1]  # updated score records cannot reuse the previous hand
    seen.clear()
    hands[0]["corner_wind"] = {"TL": "S"}
    decode.run_decode(work, hands, games)
    assert seen == [1]
    seen.clear()
    facts.append({"game": 1, "kyoku": 0, "honba": 0, "kind": "draw", "tile": "1m"})
    decode.run_decode(work, hands, games)
    assert seen == []  # unrelated hand answers do not invalidate this hand
    facts.append({"game": 0, "kyoku": 0, "honba": 0, "kind": "dismiss", "item": "x"})
    decode.run_decode(work, hands, games)
    assert seen == []  # dismissing a question is review state, not an input


def test_interrupted_decode_publication_preserves_complete_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "00.json"
    path.write_text('{"complete":true}')

    def interrupted_dump(
        value: object, stream: TextIO, **_unused_kwargs: object
    ) -> None:
        stream.write('{"partial":')
        raise OSError("interrupted")

    monkeypatch.setattr(decode.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="interrupted"):
        decode.atomic_write_json(path, {})
    assert path.read_text() == '{"complete":true}'
    assert not list(tmp_path.glob(".*.tmp"))


def _policy(sparse_hand: float = 0.2, dense_hand: float = 0.2) -> EvidencePolicy:
    return evidence_policy.resolve_policy(
        {
            "schema_version": 1,
            "sparse": {"hand": sparse_hand, "pond": 0.2, "meld": 0.35},
            "dense": {"hand": dense_hand, "pond": 0.2, "meld": 0.2},
        }
    )


def test_review_rebuild_rejects_old_sparse_policy_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    _vote_empty_readings(work, hands, evidence_policy.DEFAULT_POLICY)
    det.evidence_policy = _policy(sparse_hand=0.1)
    monkeypatch.setattr(evidence_policy, "load_policy", lambda: det.evidence_policy)
    monkeypatch.setattr(
        decode,
        "decode_hand",
        lambda *_unused_a, **_unused_k: pytest.fail("stale votes reached decoding"),
    )
    # Raw recognition remains reusable; only the observation policy is stale.
    read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands)
    with pytest.raises(ValueError, match="Analyze recording"):
        decode.run_decode(
            work,
            hands,
            [],
            force=True,
            video_path=source,
            cal=cal,
            evidence_policy=det.evidence_policy,
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


@pytest.mark.parametrize("failure", ["invalid_metadata", "policy_changed"])
def test_policy_failures_cannot_become_optional_model_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source, work, cal, hands, _det, _clf = _inputs(tmp_path, monkeypatch)
    if failure == "invalid_metadata":

        def invalid() -> None:
            raise ValueError("Invalid evidence metadata")

        monkeypatch.setattr(evidence_policy, "load_policy", invalid)
    else:
        monkeypatch.setattr(
            evidence_policy, "load_policy", lambda: _policy(dense_hand=0.1)
        )
    monkeypatch.setattr(
        detector,
        "LazyDetector",
        lambda: pytest.fail("policy preflight must precede model loading"),
    )
    with pytest.raises(ValueError, match=r"Invalid evidence metadata|policy changed"):
        decode.run_decode(
            work,
            hands,
            [],
            force=True,
            video_path=source,
            cal=cal,
            evidence_policy=evidence_policy.DEFAULT_POLICY,
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_dense_only_policy_invalidates_decode_but_preserves_observation_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, work, _cal, hands, _det, _clf = _inputs(tmp_path, monkeypatch)
    hands = hands[:1]
    monkeypatch.setattr(decode, "DATA_DIR", tmp_path)
    monkeypatch.setattr(decode, "facts_for_hand", lambda *_unused_args: {})
    _vote_empty_readings(work, hands, evidence_policy.DEFAULT_POLICY)
    paths = [work / "obs/00.json", work / "obs/provenance/00.json"]
    before = [path.read_bytes() for path in paths]
    calls = []

    def rebuild(*_unused_args: object, **_unused_kwargs: object) -> dict:
        calls.append(1)
        return {
            "decoder_version": decode.DECODER_VERSION,
            "stats": {"turns": 0, "calls": 0},
            "dora": [],
            "solver": {"status": "optimal", "objective": 0, "low_margin": []},
            "score": None,
            "result": {"han": 0, "fu": 0},
        }

    monkeypatch.setattr(decode, "decode_hand", rebuild)
    games = [Game(1, {}, {}, [HandResult(0, 0, 0, {}, "draw")])]
    first = decode.run_decode(
        work,
        hands,
        games,
        evidence_policy=evidence_policy.DEFAULT_POLICY,
    )[0]
    policy = _policy(dense_hand=0.1)
    observe.validate_observation_cache(work, hands, policy=policy)
    second = decode.run_decode(work, hands, games, evidence_policy=policy)[0]
    assert len(calls) == 2
    assert [path.read_bytes() for path in paths] == before
    assert (
        first["decode_inputs"]["dense_policy"]
        != second["decode_inputs"]["dense_policy"]
    )
    assert (
        first["decode_inputs"]["observations"]
        == second["decode_inputs"]["observations"]
    )
    # Omitting the explicit policy uses metadata only, even for nondefault policies.
    monkeypatch.setattr(evidence_policy, "load_policy", lambda: policy)
    monkeypatch.setattr(
        detector,
        "LazyDetector",
        lambda: pytest.fail("valid bound cache must not load models"),
    )
    assert decode.run_decode(work, hands, games, video_path=source)[0] == second
    assert len(calls) == 2


def _decode_resume_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Publish empty but valid evidence and record reconstruction attempts."""
    _source, work, _cal, hands, _det, _clf = _inputs(tmp_path, monkeypatch)
    hands = hands[:1]
    monkeypatch.setattr(decode, "DATA_DIR", tmp_path)
    monkeypatch.setattr(decode, "facts_for_hand", lambda *_unused_args: {})
    _vote_empty_readings(work, hands)
    seen = []

    def rebuild(*_unused_args: object, **_unused_kwargs: object) -> dict:
        seen.append(1)
        return {
            "decoder_version": decode.DECODER_VERSION,
            "stats": {"turns": 0, "calls": 0},
            "dora": [],
            "solver": {"status": "ok", "objective": 0, "low_margin": []},
            "score": None,
            "result": {"han": 0, "fu": 0},
        }

    monkeypatch.setattr(decode, "decode_hand", rebuild)
    games = [Game(1, {}, {}, [HandResult(0, 0, 0, {}, "draw")])]
    return work, hands, games, seen
