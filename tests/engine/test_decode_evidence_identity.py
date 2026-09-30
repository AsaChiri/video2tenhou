# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Review rebuilds cannot mix stale sparse evidence with current dense models."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.recognition import RecognitionStub
from video2tenhou import observe, read
from video2tenhou.engine import decode
from video2tenhou.layout import Calibration
from video2tenhou.perception import classifier, detector, evidence_policy
from video2tenhou.record import Game, HandResult

if TYPE_CHECKING:
    from typing import TextIO

    from video2tenhou.perception.evidence_policy import EvidencePolicy


@pytest.fixture(autouse=True)
def _isolated_policy_metadata(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setattr(
        evidence_policy, "load_policy", lambda: evidence_policy.DEFAULT_POLICY
    )


def _inputs(tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch") -> tuple:
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
    monkeypatch.setattr(detector, "Detector", lambda: det)
    monkeypatch.setattr(classifier, "Classifier", lambda: clf)
    monkeypatch.setattr(decode, "load_facts", lambda _path: [])
    return source, work, cal, hands, det, clf


@pytest.mark.parametrize(
    "change", ["model_runtime", "source_same_stat", "geometry", "legacy"]
)
def test_rebuild_rejects_stale_evidence_before_any_hand_is_overwritten(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", change: str
) -> None:
    """Verify rebuild rejects stale evidence before any hand is overwritten."""
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
        source.write_bytes(b"replacement data")  # same byte count and timestamp
        assert source.stat().st_size == stat.st_size
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
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
            options=decode.DecodeRunOptions(force=True, video_path=source, cal=cal),
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_read_identity_guard_checks_only_selected_hands(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify read identity guard checks only selected hands."""
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / "reads" / "01" / "done.json").unlink()
    read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands[:1])
    with pytest.raises(ValueError, match=r"hand\(s\) 1"):
        read.validate_read_cache(read.ReadContext(source, cal, work, det, clf), hands)


@pytest.mark.parametrize("component", ["Detector", "Classifier"])
def test_model_loading_failure_preserves_existing_decodes(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", component: str
) -> None:
    """Verify model loading failure preserves existing decodes."""
    source, work, cal, hands, _, _ = _inputs(tmp_path, monkeypatch)

    def broken() -> None:
        msg = "model loading failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        detector if component == "Detector" else classifier, component, broken
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
            options=decode.DecodeRunOptions(force=True, video_path=source, cal=cal),
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_explicit_missing_video_cannot_become_offline_reconstruction(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify explicit missing video cannot become offline reconstruction."""
    source, work, cal, hands, _, _ = _inputs(tmp_path, monkeypatch)
    source.unlink()
    with pytest.raises(FileNotFoundError):
        decode.run_decode(
            work, hands, [], options=decode.DecodeRunOptions(video_path=source, cal=cal)
        )


def test_decode_cache_io_failure_is_not_a_cache_miss(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify decode cache io failure is not a cache miss."""
    source, work, cal, hands, _, _ = _inputs(tmp_path, monkeypatch)
    read_text = Path.read_text

    def denied(
        path: "Path", encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == work / "decode" / "00.json":
            msg = "decode is unreadable"
            raise PermissionError(msg)
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="decode is unreadable"):
        decode.run_decode(
            work, hands, [], options=decode.DecodeRunOptions(video_path=source, cal=cal)
        )


def test_rebuild_rejects_unproven_observations_despite_matching_read_models(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify rebuild rejects unproven observations despite matching read models."""
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
            options=decode.DecodeRunOptions(force=True, video_path=source, cal=cal),
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_rebuild_accepts_current_observations_and_models(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify rebuild accepts current observations and models."""
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / "calm.jsonl").write_text("")
    for h in hands:
        for region in read.REGIONS:
            (
                work
                / "reads"
                / f"{h['hand']:02d}"
                / f"{region.replace(':', '_')}.jsonl"
            ).write_text("")
    observe.run_observe(work, hands, [], options=observe.ObservationOptions(force=True))
    seen = []

    def rebuild(
        entry: "dict",
        obs: "dict",
        result: "HandResult",
        facts: "dict",
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
            "problems": [],
        }

    monkeypatch.setattr(decode, "decode_hand", rebuild)
    monkeypatch.setattr(decode, "facts_for_hand", lambda *_unused_args: {})
    result = decode.run_decode(
        work,
        hands,
        [Game(1, {}, {}, [HandResult(i, 0, 0, {}, "draw") for i in range(2)])],
        options=decode.DecodeRunOptions(force=True, video_path=source, cal=cal),
    )
    assert seen == [0, 1]
    assert len(result) == 2


def test_decode_resume_tracks_observation_and_provenance_after_publication(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify decode resume tracks observation and provenance after publication."""
    work, hands, games, seen = _decode_resume_inputs(tmp_path, monkeypatch)
    first = decode.run_decode(work, hands, games)[
        0
    ]  # offline reconstruction remains supported
    assert len(seen) == 1
    assert decode.run_decode(work, hands, games)[0] == first
    assert len(seen) == 1

    # A completed observation publication followed by a restart returns no
    # changed hands. Identical votes still have different evidence provenance.
    manifest = work / "reads" / "00" / "done.json"
    manifest.write_text(manifest.read_text() + "\n")
    observe.run_observe(work, hands, [])
    assert observe.run_observe(work, hands, [])["changed_hands"] == []
    second = decode.run_decode(work, hands, games)[0]
    assert len(seen) == 2
    assert (
        first["decode_inputs"]["observation_sha256"]
        == second["decode_inputs"]["observation_sha256"]
    )
    assert (
        first["decode_inputs"]["provenance_sha256"]
        != second["decode_inputs"]["provenance_sha256"]
    )

    observation = work / "obs" / "00.json"
    observation.write_text(observation.read_text() + "\n")
    decode.run_decode(work, hands, games)
    assert len(seen) == 3  # actual observation bytes independently invalidate
    (work / "decode" / "00.json").write_text("{truncated")
    decode.run_decode(work, hands, games)
    assert len(seen) == 4  # incomplete output is a cache miss

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
    facts.append({"game": 1, "kyoku": 0, "honba": 0, "kind": "note"})
    decode.run_decode(work, hands, games)
    assert seen == []  # unrelated hand answers do not invalidate this hand


def test_interrupted_decode_publication_preserves_complete_result(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify interrupted decode publication preserves complete result."""
    path = tmp_path / "00.json"
    path.write_text('{"complete":true}')

    def interrupted_dump(
        value: "object", stream: "TextIO", **_unused_kwargs: object
    ) -> None:
        stream.write('{"partial":')
        msg = "interrupted"
        raise OSError(msg)

    monkeypatch.setattr(decode.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="interrupted"):
        decode.atomic_write_json(path, {})
    assert path.read_text() == '{"complete":true}'
    assert not list(tmp_path.glob(".*.tmp"))


def _policy(sparse_hand: float = 0.2, dense_hand: float = 0.2) -> "EvidencePolicy":
    return evidence_policy.resolve_policy(
        {
            "schema_version": 1,
            "sparse": {"hand": sparse_hand, "pond": 0.2, "meld": 0.35},
            "dense": {"hand": dense_hand, "pond": 0.2, "meld": 0.2},
        }
    )


def test_review_rebuild_rejects_old_sparse_policy_before_any_write(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify review rebuild rejects old sparse policy before any write."""
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / "calm.jsonl").write_text("")
    for hand in hands:
        for region in read.REGIONS:
            (
                work
                / "reads"
                / f"{hand['hand']:02d}"
                / f"{region.replace(':', '_')}.jsonl"
            ).write_text("")
    observe.run_observe(
        work,
        hands,
        [],
        options=observe.ObservationOptions(
            force=True, policy=evidence_policy.DEFAULT_POLICY
        ),
    )
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
            options=decode.DecodeRunOptions(
                force=True,
                video_path=source,
                cal=cal,
                evidence_policy=det.evidence_policy,
            ),
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


@pytest.mark.parametrize("failure", ["invalid_metadata", "policy_changed"])
def test_policy_failures_cannot_become_optional_model_fallback(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", failure: str
) -> None:
    """Verify policy failures cannot become optional model fallback."""
    source, work, cal, hands, _det, _clf = _inputs(tmp_path, monkeypatch)
    if failure == "invalid_metadata":

        def invalid() -> None:
            msg = "Invalid evidence metadata"
            raise ValueError(msg)

        monkeypatch.setattr(evidence_policy, "load_policy", invalid)
    else:
        monkeypatch.setattr(
            evidence_policy, "load_policy", lambda: _policy(dense_hand=0.1)
        )
    monkeypatch.setattr(
        detector,
        "Detector",
        lambda: pytest.fail("policy preflight must precede model loading"),
    )
    with pytest.raises(ValueError, match=r"Invalid evidence metadata|policy changed"):
        decode.run_decode(
            work,
            hands,
            [],
            options=decode.DecodeRunOptions(
                force=True,
                video_path=source,
                cal=cal,
                evidence_policy=evidence_policy.DEFAULT_POLICY,
            ),
        )
    assert all(
        (work / "decode" / f"{h['hand']:02d}.json").read_text() == '{"previous":true}'
        for h in hands
    )


def test_dense_only_policy_invalidates_decode_but_preserves_observation_bytes(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify dense only policy invalidates decode but preserves observation bytes."""
    source, work, _cal, hands, _det, _clf = _inputs(tmp_path, monkeypatch)
    hands = hands[:1]
    monkeypatch.setattr(decode, "DATA_DIR", tmp_path)
    monkeypatch.setattr(decode, "facts_for_hand", lambda *_unused_args: {})
    (work / "calm.jsonl").write_text("")
    for region in read.REGIONS:
        (work / "reads/00" / f"{region.replace(':', '_')}.jsonl").write_text("")
    observe.run_observe(
        work,
        hands,
        [],
        options=observe.ObservationOptions(
            force=True, policy=evidence_policy.DEFAULT_POLICY
        ),
    )
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
            "problems": [],
        }

    monkeypatch.setattr(decode, "decode_hand", rebuild)
    games = [Game(1, {}, {}, [HandResult(0, 0, 0, {}, "draw")])]
    first = decode.run_decode(
        work,
        hands,
        games,
        options=decode.DecodeRunOptions(evidence_policy=evidence_policy.DEFAULT_POLICY),
    )[0]
    policy = _policy(dense_hand=0.1)
    observe.validate_observation_cache(work, hands, policy=policy)
    second = decode.run_decode(
        work, hands, games, options=decode.DecodeRunOptions(evidence_policy=policy)
    )[0]
    assert len(calls) == 2
    assert [path.read_bytes() for path in paths] == before
    assert (
        first["decode_inputs"]["dense_policy"]
        != second["decode_inputs"]["dense_policy"]
    )
    assert (
        first["decode_inputs"]["observation_sha256"]
        == second["decode_inputs"]["observation_sha256"]
    )
    # Omitting the explicit policy uses metadata only, even for nondefault policies.
    monkeypatch.setattr(evidence_policy, "load_policy", lambda: policy)
    monkeypatch.setattr(
        detector,
        "Detector",
        lambda: pytest.fail("valid bound cache must not load models"),
    )
    assert (
        decode.run_decode(
            work, hands, games, options=decode.DecodeRunOptions(video_path=source)
        )[0]
        == second
    )
    assert len(calls) == 2


def _decode_resume_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Publish empty but valid evidence and record reconstruction attempts."""
    _source, work, _cal, hands, _det, _clf = _inputs(tmp_path, monkeypatch)
    hands = hands[:1]
    monkeypatch.setattr(decode, "DATA_DIR", tmp_path)
    monkeypatch.setattr(decode, "facts_for_hand", lambda *_unused_args: {})
    (work / "calm.jsonl").write_text("")
    for region in read.REGIONS:
        (work / "reads" / "00" / f"{region.replace(':', '_')}.jsonl").write_text("")
    observe.run_observe(work, hands, [], options=observe.ObservationOptions(force=True))
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
            "problems": [],
        }

    monkeypatch.setattr(decode, "decode_hand", rebuild)
    games = [Game(1, {}, {}, [HandResult(0, 0, 0, {}, "draw")])]
    return work, hands, games, seen
