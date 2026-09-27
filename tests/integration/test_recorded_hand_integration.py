"""Replay real recognition through decoding, scoring, export, and legality checks.

The compressed fixture contains no video, models, or player names. Recorded
recognition and its acquisition plan replace video access; evidence mapping and
game reconstruction run normally on CPU. See data/README.md for provenance.
"""

from tests.paths import DATA
import gzip
import json
from pathlib import Path

from video2tenhou.engine import decode, dense
from video2tenhou.engine.assemble import kyoku_from_decode, player_index
from video2tenhou.engine.decode import decode_hand
from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import replay_kyoku


def test_week11_e4_honba1_draw_order_and_export(monkeypatch, tmp_path):
    with gzip.open(DATA / "week11_e4h1.json.gz", "rt", encoding="utf-8") as f:
        fixture = json.load(f)
    requests = []
    missing = []

    def recorded(video, cal, work, det, clf, lo, hi, regions, **kwargs):
        requests.append((lo, hi, tuple(regions)))
        matching = [r for r in fixture["dense_reads"]
                    if abs(r["lo"] - lo) < 0.01 and abs(r["hi"] - hi) < 0.01
                    and set(regions) == set(r["regions"])]
        if not matching:
            missing.append((lo, hi, regions))
            raise AssertionError(f"Unrecorded dense request: {lo}-{hi} {regions}")
        return matching[0]["data"]

    def ponds(video, cal, work, det, clf, lo, hi, corners, **kwargs):
        data = recorded(video, cal, work, det, clf, lo, hi, [f"pond:{c}" for c in corners], **kwargs)
        return {c: data[f"pond:{c}"] for c in corners}

    monkeypatch.setattr(dense, "dense_reads", recorded)
    monkeypatch.setattr(dense, "dense_pond_reads", ponds)
    # Replay the acquisition plan that produced these exact recorded windows.
    # Timed alternative searches legitimately select different windows under
    # CPU contention; their threshold/bound contract has separate unit tests.
    # This fixes only acquisition, never tile assignments or confidence proofs.
    acquisition = [("W", 1), ("W", 5), ("N", 0), ("N", 4), ("N", 5)]
    monkeypatch.setattr(decode, "draws_to_reread", lambda solution, model=None: acquisition)
    result = HandResult(**fixture["result"])
    decoded = decode_hand(fixture["entry"], fixture["observations"], result, fixture["facts"],
                          models=(None, None, None, None), work_dir=tmp_path)
    north = {t["j"]: t for t in decoded["turns"] if t["seat"] == "N"}
    assert not missing  # the decoder catches reader failures; check them outside it
    assert [(north[j]["draw"], north[j]["discard"], north[j]["tsumogiri"]) for j in (4, 5)] == [
        ("2p", "2p", True), ("1z", "1z", True)]
    for j in (4, 5):
        # Alternative proofs have a wall-time ceiling. Slower/loaded CI may
        # retain an honest uncertainty question instead of a certificate, but
        # it must never silently accept an uncertified target draw.
        if north[j]["margin"] <= 0.5 + 1e-9:
            assert any((item.get("kind") in ("draw", "lost") and item.get("seat") == "N" and item.get("j") == j)
                       or (item.get("kind") == "uncertain_tiles" and any(
                           c["field"] == "draw" and c["seat"] == "N" and c["j"] == j for c in item["choices"]))
                       for item in decoded["items"])
    assert {(lo, hi, frozenset(regions)) for lo, hi, regions in requests} == {
        (row["lo"], row["hi"], frozenset(row["regions"])) for row in fixture["dense_reads"]}
    assert decoded["solver"]["status"] == "optimal"
    assert decoded["score"]["match"]
    assert (decoded["score"]["han"], decoded["score"]["fu"]) == (10, 40)
    assert not any(item["kind"] == "conflict" for item in decoded["items"])

    kyoku, confidence = kyoku_from_decode(decoded, fixture["entry"], result)
    assert replay_kyoku(kyoku.dump()) == []
    seat = player_index("N", fixture["entry"]["kyoku"])
    assert kyoku.draws[seat][4:6] == [22, 41]
    assert kyoku.discards[seat][4:6] == [60, 60]
    assert {(c["turn"], c["value"]) for c in confidence if c["seat"] == "N" and c["field"] == "draw"
            and c["turn"] in (4, 5)} == {(4, "2p"), (5, "1z")}


def test_cached_decode_does_not_load_models(monkeypatch, tmp_path):
    """Opening completed work should not initialize the GPU or require model files."""
    from video2tenhou.engine import decode
    from video2tenhou.perception import detector, evidence_policy

    work = tmp_path / "video"
    (work / "decode").mkdir(parents=True)
    (work / "obs").mkdir()
    (work / "obs/00.json").write_text("{}", encoding="utf-8")
    (work / "obs/provenance").mkdir()
    (work / "obs/provenance/00.json").write_text('{"inputs":"verified-fixture"}', encoding="utf-8")
    cached = {"hand": 0, "decoder_version": decode.DECODER_VERSION,
              "decode_inputs": decode._decode_input_binding(work, 0, evidence_policy=evidence_policy.DEFAULT_POLICY)}
    (work / "decode/00.json").write_text(json.dumps(cached), encoding="utf-8")
    monkeypatch.setattr(decode, "DATA_DIR", tmp_path)

    def unexpected():
        raise AssertionError("Models must not load for cached output")

    monkeypatch.setattr(detector, "Detector", unexpected)
    assert decode.run_decode(work, [{"hand": 0}], [], video_path=tmp_path / "video.mp4", evidence_policy=evidence_policy.DEFAULT_POLICY) == [cached]


def test_old_decoder_cache_cannot_be_reused_without_observations(monkeypatch, tmp_path):
    from video2tenhou.engine import decode

    work = tmp_path / "video"
    (work / "decode").mkdir(parents=True)
    (work / "decode/00.json").write_text('{"hand":0}', encoding="utf-8")
    monkeypatch.setattr(decode, "DATA_DIR", tmp_path)
    # With no observations available, omit the hand rather than returning a
    # reconstruction from a decoder whose evidence semantics are obsolete.
    from video2tenhou.perception.evidence_policy import DEFAULT_POLICY
    assert decode.run_decode(work, [{"hand": 0}], [], evidence_policy=DEFAULT_POLICY) == []
