"""Exercise the handoff from authoritative records to checked, downloadable logs.

The small legal ron needs no trained model, network or private recording. These
tests catch seat rotation, invalid-hand leakage and review/export disagreement.
"""
import json
from types import SimpleNamespace

import pytest

from video2tenhou import cli, record, tenhou6, timeline
from video2tenhou.observe import run_observe as actual_run_observe


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    """CPU stage doubles retain the real command, record persistence and export path."""
    from video2tenhou import calm, observe, read
    from video2tenhou.engine import decode
    from video2tenhou.perception import classifier, detector
    game, entries = aligned_record()
    work = tmp_path / "work" / "broadcast"
    work.mkdir(parents=True)
    for marker in ("calibration.changed", "inputs.changed"):
        (work / marker).write_text("rebuild needed")
    calls = []
    monkeypatch.setattr(cli, "_require_fit", lambda *args: None)
    monkeypatch.setattr(cli, "_gate", lambda *args: calls.append("geometry"))
    monkeypatch.setattr(record, "fetch_game", lambda *args: calls.append("record") or game)
    monkeypatch.setattr(timeline, "run_header", lambda *args, **kwargs: (entries, []))
    monkeypatch.setattr(calm, "run_calm", lambda *args, **kwargs: [])
    monkeypatch.setattr(detector, "Detector", lambda: object())
    monkeypatch.setattr(classifier, "Classifier", lambda: object())
    monkeypatch.setattr(read, "run_read", lambda *args, **kwargs: ({"readings": 0}, {0}))
    def observations(*args, **kwargs):
        assert kwargs["touched"] == {0}
        calls.append("observe")
        return {}
    monkeypatch.setattr(observe, "run_observe", observations)
    def decoding(*args, **kwargs):
        calls.append(("decode", kwargs["force"], kwargs.get("only")))
        return [reconstruction()]
    monkeypatch.setattr(decode, "run_decode", decoding)
    args = SimpleNamespace(calib="pml", video="broadcast.mp4", work=tmp_path / "work",
                           out=tmp_path / "out", force=False, game=[101], stop=None,
                           redo=None, reread=[], hands=[])
    return SimpleNamespace(args=args, work=work, calls=calls, entries=entries)


def test_convert_propagates_removed_evidence_and_publishes_checked_outputs(conversion):
    """A removal-only read change must still rebuild observations and the affected hand."""
    c = conversion
    cli.cmd_convert(c.args)
    assert c.calls == ["record", "geometry", "observe", ("decode", True, {0}), ("decode", False, None)]
    assert not (c.work / "calibration.changed").exists()
    assert not (c.work / "inputs.changed").exists()
    exported = json.loads((c.args.out / "broadcast/g0.json").read_text(encoding="utf-8"))
    assert tenhou6.replay(exported) == [] and len(exported["log"]) == 1
    c.calls.clear()
    cli.cmd_convert(c.args)
    assert "record" not in c.calls  # persisted authoritative record reused


def test_convert_alignment_failure_stops_before_recognition(conversion, monkeypatch):
    c = conversion
    monkeypatch.setattr(timeline, "run_header", lambda *args, **kwargs: (c.entries, ["wrong game"]))
    with pytest.raises(SystemExit):
        cli.cmd_convert(c.args)
    assert c.calls == ["record"]
    assert (c.work / "calibration.changed").exists()
    assert not c.args.out.exists()


def test_resume_after_read_publication_revotes_and_redecodes_without_new_reads(conversion, monkeypatch):
    """A prior process may finish recognition but stop before voting or decoding."""
    from video2tenhou import calm, observe, read
    from video2tenhou.calm import Interval
    from video2tenhou.train.data import CLASSES
    c = conversion
    interval = Interval("pond:TL", 0., 2., 3, True, 0., 0.)
    hdir = c.work / "reads/00"
    hdir.mkdir(parents=True)
    (hdir / "done.json").write_text('{"complete":true}')
    path = hdir / "pond_TL.jsonl"
    def write(tile):
        p = [0.] * len(CLASSES)
        p[CLASSES.index(tile)] = 1.
        path.write_text("".join(json.dumps(dict(t=t, region="pond:TL", size=[400, 700], boxes=[
            dict(xyxy=[10, 10, 50, 70], conf=.9, sideways=False, p=p)])) + "\n" for t in (0., 1., 2.)))
    write("1m")
    actual_run_observe(c.work, c.entries, [interval], log=lambda _: None)
    write("2p")
    monkeypatch.setattr(read, "run_read", lambda *args, **kwargs: ({"readings": 0}, set()))
    monkeypatch.setattr(calm, "run_calm", lambda *args, **kwargs: [interval])
    monkeypatch.setattr(observe, "run_observe", actual_run_observe)
    cli.cmd_convert(c.args)
    assert observe.load_obs(c.work, 0)["pond:TL"][0]["slots"][0]["tile"] == "2p"
    assert [call for call in c.calls if isinstance(call, tuple)] == [
        ("decode", True, {0}), ("decode", False, None)]
    c.calls.clear()
    cli.cmd_convert(c.args)
    assert [call for call in c.calls if isinstance(call, tuple)] == [("decode", False, None)]


def test_convert_failed_export_keeps_rebuild_markers(conversion, monkeypatch):
    c = conversion
    def failed(*args):
        raise OSError("output unavailable")
    monkeypatch.setattr(cli, "write_outputs", failed)
    with pytest.raises(OSError, match="output unavailable"):
        cli.cmd_convert(c.args)
    assert (c.work / "calibration.changed").exists()
    assert (c.work / "inputs.changed").exists()


def test_forced_conversion_scans_headers_once_for_alignment_and_geometry(tmp_path, monkeypatch):
    from video2tenhou import calibfit
    from video2tenhou.layout import Calibration
    from video2tenhou.perception import detector
    source = tmp_path / "broadcast.mp4"
    source.write_bytes(b"recording fixture")
    game, _ = aligned_record()
    cal = Calibration.load("pml")
    cal.fit = {"overhead": {"iou": 1.}}
    monkeypatch.setattr(Calibration, "load", classmethod(lambda cls, *args: cal))
    monkeypatch.setattr(record, "fetch_game", lambda gid: game)
    scans, checked = [], []
    def scan(*args, **kwargs):
        scans.append(True)
        yield from ({"t": float(t), "kyoku": 0, "honba": 0, "sticks": 0,
                     "scores": dict.fromkeys(timeline.CORNERS, 25000),
                     "winds": dict(zip(timeline.CORNERS, "ENSW")), "ok": True} for t in range(61))
    monkeypatch.setattr(timeline, "scan", scan)
    monkeypatch.setattr(timeline, "read_nicks", lambda *args, **kwargs: {})
    monkeypatch.setattr(detector, "Detector", lambda: object())
    monkeypatch.setattr(calibfit, "table_plate", lambda *args, **kwargs: object())
    monkeypatch.setattr(calibfit, "pond_blocks", lambda *args: None)
    def check(path, calibration, model, hands, **kwargs):
        checked.append(hands)
        return []
    monkeypatch.setattr(calibfit, "check_regions", check)
    args = SimpleNamespace(calib="pml", video=str(source), work=tmp_path / "work", out=tmp_path / "out",
                           force=True, game=[101], stop="header", skip_fit_check=False)
    cli.cmd_convert(args)
    assert len(scans) == 1 and checked == [[{"t_start": 0., "t_end": 60.}]]
    # A forced refresh scans again, but its geometry gate must not scan a second time.
    cli.cmd_convert(args)
    assert len(scans) == len(checked) == 2


def site_payload():
    return {
        "id": 101, "players": [
            {"startingDirection": s, "user": {"username": f"Player {i}"}}
            for i, s in enumerate(record.SEATS)
        ], "score": dict(zip(record.SEATS, [24000, 26000, 25000, 25000])),
        "handResults": [{"round": 0, "honba": 0, "riichiLeftover": 0,
                         "scoreDelta": dict(zip(record.SEATS, [-1000, 1000, 0, 0])),
                         "events": [{"type": "RON", "actor": "SOUTH", "target": "EAST", "han": 1, "fu": 30}]}],
    }


def reconstruction():
    hands = ["111222333444s1z5p", "123456789m123p5p", "555666777888s2z", "999s3334445556z"]
    return {
        "hand": 0, "game": 0, "kyoku": 0, "honba": 0, "dealer": "E",
        "haipai": {s: [tenhou6.tile_str(t) for t in tenhou6.tiles(h)] for s, h in zip("ESWN", hands)},
        "dora": ["1p"], "ura": [], "draws": {},
        "turns": [{"seat": "E", "j": 0, "kind": "draw", "draw": None, "discard": "5p",
                   "tsumogiri": True, "riichi": False}],
        "result": {"outcome": "ron", "winner": "S", "loser": "E", "han": 1, "fu": 30},
        "solver": {"status": "optimal"}, "score": None,
        "confidence": [], "items": [], "stats": {"turns": 1, "calls": 0},
    }


def aligned_record():
    game = record.from_dict(record.to_dict(record.parse_game(site_payload())))
    observed = timeline.Hand(0, 0, 0, 0, 60, 0, dict.fromkeys(timeline.CORNERS, 25000),
                            dict(zip(timeline.CORNERS, "ENSW")))
    entries, problems = timeline.align([observed], [game])
    assert not problems
    return game, entries


def test_record_alignment_export_and_replay(tmp_path):
    game, entries = aligned_record()
    decode = reconstruction()
    # A hand awaiting a human answer remains exportable and visibly under review.
    decode["items"] = [{"kind": "draw", "seat": "S", "hand": ["5p"], "text": "Confirm tile"}]
    cli.write_outputs(tmp_path, [game], [decode], entries, "Integration")
    exported = json.loads((tmp_path / "g0.json").read_text(encoding="utf-8"))
    assert tenhou6.replay(exported) == []
    assert exported["name"] == [f"Player {i}" for i in range(4)]
    assert exported["log"][0][5:7] == [[25], [60]]
    assert exported["log"][0][-1][1] == [-1000, 1000, 0, 0]
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert review[0]["hand"] == 0 and review[0]["tiles"] == ["5p"]
    assert "| review | yes |" in (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "g0.html").read_text(encoding="utf-8")
    assert "https://tenhou.net/5/" in html and "https://tenhou.net/6/" in html


def test_illegal_reconstruction_is_excluded_with_actionable_conflict(tmp_path):
    game, entries = aligned_record()
    decode = reconstruction()
    decode["haipai"]["S"][0] = "9p"  # destroys the advertised winning hand
    cli.write_outputs(tmp_path, [game], [decode], entries, "Invalid")
    assert json.loads((tmp_path / "g0.json").read_text(encoding="utf-8"))["log"] == []
    assert "| conflict | no |" in (tmp_path / "report.md").read_text(encoding="utf-8")
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert any(item["kind"] == "conflict" and "winning hand" in item["text"] for item in review)


def test_site_record_disagreement_stops_alignment():
    game, entries = aligned_record()
    game.hands[0].honba = 1
    observed = timeline.Hand(0, 0, 0, 0, 60, 0, dict.fromkeys(timeline.CORNERS, 25000),
                            dict(zip(timeline.CORNERS, "ENSW")))
    _, problems = timeline.align([observed], [game])
    assert any("vs site" in problem for problem in problems)


def test_missing_decode_is_reported_instead_of_an_empty_complete_game(tmp_path):
    game, entries = aligned_record()
    cli.write_outputs(tmp_path, [game], [], entries, "Missing evidence")
    assert json.loads((tmp_path / "g0.json").read_text(encoding="utf-8"))["log"] == []
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert len(review) == 1 and review[0]["kind"] == "conflict" and review[0]["hand"] == 0
    assert "Analyze" in review[0]["text"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| conflict | no |" in report and "0 of 1 hands written" in report
