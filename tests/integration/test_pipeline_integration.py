# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Convert turns an authoritative record and stage evidence into checked logs.

Video, model, scoring-site and solver boundaries are replaced; the command's record
persistence, alignment and geometry gates, observation voting and export run for
real. The small legal ron needs no trained model, network or private recording.
These tests catch seat rotation, invalid-hand leakage and review/export disagreement.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.builders import publish_reads
from video2tenhou import (
    calibfit,
    calm,
    cli,
    export,
    observe,
    read,
    record,
    tenhou6,
    timeline,
)
from video2tenhou.calm import Interval
from video2tenhou.engine import decode
from video2tenhou.layout import Calibration
from video2tenhou.perception import classifier, detector
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY

INTERVAL = Interval("pond:TL", 0.0, 2.0, 3, calm=True, motion=0.0, skin=0.0)


def pond_tile(work: Path) -> str:
    """Return the first top-left pond tile voted for hand 0."""
    return observe.load_obs(work, 0)["pond:TL"][0]["slots"][0]["tile"]


@pytest.fixture
def conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A recording whose hand-0 readings are published and whose inputs changed."""
    game, entries = aligned_record()
    work = tmp_path / "work" / "broadcast"
    publish_reads(work / "reads/00", "1m", complete=True)
    for marker in ("calibration.changed", "inputs.changed"):
        (work / marker).write_text("rebuild needed")
    fetched, rebuilt, decodes = [], [], {}
    monkeypatch.setattr(
        record, "fetch_game", lambda game_id: fetched.append(game_id) or game
    )
    monkeypatch.setattr(timeline, "run_header", lambda *_args, **_kw: (entries, []))
    monkeypatch.setattr(calm, "run_calm", lambda *_args, **_kw: [INTERVAL])
    monkeypatch.setattr(
        detector, "Detector", lambda: SimpleNamespace(evidence_policy=DEFAULT_POLICY)
    )
    monkeypatch.setattr(classifier, "Classifier", object)
    monkeypatch.setattr(
        read, "run_read", lambda *_args, **_kw: ({"readings": 0}, set())
    )

    def run_decode(
        work_dir: Path,
        hands: list[dict],
        _games: list,
        *,
        force: bool = False,
        only: set[int] | None = None,
        **_unused_kwargs: object,
    ) -> list[dict]:
        """Rebuild forced or new hands from their observations; reuse the rest."""
        for entry in hands:
            hand = entry["hand"]
            forced = force and (only is None or hand in only)
            if forced or hand not in decodes:
                rebuilt.append(pond_tile(work_dir))
                decodes[hand] = reconstruction()
        return [decodes[entry["hand"]] for entry in hands]

    monkeypatch.setattr(decode, "run_decode", run_decode)
    args = Namespace(
        calib="pml",
        video="broadcast.mp4",
        work=tmp_path / "work",
        out=tmp_path / "out",
        force=False,
        game=[101],
        stop=None,
        redo=None,
        reread=[],
        hands=[],
        skip_fit_check=True,
    )
    return SimpleNamespace(
        args=args,
        work=work,
        out=tmp_path / "out" / "broadcast",
        fetched=fetched,
        rebuilt=rebuilt,
    )


def test_convert_publishes_a_replayable_log_and_reuses_the_saved_record(
    conversion: SimpleNamespace,
) -> None:
    c = conversion
    cli.cmd_convert(c.args)
    exported = json.loads((c.out / "g0.json").read_text(encoding="utf-8"))
    assert tenhou6.replay(exported) == []
    assert len(exported["log"]) == 1
    assert c.rebuilt == ["1m"]
    assert not (c.work / "calibration.changed").exists()
    assert not (c.work / "inputs.changed").exists()
    cli.cmd_convert(c.args)
    assert c.fetched == [101]
    assert c.rebuilt == ["1m"]  # unchanged evidence keeps its reconstruction


def test_misaligned_record_is_refused_before_any_evidence_is_voted(
    conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = conversion
    _, entries = aligned_record()
    monkeypatch.setattr(
        timeline, "run_header", lambda *_args, **_kw: (entries, ["wrong game"])
    )
    with pytest.raises(cli.CommandFailed, match="do not match the score record"):
        cli.cmd_convert(c.args)
    assert not (c.work / "obs").exists()
    assert not c.out.exists()
    assert (c.work / "calibration.changed").exists()


def test_readings_published_before_an_interruption_are_voted_and_decoded(
    conversion: SimpleNamespace,
) -> None:
    """A later process published new readings and stopped before voting."""
    c = conversion
    cli.cmd_convert(c.args)
    publish_reads(c.work / "reads/00", "2p", complete=True)
    cli.cmd_convert(c.args)
    assert pond_tile(c.work) == "2p"
    assert c.rebuilt == ["1m", "2p"]
    cli.cmd_convert(c.args)
    assert c.rebuilt == ["1m", "2p"]


def test_convert_failed_export_keeps_rebuild_markers(
    conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = conversion

    def failed(*_args: object, **_kw: object) -> None:
        raise OSError("output unavailable")

    monkeypatch.setattr(export, "write_outputs", failed)
    with pytest.raises(OSError, match="output unavailable"):
        cli.cmd_convert(c.args)
    assert (c.work / "calibration.changed").exists()
    assert (c.work / "inputs.changed").exists()


@pytest.fixture
def gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fitted recording whose geometry checks and table-timing scans are counted."""
    source = tmp_path / "broadcast.mp4"
    source.write_bytes(b"recording fixture")
    game, _ = aligned_record()
    cal = Calibration.load("pml")
    cal.fit = {"overhead": {"iou": 1.0}}
    monkeypatch.setattr(Calibration, "load", classmethod(lambda _cls, *_args: cal))
    monkeypatch.setattr(record, "fetch_game", lambda _game_id: game)
    gate = SimpleNamespace(
        samples=[{"t_start": 30.0, "t_end": 30.0}], checked=[], checks=[], scans=0
    )

    def scan(*_args: object, **_kw: object) -> dict:
        gate.scans += 1
        return {
            c: [
                {"t0": t, "t1": t + 1, "count": count, "n_used": 1, "partial": False}
                for t, count in [(0, 1), (20, 4), (40, 8), (60, 10)]
            ]
            for c in timeline.CORNERS
        }

    def check(
        _recording: object, _cal: Calibration, hands: list[dict], **_kw: object
    ) -> list:
        gate.checked.append(hands)
        return gate.checks

    monkeypatch.setattr(timeline, "read_pond_counts", scan)
    monkeypatch.setattr(timeline.calm, "run_calm", lambda *_args, **_kw: [])
    monkeypatch.setattr(
        timeline.video, "probe", lambda *_args: SimpleNamespace(duration=61)
    )
    monkeypatch.setattr(detector, "Detector", lambda: SimpleNamespace(id="test"))
    monkeypatch.setattr(classifier, "Classifier", object)
    monkeypatch.setattr(calibfit, "prepare_table_samples", lambda *_args: gate.samples)
    monkeypatch.setattr(calibfit, "table_plate", lambda *_args, **_kw: object())
    monkeypatch.setattr(calibfit, "pond_blocks", lambda *_args: None)
    monkeypatch.setattr(calibfit, "check_regions", check)
    gate.args = Namespace(
        calib="pml",
        video=str(source),
        work=tmp_path / "work",
        out=tmp_path / "out",
        force=True,
        game=[101],
        stop="header",
        skip_fit_check=False,
    )
    return gate


def test_cut_tiles_stop_conversion_before_table_timing_is_read(
    gated: SimpleNamespace,
) -> None:
    cut = calibfit.RegionCheck("pond:TL", held=3, cut=3, frames=8)
    cut.verdict()
    gated.checks = [cut]
    with pytest.raises(cli.CommandFailed, match="Adjust the table borders"):
        cli.cmd_convert(gated.args)
    assert gated.scans == 0


def test_forced_conversion_checks_geometry_on_independent_table_samples(
    gated: SimpleNamespace,
) -> None:
    """Geometry is measured on table samples, never on the hand timing it gates."""
    cli.cmd_convert(gated.args)
    cli.cmd_convert(gated.args)
    assert gated.checked == [gated.samples, gated.samples]
    assert gated.scans == 2


def site_payload() -> dict:
    """Create an authoritative game response for pipeline integration."""
    return {
        "id": 101,
        "players": [
            {"startingDirection": s, "user": {"username": f"Player {i}"}}
            for i, s in enumerate(record.SEATS)
        ],
        "score": dict(zip(record.SEATS, [24000, 26000, 25000, 25000], strict=False)),
        "handResults": [
            {
                "round": 0,
                "honba": 0,
                "riichiLeftover": 0,
                "scoreDelta": dict(
                    zip(record.SEATS, [-1000, 1000, 0, 0], strict=False)
                ),
                "events": [
                    {
                        "type": "RON",
                        "actor": "SOUTH",
                        "target": "EAST",
                        "han": 1,
                        "fu": 30,
                    }
                ],
            }
        ],
    }


def reconstruction() -> dict:
    """Create a complete synthetic hand suitable for replay validation."""
    hands = [
        "111222333444s1z5p",
        "123456789m123p5p",
        "555666777888s2z",
        "999s3334445556z",
    ]
    return {
        "hand": 0,
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "dealer": "E",
        "haipai": {
            s: [tenhou6.tile_str(t) for t in tenhou6.tiles(h)]
            for s, h in zip("ESWN", hands, strict=False)
        },
        "dora": ["1p"],
        "ura": [],
        "draws": {},
        "turns": [
            {
                "seat": "E",
                "j": 0,
                "kind": "draw",
                "draw": None,
                "discard": "5p",
                "tsumogiri": True,
                "riichi": False,
            }
        ],
        "result": {"outcome": "ron", "winner": "S", "loser": "E", "han": 1, "fu": 30},
        "solver": {"status": "optimal"},
        "score": None,
        "confidence": [],
        "items": [],
        "stats": {"turns": 1, "calls": 0},
    }


def aligned_record() -> tuple:
    """Align synthetic site results with one known recording window."""
    game = record.from_dict(record.to_dict(record.parse_game(site_payload())))
    entries, problems = timeline.site_entries([(0, 60)], [game])
    assert not problems
    return game, entries


def test_record_alignment_export_and_replay(tmp_path: Path) -> None:
    game, entries = aligned_record()
    decode = reconstruction()
    # A hand awaiting a human answer remains exportable and visibly under review.
    decode["items"] = [
        {"kind": "draw", "seat": "S", "hand": ["5p"], "text": "Confirm tile"}
    ]
    export.write_outputs(tmp_path, [game], [decode], entries, decode_dir=tmp_path)
    exported = json.loads((tmp_path / "g0.json").read_text(encoding="utf-8"))
    assert tenhou6.replay(exported) == []
    assert exported["name"] == [f"Player {i}" for i in range(4)]
    assert exported["log"][0][5:7] == [[25], [60]]
    assert exported["log"][0][-1][1] == [-1000, 1000, 0, 0]
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert review[0]["hand"] == 0
    assert review[0]["tiles"] == ["5p"]
    assert "| review | yes |" in (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "g0.html").read_text(encoding="utf-8")
    assert "https://tenhou.net/5/" in html
    assert "https://tenhou.net/6/" in html


def test_illegal_reconstruction_is_excluded_with_actionable_conflict(
    tmp_path: Path,
) -> None:
    game, entries = aligned_record()
    decode = reconstruction()
    decode["haipai"]["S"][0] = "9p"  # destroys the advertised winning hand
    export.write_outputs(tmp_path, [game], [decode], entries, decode_dir=tmp_path)
    assert json.loads((tmp_path / "g0.json").read_text(encoding="utf-8"))["log"] == []
    assert "| conflict | no |" in (tmp_path / "report.md").read_text(encoding="utf-8")
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    (conflict,) = [item for item in review if item["kind"] == "conflict"]
    winner = decode["result"]["winner"]
    assert ("not_winning", winner) in {
        (v["kind"], v["seat"]) for v in conflict["violations"]
    }


def test_site_record_supplies_hand_metadata() -> None:
    game, entries = aligned_record()
    game.hands[0].honba = 1
    entries, problems = timeline.site_entries([(0, 60)], [game])
    assert not problems
    assert entries[0]["honba"] == 1


def test_missing_decode_is_reported_instead_of_an_empty_complete_game(
    tmp_path: Path,
) -> None:
    game, entries = aligned_record()
    export.write_outputs(tmp_path, [game], [], entries, decode_dir=tmp_path)
    assert json.loads((tmp_path / "g0.json").read_text(encoding="utf-8"))["log"] == []
    review = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert len(review) == 1
    assert review[0]["kind"] == "conflict"
    assert review[0]["hand"] == 0
    assert "Analyze" in review[0]["text"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| conflict | no |" in report
    assert "0 of 1 hands written" in report
