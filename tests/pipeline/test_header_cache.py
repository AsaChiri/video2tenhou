"""Header cache provenance and interrupted-write integration coverage."""
import copy
import json
import os
from types import SimpleNamespace

import pytest

from video2tenhou import timeline
from video2tenhou.layout import Calibration, CORNERS
from video2tenhou.record import Game, HandResult


@pytest.fixture
def header(tmp_path, monkeypatch):
    source = tmp_path / "broadcast.mp4"
    source.write_bytes(b"video fixture: decoding is replaced")
    work = tmp_path / "work"
    cal = Calibration.load("pml")
    winds = dict(zip(CORNERS, "ESWN"))
    names = dict(zip(CORNERS, ("a", "b", "c", "d")))
    game = Game(17, dict(zip(("EAST", "SOUTH", "WEST", "NORTH"), names.values())), {},
                [HandResult(0, 0, 0, {s: 0 for s in ("EAST", "SOUTH", "WEST", "NORTH")}, "draw")])
    rows = [{"t": float(t), "kyoku": 0, "honba": 0, "sticks": 0,
             "scores": {c: 25000 for c in CORNERS}, "winds": winds, "ok": True} for t in range(41)]
    scans, nicks = [], []

    def scan(path, calibration, fps=1.):
        scans.append((path, calibration, fps))
        yield from copy.deepcopy(rows)

    def read_nicks(path, t, cal=None):
        nicks.append(cal)
        return names

    monkeypatch.setattr(timeline, "scan", scan)
    monkeypatch.setattr(timeline, "read_nicks", read_nicks)

    def run(calibration=None, **kwargs):
        entries, problems = timeline.run_header(source, calibration or cal, [game], work, log=lambda *args: None, **kwargs)
        assert problems == [] and len(entries) == 1
        assert entries[0]["t_overlay"] == [0., 40.]
        return entries

    return SimpleNamespace(source=source, work=work, cal=cal, scans=scans, nicks=nicks, rows=rows, run=run, scan=scan)


def test_completed_scan_is_reused_but_names_and_alignment_still_run(header):
    first = header.run()
    assert header.run() == first
    assert len(header.scans) == 1
    assert header.nicks == [header.cal, header.cal]
    metadata = json.loads((header.work / "overlay.meta.json").read_text())
    assert metadata["readings"] == 41 and len(metadata["sha256"]) == 64
    header.run(force=True)
    assert len(header.scans) == 2


def test_overlay_crops_frame_sampling_and_source_changes_invalidate(header):
    header.run()
    data = copy.deepcopy(header.cal.data)
    data["overlay"]["strip"]["TL"][0] += 20
    moved = Calibration(data)
    header.run(moved)
    data["overlay"]["round_num"][0] += 10
    moved = Calibration(data)
    header.run(moved)
    data["frame"] = [3840, 2160]
    resized = Calibration(data)
    header.run(resized)
    header.run(resized, fps=2.)
    header.source.write_bytes(b"a replacement recording with the same basename")
    header.run(resized, fps=2.)
    assert len(header.scans) == 6


def test_table_camera_edits_do_not_rescan_unchanged_overlay_pixels(header):
    header.run()
    data = copy.deepcopy(header.cal.data)
    data["overhead"]["center"][0] += 20
    data["hand"]["TL"]["rect"][0] += 20
    data["notes"] = "Updated table fit; header pixels are unchanged"
    header.run(Calibration(data))
    assert len(header.scans) == 1


def test_same_size_timestamp_replacement_invalidates_numeric_headers(header):
    header.run()
    stamp = header.source.stat()
    content = header.source.read_bytes()
    header.source.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(header.source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    header.run()
    assert len(header.scans) == 2


def test_unsigned_or_truncated_cache_is_never_reused(header):
    header.work.mkdir()
    overlay = header.work / "overlay.jsonl"
    overlay.write_text(json.dumps(header.rows[0]) + "\n")  # legacy interrupted cache
    header.run()
    assert len(header.scans) == 1
    overlay.write_text(json.dumps(header.rows[0]) + "\n")  # valid JSON, wrong completed-file digest
    header.run()
    assert len(header.scans) == 2
    (header.work / "overlay.meta.json").write_text("{broken")
    header.run()
    assert len(header.scans) == 3


@pytest.mark.parametrize("previous", [False, True])
def test_interrupted_scan_never_publishes_partial_rows(header, monkeypatch, previous):
    if previous:
        header.run()
    overlay = header.work / "overlay.jsonl"
    metadata = header.work / "overlay.meta.json"
    old = (overlay.read_bytes(), metadata.read_bytes()) if previous else None

    def interrupted(*args, **kwargs):
        yield header.rows[0]
        raise RuntimeError("decoder interrupted")

    monkeypatch.setattr(timeline, "scan", interrupted)
    with pytest.raises(RuntimeError, match="decoder interrupted"):
        header.run(force=True)
    if previous:
        assert (overlay.read_bytes(), metadata.read_bytes()) == old
    else:
        assert not overlay.exists() and not metadata.exists()
    assert not list(header.work.glob("*.tmp"))
    monkeypatch.setattr(timeline, "scan", header.scan)
    header.run()
    assert len(header.scans) == 1  # reuse old complete cache, or complete the first scan
