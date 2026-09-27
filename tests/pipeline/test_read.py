import json
import os

import pytest

import numpy as np

from video2tenhou import read
from video2tenhou.calm import Interval, REGIONS
from video2tenhou.layout import Calibration
from video2tenhou.perception.reader import Reading


class Stub:
    def __init__(self, id_):
        self.id = id_


def fake_pipeline(monkeypatch):
    """No video, no models: frames at 2 fps, empty readings; records the times read."""
    seen = []
    monkeypatch.setattr(read, "source_identity", lambda path, **kwargs: "test-video-content")

    def sample(path, fps=2.0, start=0.0, end=None):
        t = round(start * fps) / fps
        while t <= end:
            yield t, np.zeros((4, 4, 3), np.uint8)
            t += 1.0 / fps

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", lambda frame, cal, kind, corner: (frame, None))

    def read_regions(items, det, clf):
        seen.extend((name, t) for t, name, _ in items)
        return [Reading(t, name, (4, 4), []) for t, name, _ in items]

    monkeypatch.setattr(read, "read_regions", read_regions)
    return seen


def test_read_lines_drops_a_truncated_last_line(tmp_path, capsys):
    p = tmp_path / "pond_TL.jsonl"
    p.write_text('{"t": 1.0, "boxes": []}\n{"t": 1.5, "boxes": []}\n{"t": 2.0, "bo', encoding="utf-8")
    rows = read.read_lines(p)
    assert [r["t"] for r in rows] == [1.0, 1.5]
    assert "malformed" in capsys.readouterr().out
    assert [json.loads(l)["t"] for l in p.read_text(encoding="utf-8").splitlines()] == [1.0, 1.5]
    assert read.read_lines(tmp_path / "missing.jsonl") == []


def test_run_read_rereads_when_the_models_change_and_counts_the_files(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    hands = [{"hand": 3, "t_start": 100.0, "t_end": 110.0}]
    ivs = [Interval(r, 100.0, 110.0, 21, True, 0.0, 0.0) for r in REGIONS]
    st, touched = read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("det-a"), Stub("clf-a"), cap=3)
    assert touched == {3} and st["readings"] == 3 * len(REGIONS)
    done = json.loads((tmp_path / "reads" / "03" / "done.json").read_text())
    assert done["readings"] == 3 * len(REGIONS) and done["detector"] == "det-a"
    # same models: nothing to do, the hand is not touched
    seen.clear()
    st, touched = read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("det-a"), Stub("clf-a"), cap=3)
    assert touched == set() and not seen
    # a new detector: every region of the hand is read again, the count is the file's, not a running total
    st, touched = read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("det-b"), Stub("clf-a"), cap=3)
    assert touched == {3} and len(seen) == 3 * len(REGIONS)
    done = json.loads((tmp_path / "reads" / "03" / "done.json").read_text())
    assert done["readings"] == 3 * len(REGIONS) and done["detector"] == "det-b"
    # --reread pond: the pond files are rewritten, the others kept; the count stays the number of lines
    st, touched = read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("det-b"), Stub("clf-a"), cap=3, reread={"pond"})
    done = json.loads((tmp_path / "reads" / "03" / "done.json").read_text())
    assert done["readings"] == 3 * len(REGIONS) and touched == {3}


def test_run_read_repairs_a_truncated_line_and_reads_its_time_again(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    hands = [{"hand": 0, "t_start": 10.0, "t_end": 12.0}]
    ivs = [Interval(r, 10.0, 12.0, 5, True, 0.0, 0.0) for r in REGIONS]
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    p = tmp_path / "reads" / "00" / "pond_TL.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:10], encoding="utf-8")   # a crash mid-line
    seen.clear()
    st, touched = read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    assert seen == [("pond:TL", 12.0)] and touched == {0}
    assert [json.loads(l)["t"] for l in p.read_text(encoding="utf-8").splitlines()] == [10.0, 10.5, 11.0, 11.5, 12.0]
    assert read.load_reads(tmp_path, 0)["pond:TL"][-1]["t"] == 12.0


def test_dense_cache_is_keyed_by_models_and_geometry(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    read.dense_reads("v.mp4", cal, tmp_path, Stub("d1"), Stub("c1"), 5.0, 6.0, ["hand:TL"])
    per_call = len(seen)
    assert per_call >= 5
    read.dense_reads("v.mp4", cal, tmp_path, Stub("d1"), Stub("c1"), 5.0, 6.0, ["hand:TL"])   # cached
    read.dense_reads("v.mp4", cal, tmp_path, Stub("d2"), Stub("c1"), 5.0, 6.0, ["hand:TL"])   # another detector
    cal2 = Calibration(json.loads(json.dumps(cal.data)))
    cal2.data["hand"]["TL"]["roll"] = 0.0
    read.dense_reads("v.mp4", cal2, tmp_path, Stub("d1"), Stub("c1"), 5.0, 6.0, ["hand:TL"])  # another geometry
    cal3 = Calibration(json.loads(json.dumps(cal.data)))
    cal3.data["notes"] = ["something else"]
    read.dense_reads("v.mp4", cal3, tmp_path, Stub("d1"), Stub("c1"), 5.0, 6.0, ["hand:TL"])  # notes: same pixels, cached
    assert len(list((tmp_path / "dense").glob("*.json"))) == 3
    assert len(seen) == 3 * per_call
    read.clear_dense(tmp_path)
    assert not (tmp_path / "dense").exists()


@pytest.mark.parametrize("damage", ["manifest_list", "geometry_list"])
def test_sparse_resume_replaces_unproven_manifest_shapes(tmp_path, monkeypatch, damage):
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    hands = [{"hand": 0, "t_start": 10., "t_end": 11.}]
    intervals = [Interval("hand:TL", 10., 11., 3, True, 0., 0.)]
    read.run_read("video", cal, tmp_path, hands, intervals, Stub("d"), Stub("c"))
    path = tmp_path / "reads/00/done.json"
    manifest = json.loads(path.read_text())
    if damage == "manifest_list":
        manifest = []
    else:
        manifest["geometry"] = ["unproven crop"]
    path.write_text(json.dumps(manifest))
    seen.clear()

    _, touched = read.run_read("video", cal, tmp_path, hands, intervals, Stub("d"), Stub("c"))
    assert touched == {0} and seen == [("hand:TL", t) for t in (10., 10.5, 11.)]
    assert isinstance(json.loads(path.read_text())["geometry"], dict)
    seen.clear()
    assert read.run_read("video", cal, tmp_path, hands, intervals, Stub("d"), Stub("c"))[1] == set()
    assert not seen


@pytest.mark.parametrize("saved", [[], {}, {"hand:TL": None}, {"hand:TL": [None]},
                                  {"hand:TL": [{"t": 5.}]}])
def test_dense_resume_replaces_incomplete_region_payload(tmp_path, monkeypatch, saved):
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    expected = read.dense_reads("video", cal, tmp_path, Stub("d"), Stub("c"), 5., 6., ["hand:TL"])
    cache = next((tmp_path / "dense").glob("*.json"))
    cache.write_text(json.dumps(saved))
    seen.clear()
    assert read.dense_reads("video", cal, tmp_path, Stub("d"), Stub("c"), 5., 6., ["hand:TL"]) == expected
    assert seen and json.loads(cache.read_text()) == expected
    seen.clear()
    assert read.dense_reads("video", cal, tmp_path, Stub("d"), Stub("c"), 5., 6., ["hand:TL"]) == expected
    assert not seen


def test_only_the_regions_the_fit_moved_are_read_again(tmp_path, monkeypatch):
    """A rectangle dragged in the tool must not cost a re-read of the whole video, and must not leave the
    old crop's boxes in place either (DESIGN.md 4.2a)."""
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    hands = [{"hand": 0, "t_start": 10.0, "t_end": 12.0}]
    ivs = [Interval(r, 10.0, 12.0, 5, True, 0.0, 0.0) for r in REGIONS]
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    first = len(seen)
    assert first == 5 * len(REGIONS)

    seen.clear()
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    assert seen == []                                    # nothing moved: nothing is read again

    data = json.loads(json.dumps(cal.data))
    data["meld"]["TL"]["rect"] = [500, 400, 150, 190]         # the rectangle a human dragged in the tool
    moved = Calibration(data)
    seen.clear()
    _, touched = read.run_read("v.mp4", moved, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    assert {r for r, _ in seen} == {"meld:TL"} and touched == {0}
    assert len(seen) < first                             # the other eleven regions kept their readings

    seen.clear()
    read.run_read("v.mp4", moved, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    assert seen == []                                    # and the new geometry is now the cached one


def test_a_read_cache_from_before_per_region_keys_is_not_trusted(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    hands = [{"hand": 0, "t_start": 10.0, "t_end": 12.0}]
    ivs = [Interval(r, 10.0, 12.0, 5, True, 0.0, 0.0) for r in REGIONS]
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    first = len(seen)
    done = tmp_path / "reads" / "00" / "done.json"
    d = json.loads(done.read_text())
    d["geometry"] = "8ccc2663cf85"                       # the old single-string key
    done.write_text(json.dumps(d))
    seen.clear()
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=5)
    assert len(seen) == first                            # every region read again


def test_dense_cache_separates_sampling_rates_and_nearby_window_bounds(tmp_path, monkeypatch):
    """Both used to collide, returning evidence from the wrong sample schedule."""
    seen = fake_pipeline(monkeypatch)
    cal = Calibration.load("pml")
    det, clf = Stub("d"), Stub("c")
    read.dense_reads("v.mp4", cal, tmp_path, det, clf, 5., 6., ["hand:TL"], fps=2.)
    before = len(seen)
    read.dense_reads("v.mp4", cal, tmp_path, det, clf, 5., 6., ["hand:TL"], fps=5.)
    assert len(seen) > before
    read.dense_reads("v.mp4", cal, tmp_path, det, clf, 5.01, 6., ["hand:TL"], fps=5.)
    assert len(list((tmp_path / "dense").glob("*.json"))) == 3


def test_read_flushes_incomplete_batches_at_hand_boundaries(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    monkeypatch.setattr(read, "READ_BATCH", 5)
    cal = Calibration.load("pml")
    hands = [{"hand": i, "t_start": float(i * 10), "t_end": float(i * 10 + 1)} for i in range(2)]
    ivs = [Interval(r, h["t_start"], h["t_end"], 3, True, 0., 0.) for h in hands for r in REGIONS]
    stats, touched = read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("d"), Stub("c"), cap=3)
    assert touched == {0, 1} and stats["readings"] == 72
    assert len(seen) == 72
    for h in hands:
        rows = read.load_reads(tmp_path, h["hand"])
        assert all([d["t"] for d in values] == [h["t_start"] + dt for dt in (0., .5, 1.)]
                   for values in rows.values())


def test_changed_plan_removes_obsolete_rows_and_matches_forced_evidence(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    cal, det, clf = Calibration.load("pml"), Stub("d"), Stub("c")
    hands = [{"hand": 0, "t_start": 10., "t_end": 14.}]
    old = [Interval(r, 10., 14., 9, True, 0., 0.) for r in REGIONS]
    new = [Interval("pond:TL", 11., 13., 5, True, 0., 0.)]
    read.run_read("v.mp4", cal, tmp_path, hands, old, det, clf, cap=9)
    seen.clear()
    stats, touched = read.run_read("v.mp4", cal, tmp_path, hands, new, det, clf, cap=9)
    assert touched == {0} and stats["readings"] == 0 and seen == []
    incremental = read.load_reads(tmp_path, 0)
    assert [row["t"] for row in incremental["pond:TL"]] == [11., 11.5, 12., 12.5, 13.]
    assert all(not rows for region, rows in incremental.items() if region != "pond:TL")
    read.run_read("v.mp4", cal, tmp_path / "forced", hands, new, det, clf, cap=9, force=True)
    assert read.load_reads(tmp_path / "forced", 0) == incremental


def test_retry_uses_identical_decode_window_and_is_atomic_on_failure(tmp_path, monkeypatch):
    fake_pipeline(monkeypatch)
    windows = []
    original = read.video.sample
    def sample(path, **kwargs):
        windows.append((kwargs["start"], kwargs["end"]))
        yield from original(path, **kwargs)
    monkeypatch.setattr(read.video, "sample", sample)
    cal, det, clf = Calibration.load("pml"), Stub("d"), Stub("c")
    hands = [{"hand": 0, "t_start": 10., "t_end": 14.}]
    ivs = [Interval("pond:TL", 10., 14., 9, True, 0., 0.)]
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, det, clf, cap=9)
    region = tmp_path / "reads/00/pond_TL.jsonl"
    lines = region.read_text().splitlines()
    region.write_text("\n".join(lines[:3] + lines[4:]) + "\n")
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, det, clf, cap=9)
    assert windows == [(10., 14.5), (10., 14.5)]
    before = {p.name: p.read_bytes() for p in region.parent.iterdir()}
    def fail(*args, **kwargs):
        raise RuntimeError("decoder failed")
        yield
    monkeypatch.setattr(read.video, "sample", fail)
    with pytest.raises(RuntimeError, match="decoder failed"):
        read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("new"), clf)
    assert {p.name: p.read_bytes() for p in region.parent.iterdir()} == before


def test_source_content_classifier_calibration_and_preprocessing_invalidate(tmp_path, monkeypatch):
    identity = read.source_identity
    seen = fake_pipeline(monkeypatch)
    monkeypatch.setattr(read, "source_identity", identity)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"old video")
    stamp = source.stat()
    cal, det, clf = Calibration.load("pml"), Stub("d"), Stub("c")
    clf.classes, clf.T = ["1m", "2m"], 1.
    hands = [{"hand": 0, "t_start": 0., "t_end": 1.}]
    ivs = [Interval("hand:TL", 0., 1., 3, True, 0., 0.)]
    work = tmp_path / "work"
    read.run_read(source, cal, work, hands, ivs, det, clf)
    for change in ("content", "temperature", "preprocessing"):
        seen.clear()
        if change == "content":
            source.write_bytes(b"new video")
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        elif change == "temperature":
            clf.T = 2.
        else:
            monkeypatch.setattr(read, "PREPROCESSING", "changed-crop-pipeline")
        _, touched = read.run_read(source, cal, work, hands, ivs, det, clf)
        assert touched == {0} and len(seen) == 3


def test_incomplete_video_never_publishes_completion(tmp_path, monkeypatch):
    fake_pipeline(monkeypatch)
    monkeypatch.setattr(read.video, "sample", lambda *a, **k: iter([]))
    with pytest.raises(RuntimeError, match="before planned evidence"):
        read.run_read("v.mp4", Calibration.load("pml"), tmp_path,
                      [{"hand": 0, "t_start": 0., "t_end": 1.}],
                      [Interval("hand:TL", 0., 1., 3, True, 0., 0.)], Stub("d"), Stub("c"))
    assert not (tmp_path / "reads/00/done.json").exists()


def test_real_video_incremental_plan_has_identical_frame_evidence(tmp_path, monkeypatch):
    import hashlib
    import shutil
    import subprocess
    from types import SimpleNamespace
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=80x60:rate=24000/1001", "-t", "3", "-pix_fmt", "yuv420p", str(source)], check=True)
    monkeypatch.setattr(read, "region_upright", lambda frame, *a: (frame, None))
    def recognize(items, *models):
        out = []
        for t, region, frame in items:
            value = {"t": t, "region": region, "pixels": hashlib.sha256(frame.tobytes()).hexdigest()}
            out.append(SimpleNamespace(t=t, region=region, to_dict=lambda value=value: value))
        return out
    monkeypatch.setattr(read, "read_regions", recognize)
    cal, det, clf = Calibration.load("pml"), Stub("d"), Stub("c")
    hands = [{"hand": 0, "t_start": 0., "t_end": 2.}]
    initial = [Interval("hand:TL", 0., 1., 3, True, 0., 0.)]
    final = [Interval("hand:TL", .5, 2., 4, True, 0., 0.)]
    read.run_read(source, cal, tmp_path / "incremental", hands, initial, det, clf)
    read.run_read(source, cal, tmp_path / "incremental", hands, final, det, clf)
    read.run_read(source, cal, tmp_path / "forced", hands, final, det, clf, force=True)
    assert read.load_reads(tmp_path / "incremental", 0) == read.load_reads(tmp_path / "forced", 0)


def test_dense_provenance_and_interrupted_cache_recovery(tmp_path, monkeypatch):
    identity = read.source_identity
    seen = fake_pipeline(monkeypatch)
    monkeypatch.setattr(read, "source_identity", identity)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"source-a")
    cal, det, clf = Calibration.load("pml"), Stub("d"), Stub("c")
    clf.classes, clf.T = ["1m"], 1.
    def run():
        return read.dense_reads(source, cal, tmp_path, det, clf, 0., 1., ["hand:TL"])
    first = run()
    cache = next((tmp_path / "dense").glob("*.json"))
    cache.write_text('{"interrupted')
    seen.clear()
    assert run() == first and seen
    count = len(list((tmp_path / "dense").glob("*.json")))
    stamp = source.stat()
    source.write_bytes(b"source-b")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    identity(source, refresh=True)  # stage entry refreshes content before dense windows
    run()
    clf.T = 2.
    run()
    monkeypatch.setattr(read, "PREPROCESSING", "new-pipeline")
    run()
    assert len(list((tmp_path / "dense").glob("*.json"))) == count + 3


def test_interrupted_region_publication_cannot_authenticate_mixed_models(tmp_path, monkeypatch):
    seen = fake_pipeline(monkeypatch)
    cal, clf = Calibration.load("pml"), Stub("c")
    hands = [{"hand": 0, "t_start": 0., "t_end": 1.}]
    ivs = [Interval(r, 0., 1., 3, True, 0., 0.) for r in REGIONS]
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("old"), clf)
    publish = read._atomic_text
    calls = []
    def interrupted(path, content):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk unavailable")
        publish(path, content)
    monkeypatch.setattr(read, "_atomic_text", interrupted)
    with pytest.raises(OSError, match="disk unavailable"):
        read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("new"), clf)
    assert not (tmp_path / "reads/00/done.json").exists()
    monkeypatch.setattr(read, "_atomic_text", publish)
    seen.clear()
    read.run_read("v.mp4", cal, tmp_path, hands, ivs, Stub("new"), clf)
    assert len(seen) == 3 * len(REGIONS)
