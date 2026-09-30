"""Browser launcher and scriptable pipeline commands.

    video2tenhou web                                 # complete browser workflow
    video2tenhou download <video-url> <out.mp4> [--start HH:MM:SS --end HH:MM:SS]
    video2tenhou trim <video> <out.mp4> [--start HH:MM:SS --end HH:MM:SS]
    video2tenhou calib fit <video> [--calib pml] [--keep overhead hand ...] [--force]
    video2tenhou calib check <video> [--t <seconds> ...] [--calib pml] [--out work/calib]
    video2tenhou convert <video> --game <id> [--game <id>] [--out out] [--calib pml]

A new video starts with `calib fit`: the geometry of the composite is
measured and checked before tile recognition, because later stages read
the crops it defines. Calibration samples visible table tiles independently
of broadcast overlay text (DESIGN.md 4.2a).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .layout import fit_path


def cmd_download(a):
    """Download a broadcast using the source and optional clip bounds from argparse."""
    from . import video

    video.download(a.url, a.out, a.start, a.end)


def cmd_trim(a):
    """Create a separate recording for the selected local time range."""
    from . import video

    video.trim(a.source, a.out, a.start, a.end)


def _work_dir(a) -> Path:
    return Path(a.work) / Path(a.video).stem


def _print_fit(cal, video):
    if cal.fit:
        oh = cal.fit.get("overhead") or {}
        print(
            f"  fit {fit_path(video)}: overhead centre {oh.get('center')} angle {oh.get('angle')} "
            f"scale {oh.get('scale')} (match {oh.get('iou')})"
        )
    else:
        print(
            f"  NO FIT for this video: the layout's own numbers are used, which are right only for the video "
            f"it was drawn on. Run `video2tenhou calib fit {video}`."
        )


def cmd_calib_fit(a):
    """Measure per-recording geometry and fail when borders cut detected tiles."""
    from . import calibfit
    from .layout import Calibration
    from .perception.detector import Detector

    work = _work_dir(a)
    cal = Calibration.load(a.calib, a.video)
    det = None if a.no_models else Detector()
    calibfit.run_fit(
        Path(a.video), cal, work, det=det, force=a.force, keep=set(a.keep or [])
    )
    cal = Calibration.load(a.calib, a.video)
    print(f"[0 fit] {a.video}")
    _print_fit(cal, a.video)
    for c in ("TL", "TR", "BL", "BR"):
        print(f"  hand {c}: roll {cal.roll(c):+.1f} deg")
    if det is not None:
        checks = calibfit.check_all(Path(a.video), cal, det, work)
        _report_checks(checks)
    else:
        checks = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"fit_{Path(a.video).stem}.jpg"
    cv2.imwrite(
        str(p),
        calibfit.fit_sheet(Path(a.video), cal, work, checks),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    print(f"  {p}")
    if any(not c.ok for c in checks):
        print(
            "  Fix the failing regions: video2tenhou web -> recording Settings -> Calibration"
        )
        raise SystemExit(1)


def _report_checks(checks) -> bool:
    print("  border check (a region's border must not cut a tile):")
    for c in sorted(checks, key=lambda c: c.region):
        print(c.line())
    return all(c.ok for c in checks)


def cmd_calib_check(a):
    """Validate existing geometry and optionally render time-specific contact sheets."""
    from . import calibfit, video
    from .layout import Calibration, contact_sheet

    work = _work_dir(a)
    cal = Calibration.load(a.calib, a.video)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[calib check] {a.video} with layout {a.calib}")
    _print_fit(cal, a.video)
    ok = True
    if not a.no_models:
        from .perception.detector import Detector

        checks = calibfit.check_all(Path(a.video), cal, Detector(), work)
        ok = _report_checks(checks)
        p = out / f"fit_{Path(a.video).stem}.jpg"
        cv2.imwrite(
            str(p),
            calibfit.fit_sheet(Path(a.video), cal, work, checks),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        print(f"  {p}")
    for t in a.t or []:
        frame = video.frame_at(a.video, t)
        p = out / f"calib_{Path(a.video).stem}_{int(t)}.jpg"
        cv2.imwrite(str(p), contact_sheet(frame, cal), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {p}")
    if not ok:
        raise SystemExit(1)


def cmd_convert(a):
    """Run cached stages through export, retaining geometry and site-record gates."""
    import json

    from . import record, timeline
    from .layout import Calibration

    cal = Calibration.load(a.calib, a.video)
    work = Path(a.work) / Path(a.video).stem
    work.mkdir(parents=True, exist_ok=True)
    _require_fit(a, cal)
    rec = work / "record.json"
    if rec.exists() and not a.force:
        games = [record.from_dict(d) for d in json.load(open(rec, encoding="utf-8"))]
        if [g.id for g in games] != a.game:
            games = None
    else:
        games = None
    if games is None:
        games = [record.fetch_game(g) for g in a.game]
        json.dump(
            [record.to_dict(g) for g in games],
            open(rec, "w", encoding="utf-8"),
            indent=1,
            ensure_ascii=False,
        )
    _gate(a, cal, work)
    print(f"[1 header] table clearings and scoremj: {a.video} -> {work}")
    entries, problems = timeline.run_header(a.video, cal, games, work, force=a.force)
    for e in entries:
        print(
            f"  hand {e['hand']:2d} game {e['game']} {e['kyoku']}/{e['honba']} sticks {e['sticks']} "
            f"{e['t_start']:7.0f}-{e['t_end']:7.0f} s  winds {e['corner_wind']}  site {e['site_index']}"
        )
    if problems:
        print("PROBLEMS (table timing and site record disagree):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print(f"  {len(entries)} hands agree with the site record")
    if a.stop == "header":
        return
    from . import calm

    print("[2 calm]")
    # Stage 1 already refreshed calm evidence when --force was requested.
    ivs = calm.run_calm(a.video, cal, work, force=False)
    for name, s in calm.summary(ivs).items():
        print(
            f"  {name:8s} calm {s['calm_fraction']:.0%} in {s['calm_intervals']} intervals, median {s['median_calm_s']} s"
        )
    if a.stop == "calm":
        return
    from . import observe, read
    from .perception.classifier import Classifier
    from .perception.detector import Detector

    print("[3 read]")
    det, clf = Detector(), Classifier()
    if a.force or a.redo == "read":
        read.clear_dense(work)  # dense reads start over with the calm reads
    st, touched = read.run_read(
        a.video,
        cal,
        work,
        entries,
        ivs,
        det,
        clf,
        force=a.force or a.redo == "read",
        reread=set(a.reread or []),
    )
    print(f"  {st}" + (f"  new readings in hands {sorted(touched)}" if touched else ""))
    if a.stop == "read":
        return
    print("[4 observe]")
    # a hand with new readings is observed (and decoded) again even when nothing else is forced
    st = observe.run_observe(
        work,
        entries,
        ivs,
        force=a.force or a.redo in ("read", "observe") or bool(a.reread),
        touched=touched,
        policy=det.evidence_policy,
    )
    touched = set(touched) | set(st.get("changed_hands", []))
    print(f"  {st}")
    if a.stop == "observe":
        return
    from .engine.decode import run_decode

    print("[5 decode]")
    only = set(a.hands) if a.hands else None
    force_all = (a.force or a.redo is not None or bool(a.reread)) and not only
    redo = (only or set()) | (set() if force_all else touched)
    if redo:
        run_decode(
            work,
            entries,
            games,
            force=True,
            only=redo,
            video_path=Path(a.video),
            cal=cal,
            evidence_policy=det.evidence_policy,
        )
    decodes = run_decode(
        work,
        entries,
        games,
        force=force_all,
        video_path=Path(a.video),
        cal=cal,
        evidence_policy=det.evidence_policy,
    )
    if a.stop == "decode":
        return
    print("[6 write]")
    write_outputs(
        Path(a.out) / Path(a.video).stem, games, decodes, entries, Path(a.video).stem
    )
    (work / "calibration.changed").unlink(missing_ok=True)
    (work / "inputs.changed").unlink(missing_ok=True)


def hand_status(d: dict, left_out: bool) -> str:
    """Complete (nothing open) / review (open questions) / conflict (no legal reconstruction, or a log the replayer
    rejects: nothing is written for it).
    """
    if left_out or any(i["kind"] == "conflict" for i in d["items"]):
        return "conflict"
    return "review" if d["items"] else "complete"


def write_outputs(
    out: Path, games, decodes: list[dict], entries: list[dict], title: str
) -> None:
    """g<k>.json (the logs, without the hands left out), g<k>.html (their tenhou URLs), g<k>.confidence.json,
    review.json and report.md.
    """
    import json

    from .engine.assemble import game_from_decodes
    from .engine.validation import review_artifact

    out.mkdir(parents=True, exist_ok=True)
    report, review = [], []
    for gi, game in enumerate(games):
        ds = [d for d in decodes if d["game"] == gi]
        expected = [e for e in entries if e["game"] == gi]
        g, conf, left_out = game_from_decodes(ds, entries, game, title)
        (out / f"g{gi}.json").write_text(g.dumps(), encoding="utf-8")
        (out / f"g{gi}.html").write_text(g.links_html(), encoding="utf-8")
        json.dump(
            {str(h): rows for h, rows in conf.items()},
            open(out / f"g{gi}.confidence.json", "w", encoding="utf-8"),
            indent=1,
        )
        for d in sorted(ds, key=lambda d: d["hand"]):
            entry = next(e for e in entries if e["hand"] == d["hand"])
            d = review_artifact(d, entry, left_out.get(d["hand"], []))
            items = d["items"]
            sc = d["score"]
            score = (
                "-"
                if not sc
                else (
                    "ok"
                    if sc["match"]
                    else (
                        sc["error"]
                        or f"{sc['han']}/{sc['fu']} vs {sc['site'][0]}/{sc['site'][1]}"
                    )
                )
            )
            # lost: the tiles nothing showed (draws, kan indicators), written as the rules' guess; each is an open
            # question until the reviewer supplies it or says Can't tell
            lost = sum(1 for r in d.get("confidence", []) if r.get("lost"))
            report.append(
                f"| {d['hand']} | {gi} | {d['kyoku']}/{d['honba']} | {hand_status(d, d['hand'] in left_out)} | "
                f"{'no' if d['hand'] in left_out else 'yes'} | {d['stats']['turns']} | {d['stats']['calls']} | "
                f"{len(items)} | {lost} | {score} |"
            )
            for it in items:
                # an item may carry its own `hand` (a reconstructed hand of tiles): it must not overwrite the hand number
                row = dict(it)
                if isinstance(row.get("hand"), list):
                    row["tiles"] = row.pop("hand")
                review.append({**row, "hand": d["hand"]})
        decoded_ids = {d["hand"] for d in ds}
        for entry in expected:
            if entry["hand"] in decoded_ids:
                continue
            reason = "No current reconstruction is available. Analyze this recording again to rebuild missing or outdated evidence."
            review.append({"kind": "conflict", "hand": entry["hand"], "text": reason})
            report.append(
                f"| {entry['hand']} | {gi} | {entry['kyoku']}/{entry['honba']} | conflict | no | 0 | 0 | 1 | 0 | - |"
            )
        report.append(
            f"\nhanchan {gi}: {len(g.kyokus)} of {len(expected)} hands written"
        )
        for h, v in sorted(left_out.items()):
            report.append(
                f"  - hand {h} left out: {v[0]}"
                + (f" (+{len(v) - 1} more)" if len(v) > 1 else "")
            )
        report.append("")
    json.dump(
        review,
        open(out / "review.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    head = (
        "| hand | game | kyoku/honba | status | written | turns | calls | open items | lost | score |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    (out / "report.md").write_text(head + "\n".join(report) + "\n", encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))


def _require_fit(a, cal):
    """Reject missing geometry before spending time scanning a recording."""
    if not a.skip_fit_check and cal.fit is None:
        raise SystemExit(
            f"[0 fit] {a.video} has no calibration fit. The layout's numbers are the reference video's; "
            f"on any other video the ponds and meld insets land in the wrong place.\n"
            f"        Run: video2tenhou calib fit {a.video}"
        )


def _gate(a, cal, work):
    """Validate tile geometry after alignment and before any tile-reading stage.

    Independent table samples check region geometry; border failures stop
    processing because later tile answers cannot repair clipped image evidence.
    """
    from . import calibfit

    _require_fit(a, cal)
    if a.skip_fit_check:
        return
    from .perception.detector import Detector

    print("[0 fit] checking this video's geometry")
    _print_fit(cal, a.video)
    checks = calibfit.check_all(Path(a.video), cal, Detector(), work)
    if not _report_checks(checks):
        bad = ", ".join(c.region for c in checks if not c.ok)
        raise SystemExit(
            f"        {bad} would be read from a crop that cuts tiles: fix the fit first "
            "(video2tenhou web -> Settings -> Calibration), or pass --skip-fit-check to run anyway."
        )


def cmd_web(a):
    """Launch the complete local browser workflow using the selected data directory."""
    import os
    import subprocess

    from .paths import DATA_DIR

    root = Path(a.data).expanduser().resolve()
    if root != DATA_DIR:
        # Workspace paths are import-time constants. Set the environment before
        # importing the pipeline in a child, rather than partially rebinding them.
        command = [
            sys.executable,
            "-m",
            "video2tenhou.cli",
            "web",
            "--data",
            str(root),
            "--port",
            str(a.port),
        ]
        if a.no_browser:
            command.append("--no-browser")
        return subprocess.call(
            command, env={**os.environ, "VIDEO2TENHOU_HOME": str(root)}
        )
    import torch

    from .perception.device import select_device

    device = select_device()
    os.environ["VIDEO2TENHOU_DEVICE"] = device
    description = (
        torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else "CPU (processing will be slower)"
    )
    print(
        f"Runtime ready: {description}; PyTorch {torch.__version__}, CUDA {torch.version.cuda or 'none'}.",
        flush=True,
    )
    from .tool.server import serve_workspace

    serve_workspace(root, a.port, open_browser=not a.no_browser)


def main(argv=None):
    """Dispatch CLI commands; no arguments opens the browser workspace."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["web"]
    ap = argparse.ArgumentParser(
        prog="video2tenhou",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    from .paths import DATA_DIR

    w = sub.add_parser("web", help="open the complete browser workflow")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument(
        "--data",
        default=str(DATA_DIR),
        help="directory containing recordings, models, labels and work",
    )
    w.add_argument(
        "--no-browser",
        action="store_true",
        help="serve without opening a browser window",
    )
    w.set_defaults(fn=cmd_web)

    d = sub.add_parser("download")
    d.add_argument("url")
    d.add_argument("out")
    d.add_argument("--start")
    d.add_argument("--end")
    d.set_defaults(fn=cmd_download)

    t = sub.add_parser(
        "trim", help="create a separate MP4 for a local recording's time range"
    )
    t.add_argument("source")
    t.add_argument("out")
    t.add_argument("--start")
    t.add_argument("--end")
    t.set_defaults(fn=cmd_trim)

    c = sub.add_parser("calib")
    csub = c.add_subparsers(dest="sub", required=True)
    cf = csub.add_parser(
        "fit", help="measure this video's geometry and write labels/<video>/calib.json"
    )
    cf.add_argument("video")
    cf.add_argument("--calib", default="pml")
    cf.add_argument("--out", default=str(DATA_DIR / "work" / "calib"))
    cf.add_argument("--work", default=str(DATA_DIR / "work"))
    cf.add_argument(
        "--keep",
        nargs="*",
        choices=["overhead", "hand", "meld", "cam"],
        help="parts of an existing fit to leave alone",
    )
    cf.add_argument("--force", action="store_true", help="rebuild the table plate")
    cf.add_argument(
        "--no-models",
        action="store_true",
        help="fit the overhead only; no detector, no border check",
    )
    cf.set_defaults(fn=cmd_calib_fit)
    cc = csub.add_parser(
        "check", help="does this video's geometry fit it? (exit 1 when not)"
    )
    cc.add_argument("video")
    cc.add_argument(
        "--t",
        type=float,
        action="append",
        help="also write a contact sheet at this time",
    )
    cc.add_argument("--calib", default="pml")
    cc.add_argument("--out", default=str(DATA_DIR / "work" / "calib"))
    cc.add_argument("--work", default=str(DATA_DIR / "work"))
    cc.add_argument(
        "--no-models",
        action="store_true",
        help="print the fit without running the detector",
    )
    cc.set_defaults(fn=cmd_calib_check)

    v = sub.add_parser("convert")
    v.add_argument("video")
    v.add_argument("--game", type=int, action="append", required=True)
    v.add_argument("--out", default=str(DATA_DIR / "out"))
    v.add_argument("--calib", default="pml")
    v.add_argument("--work", default=str(DATA_DIR / "work"))
    v.add_argument("--force", action="store_true", help="recompute cached stages")
    v.add_argument(
        "--stop",
        choices=["header", "calm", "read", "observe", "decode"],
        help="stop after this stage",
    )
    v.add_argument(
        "--redo",
        choices=["read", "observe", "decode"],
        help="recompute this stage even if cached",
    )
    v.add_argument(
        "--reread",
        choices=["pond", "hand", "meld"],
        nargs="*",
        help="re-read these region kinds from scratch (after a calibration change), then observe and decode again",
    )
    v.add_argument("--hands", type=int, nargs="*", help="re-decode only these hands")
    v.add_argument(
        "--skip-fit-check",
        action="store_true",
        help="run even when the geometry check fails (the log will be wrong)",
    )
    v.set_defaults(fn=cmd_convert)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
