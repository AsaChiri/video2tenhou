# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Browser launcher and scriptable pipeline commands.

    video2tenhou web                                 # complete browser workflow
    video2tenhou download <video-url> <out.mp4> [--start HH:MM:SS --end HH:MM:SS]
    video2tenhou trim <video> <out.mp4> [--start HH:MM:SS --end HH:MM:SS]
    video2tenhou calib fit <video> [--calib pml] [--keep overhead hand ...] [--force]
    video2tenhou calib check <video> [--t <seconds> ...] [--calib pml]
    video2tenhou convert <video> --game <id> [--game <id>] [--out out] [--calib pml]
    video2tenhou rebuild <video> [--hands <i> ...] [--force]

A new video starts with `calib fit`: the geometry of the composite is measured and
checked before tile recognition, because later stages read the crops it defines.
Calibration samples visible table tiles independently of broadcast overlay text
(DESIGN.md 4.2a).

Every command except `web` logs progress on stderr and ends with one JSON document on
stdout when it has an outcome to report: `{"result": ...}` and/or `{"error": "..."}`.
An error message tells the user what to do; tracebacks stay in the log.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from video2tenhou.files import atomic_write_json
from video2tenhou.logging_setup import RESULT, command_logging

from . import paths

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

    import numpy as np

    from video2tenhou.calibfit import RegionCheck
    from video2tenhou.layout import Calibration
    from video2tenhou.perception.evidence_policy import EvidencePolicy
    from video2tenhou.read import ReadContext
    from video2tenhou.record import Game


LOGGER = logging.getLogger("video2tenhou.cli")


class CommandFailed(SystemExit):
    """Stop a command with a message for the user and an optional partial result."""

    def __init__(self, message: str, result: object = None) -> None:
        """Keep the message as the exit status, as SystemExit does for text."""
        super().__init__(message)
        self.result = result


def cmd_download(a: Namespace) -> None:
    """Download a broadcast using the source and optional clip bounds from argparse."""
    from . import video  # noqa: PLC0415

    try:
        video.download(a.url, a.out, a.start, a.end)
    except subprocess.CalledProcessError as error:
        raise CommandFailed(
            "The download failed. Check the video URL and the connection, then retry."
        ) from error


def cmd_trim(a: Namespace) -> None:
    """Create a separate recording for the selected local time range."""
    from . import video  # noqa: PLC0415

    try:
        video.trim(a.source, a.out, a.start, a.end)
    except subprocess.CalledProcessError as error:
        raise CommandFailed(
            "The selected time range could not be cut from the recording."
        ) from error


def _work_dir(a: Namespace) -> Path:
    return Path(a.work) / Path(a.video).stem


def _print_fit(cal: Calibration, video: str | Path) -> None:
    from .layout import fit_path  # noqa: PLC0415

    if cal.fit:
        oh = cal.fit.get("overhead") or {}
        LOGGER.info(
            "  fit %s: overhead centre %s angle %s scale %s (match %s)",
            fit_path(video),
            oh.get("center"),
            oh.get("angle"),
            oh.get("scale"),
            oh.get("iou"),
        )
    else:
        LOGGER.info(
            "  NO FIT for this video: the layout's own numbers are used, which "
            "are right only for the video it was drawn on. "
            "Run `video2tenhou calib fit %s`.",
            video,
        )


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    import cv2  # noqa: PLC0415

    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    LOGGER.info("  %s", path)


def cmd_calib_fit(a: Namespace) -> dict:
    """Measure per-recording geometry and fail when borders cut detected tiles."""
    from . import calibfit  # noqa: PLC0415
    from .layout import Calibration  # noqa: PLC0415
    from .perception.detector import Detector  # noqa: PLC0415

    work = _work_dir(a)
    cal = Calibration.load(a.calib, a.video)
    det = None if a.no_models else Detector()
    recording = calibfit.Recording(Path(a.video), det)  # shared by fit and check
    calibfit.run_fit(recording, cal, work, force=a.force, keep=set(a.keep or []))
    cal = Calibration.load(a.calib, a.video)
    LOGGER.info("[0 fit] %s", a.video)
    _print_fit(cal, a.video)
    for c in ("TL", "TR", "BL", "BR"):
        LOGGER.info("  hand %s: roll %+.1f deg", c, cal.roll(c))
    checks = (
        []
        if det is None
        else calibfit.check_all(Path(a.video), cal, det, work, recording=recording)
    )
    _report_checks(checks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_jpeg(
        out / f"fit_{Path(a.video).stem}.jpg",
        calibfit.fit_sheet(Path(a.video), cal, work, checks),
        92,
    )
    return _checks_outcome(checks)


def _report_checks(checks: list[RegionCheck]) -> None:
    if checks:
        LOGGER.info("  border check (a region's border must not cut a tile):")
    for c in sorted(checks, key=lambda c: c.region):
        LOGGER.info("%s", c.line())


def _border_problem(check: RegionCheck) -> str:
    if check.region == "overhead":
        return "does not match the table"
    if check.foreign:
        return "reads another pond"
    return "cuts tiles" if check.cut else "sees no tiles"


def border_failure(checks: list[RegionCheck]) -> str | None:
    """Describe failing border checks as one instruction, or None when all pass."""
    failing = [c for c in sorted(checks, key=lambda c: c.region) if not c.ok]
    if not failing:
        return None
    problems = "; ".join(f"{c.region} {_border_problem(c)}" for c in failing)
    return f"Adjust the table borders in Calibration: {problems}."


def _checks_outcome(checks: list[RegionCheck]) -> dict:
    """Return every region's check, stopping the command when one fails."""
    result = {
        c.region: {"level": c.level, "held": c.held, "cut": c.cut, "note": c.note}
        for c in checks
    }
    failure = border_failure(checks)
    if failure:
        raise CommandFailed(failure, result)
    return result


def cmd_calib_check(a: Namespace) -> dict:
    """Validate existing geometry and optionally render time-specific contact sheets."""
    from . import calibfit, video  # noqa: PLC0415
    from .layout import Calibration, contact_sheet  # noqa: PLC0415

    work = _work_dir(a)
    cal = Calibration.load(a.calib, a.video)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[calib check] %s with layout %s", a.video, a.calib)
    _print_fit(cal, a.video)
    checks = []
    if not a.no_models:
        from .perception.detector import Detector  # noqa: PLC0415

        checks = calibfit.check_all(Path(a.video), cal, Detector(), work)
        _report_checks(checks)
        _write_jpeg(
            out / f"fit_{Path(a.video).stem}.jpg",
            calibfit.fit_sheet(Path(a.video), cal, work, checks),
            92,
        )
    for t in a.t or []:
        frame = video.frame_at(a.video, t)
        _write_jpeg(
            out / f"calib_{Path(a.video).stem}_{int(t)}.jpg",
            contact_sheet(frame, cal),
            90,
        )
    return _checks_outcome(checks)


def _convert_header(
    a: Namespace, context: ReadContext
) -> tuple[list[Game], list[dict]]:
    """Bind the recording's authoritative games before accepting hand timing."""
    from . import record, timeline  # noqa: PLC0415

    work = context.work
    rec = work / "record.json"
    if rec.exists() and not a.force:
        games = [
            record.from_dict(d) for d in json.loads(rec.read_text(encoding="utf-8"))
        ]
        if [g.id for g in games] != a.game:
            games = None
    else:
        games = None
    if games is None:
        games = [record.fetch_game(g) for g in a.game]
        atomic_write_json(
            rec,
            [record.to_dict(g) for g in games],
            indent=1,
        )
    _gate(a, context)
    LOGGER.info("[1 header] table clearings and scoremj: %s -> %s", a.video, work)
    entries, problems = timeline.run_header(context, games, force=a.force)
    for e in entries:
        LOGGER.info(
            "  hand %2d game %s %s/%s sticks %s %7.0f-%7.0f s  winds %s  site %s",
            e["hand"],
            e["game"],
            e["kyoku"],
            e["honba"],
            e["sticks"],
            e["t_start"],
            e["t_end"],
            e["corner_wind"],
            e["site_index"],
        )
    if problems:
        LOGGER.info("PROBLEMS (table timing and site record disagree):")
        for p in problems:
            LOGGER.info("  - %s", p)
        raise CommandFailed(
            "The hands found in the video do not match the score record. Check the "
            "game IDs and the recording's time range in Settings."
        )
    LOGGER.info("  %s hands agree with the site record", len(entries))
    return games, entries


def cmd_convert(a: Namespace) -> None:
    """Run cached stages through export, retaining geometry and site-record gates."""
    from .layout import Calibration  # noqa: PLC0415

    cal = Calibration.load(a.calib, a.video)
    work = Path(a.work) / Path(a.video).stem
    work.mkdir(parents=True, exist_ok=True)
    _require_fit(a, cal)
    recognized = _recognize(a, cal, work)
    if recognized is None:
        return
    games, entries, touched, policy = recognized
    from .engine.decode import run_decode  # noqa: PLC0415
    from .export import write_outputs  # noqa: PLC0415

    LOGGER.info("[5 decode]")
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
            evidence_policy=policy,
        )
    decodes = run_decode(
        work,
        entries,
        games,
        force=force_all,
        video_path=Path(a.video),
        cal=cal,
        evidence_policy=policy,
    )
    if a.stop == "decode":
        return
    write_outputs(
        Path(a.out) / Path(a.video).stem,
        games,
        decodes,
        entries,
        decode_dir=work / "decode",
    )
    (work / "calibration.changed").unlink(missing_ok=True)
    (work / "inputs.changed").unlink(missing_ok=True)


def _recognize(
    a: Namespace, cal: Calibration, work: Path
) -> tuple[list[Game], list[dict], set[int], EvidencePolicy] | None:
    """Gate, time, read and vote the recording with one detector/classifier pair.

    Returns None when `--stop` ends the conversion before reconstruction. The models
    are local to this stage, so they are released before reconstruction loads its
    own only if a hand rereads the video; one pair is resident at a time.
    """
    from . import calm, observe, read  # noqa: PLC0415
    from .perception.classifier import Classifier  # noqa: PLC0415
    from .perception.detector import Detector  # noqa: PLC0415

    det = Detector()
    context = read.ReadContext(a.video, cal, work, det, Classifier())
    games, entries = _convert_header(a, context)
    if a.stop == "header":
        return None
    LOGGER.info("[2 calm]")
    # Stage 1 already refreshed calm evidence when --force was requested.
    ivs = calm.run_calm(a.video, cal, work, force=False)
    for name, s in calm.summary(ivs).items():
        LOGGER.info(
            "  %-8s calm %s in %s intervals, median %s s",
            name,
            format(s["calm_fraction"], ".0%"),
            s["calm_intervals"],
            s["median_calm_s"],
        )
    if a.stop == "calm":
        return None
    LOGGER.info("[3 read]")
    if a.force or a.redo == "read":
        read.clear_dense(work)  # dense reads start over with the calm reads
    st, touched = read.run_read(
        context,
        entries,
        ivs,
        force=a.force or a.redo == "read",
        reread=set(a.reread or []),
    )
    LOGGER.info(
        "  %s%s",
        st,
        f"  new readings in hands {sorted(touched)}" if touched else "",
    )
    if a.stop == "read":
        return None
    LOGGER.info("[4 observe]")
    # A hand with new readings is voted and decoded again even when nothing else is
    # forced.
    st = observe.run_observe(
        work,
        entries,
        ivs,
        force=a.force or a.redo in ("read", "observe") or bool(a.reread),
        touched=touched,
        policy=det.evidence_policy,
    )
    LOGGER.info("  %s", st)
    if a.stop == "observe":
        return None
    return (
        games,
        entries,
        set(touched) | set(st.get("changed_hands", [])),
        det.evidence_policy,
    )


def cmd_rebuild(a: Namespace) -> None:
    """Reconstruct hands with the saved answers, then rewrite every export.

    Without `--force`, a hand whose decode already matches its evidence, record and
    answers is reused. Evidence is never re-read: changed geometry or project inputs
    require `convert` first.
    """
    from . import record  # noqa: PLC0415
    from .engine.decode import DECODER_VERSION, run_decode  # noqa: PLC0415
    from .export import write_outputs  # noqa: PLC0415
    from .layout import Calibration  # noqa: PLC0415

    work = _work_dir(a)
    if any((work / m).exists() for m in ("calibration.changed", "inputs.changed")):
        raise CommandFailed(
            "The table geometry or project settings changed. Choose Analyze "
            "recording to refresh the readings before updating hands."
        )
    hands = json.loads((work / "hands.json").read_text(encoding="utf-8"))
    games = [
        record.from_dict(row)
        for row in json.loads((work / "record.json").read_text(encoding="utf-8"))
    ]
    known = {h["hand"] for h in hands}
    selected = set(a.hands) if a.hands else known
    if not selected <= known:
        raise CommandFailed("Unknown hand selected for rebuilding.")
    run_decode(
        work,
        hands,
        games,
        force=a.force,
        only=selected,
        video_path=Path(a.video),
        cal=Calibration.load(a.calib, a.video),
    )
    # A decode by another decoder version is stale and never exported.
    saved = [work / "decode" / f"{h['hand']:02d}.json" for h in hands]
    decodes = [
        decoded
        for path in saved
        if path.exists()
        and (decoded := json.loads(path.read_text(encoding="utf-8"))).get(
            "decoder_version"
        )
        == DECODER_VERSION
    ]
    write_outputs(
        Path(a.out) / Path(a.video).stem,
        games,
        decodes,
        hands,
        decode_dir=work / "decode",
    )
    current = {d["hand"] for d in decodes}
    unread = sorted(h + 1 for h in selected if h not in current)
    if unread:
        names = ", ".join(map(str, unread))
        raise CommandFailed(
            (
                f"Hand {names} has no tile readings."
                if len(unread) == 1
                else f"Hands {names} have no tile readings."
            )
            + " Choose Analyze recording to read the video again."
        )


def _require_fit(a: Namespace, cal: Calibration) -> None:
    """Reject missing geometry before spending time scanning a recording."""
    if not a.skip_fit_check and cal.fit is None:
        LOGGER.info(
            "[0 fit] %s has no calibration fit; the layout's numbers belong to its "
            "reference video. Run: video2tenhou calib fit %s",
            a.video,
            a.video,
        )
        raise CommandFailed(
            "This recording has no table calibration. Prepare the recording first."
        )


def _gate(a: Namespace, context: ReadContext) -> None:
    """Validate tile geometry after alignment and before any tile-reading stage.

    Independent table samples check region geometry; border failures stop
    processing because later tile answers cannot repair clipped image evidence.
    """
    from . import calibfit  # noqa: PLC0415

    cal = context.calibration
    _require_fit(a, cal)
    if a.skip_fit_check:
        return
    LOGGER.info("[0 fit] checking this video's geometry")
    _print_fit(cal, a.video)
    checks = calibfit.check_all(Path(a.video), cal, context.detector, context.work)
    _report_checks(checks)
    failure = border_failure(checks)
    if failure:
        LOGGER.info("  Pass --skip-fit-check to convert anyway.")
        raise CommandFailed(failure)


def cmd_web(a: Namespace) -> int | None:
    """Launch the complete local browser workflow using the selected data directory."""
    root = Path(a.data).expanduser().resolve()
    if root != paths.DATA_DIR:
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
        return subprocess.call(  # noqa: S603
            command, env={**os.environ, "VIDEO2TENHOU_HOME": str(root)}
        )
    import torch  # noqa: PLC0415

    from .perception.device import select_device  # noqa: PLC0415

    device = select_device()
    os.environ["VIDEO2TENHOU_DEVICE"] = device
    description = (
        torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else "CPU (processing will be slower)"
    )
    LOGGER.info(
        "Runtime ready: %s; PyTorch %s, CUDA %s.",
        description,
        torch.__version__,
        torch.version.cuda or "none",
    )
    from .tool.server import serve_workspace  # noqa: PLC0415

    serve_workspace(root, a.port, open_browser=not a.no_browser)
    return None


def run_command(a: Namespace) -> int:
    """Run one command and report its outcome as the final JSON line on stdout.

    A `SystemExit` with a message is a failure the user can act on; any other
    exception is logged with its traceback and reported as unexpected.
    """
    try:
        result = a.fn(a)
    except SystemExit as stop:
        if not isinstance(stop.code, str):
            raise
        outcome = {"error": stop.code}
        if isinstance(stop, CommandFailed) and stop.result is not None:
            outcome["result"] = stop.result
        RESULT.info("%s", json.dumps(outcome))
        return 1
    except Exception as error:
        LOGGER.exception("%s failed", a.cmd)
        detail = str(error) or type(error).__name__
        outcome = {"error": f"Unexpected error: {detail}", "unexpected": True}
        RESULT.info("%s", json.dumps(outcome))
        return 1
    if result is not None:
        RESULT.info("%s", json.dumps({"result": result}))
    return 0


def _calibration_commands(parser: argparse.ArgumentParser) -> None:
    """Define calibration commands and their shared workspace defaults."""
    csub = parser.add_subparsers(dest="sub", required=True)
    cf = csub.add_parser(
        "fit", help="measure this video's geometry and write labels/<video>/calib.json"
    )
    cf.add_argument("video")
    cf.add_argument("--calib", default="pml")
    cf.add_argument("--out", default=str(paths.DATA_DIR / "work" / "calib"))
    cf.add_argument("--work", default=str(paths.DATA_DIR / "work"))
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
    cc.add_argument("--out", default=str(paths.DATA_DIR / "work" / "calib"))
    cc.add_argument("--work", default=str(paths.DATA_DIR / "work"))
    cc.add_argument(
        "--no-models",
        action="store_true",
        help="print the fit without running the detector",
    )
    cc.set_defaults(fn=cmd_calib_check)


def _stage_commands(add_parser: Callable[..., argparse.ArgumentParser]) -> None:
    """Define conversion and review rebuild commands."""
    v = add_parser("convert")
    v.add_argument("video")
    v.add_argument("--game", type=int, action="append", required=True)
    v.add_argument("--out", default=str(paths.DATA_DIR / "out"))
    v.add_argument("--calib", default="pml")
    v.add_argument("--work", default=str(paths.DATA_DIR / "work"))
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
        help=(
            "re-read these region kinds from scratch (after a calibration change), "
            "then observe and decode again"
        ),
    )
    v.add_argument("--hands", type=int, nargs="*", help="re-decode only these hands")
    v.add_argument(
        "--skip-fit-check",
        action="store_true",
        help="run even when the geometry check fails (the log will be wrong)",
    )
    v.set_defaults(fn=cmd_convert)
    r = add_parser(
        "rebuild", help="reconstruct hands with saved answers and rewrite the exports"
    )
    r.add_argument("video")
    r.add_argument("--hands", type=int, nargs="*", help="these hands (default: all)")
    r.add_argument("--force", action="store_true", help="decode even if up to date")
    r.add_argument("--out", default=str(paths.DATA_DIR / "out"))
    r.add_argument("--calib", default="pml")
    r.add_argument("--work", default=str(paths.DATA_DIR / "work"))
    r.set_defaults(fn=cmd_rebuild)


@command_logging
def main(argv: list[str] | None = None) -> int | None:
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

    w = sub.add_parser("web", help="open the complete browser workflow")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument(
        "--data",
        default=str(paths.DATA_DIR),
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
    _calibration_commands(c)
    _stage_commands(sub.add_parser)

    a = ap.parse_args(argv)
    return a.fn(a) if a.cmd == "web" else run_command(a)


if __name__ == "__main__":
    sys.exit(main())
