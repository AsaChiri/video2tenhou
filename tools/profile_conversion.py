# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Profile one complete ``convert`` run by stage, hand, solver search and dense read.

Usage: uv run python tools/profile_conversion.py --profile-output work/profile
video.mp4 --game 123 --work work/cold --out out/cold

Pipeline functions are wrapped with unittest.mock for this run only, so the
package carries no profiling hooks. Caches are never removed; the summary records
whether the recording's work and output directories started empty. Spans are
inclusive and can overlap: nested or concurrent times must not be added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import cv2
import torch
from ortools.sat.python.cp_model import CpSolver

from video2tenhou import calm, cli, export, observe, read, record, timeline, video
from video2tenhou.commands import executable
from video2tenhou.engine import decode, dense, solver
from video2tenhou.files import atomic_write_text
from video2tenhou.layout import Calibration
from video2tenhou.perception.classifier import Classifier
from video2tenhou.perception.detector import Detector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ortools.sat.python.cp_model import (
        CpModel,
        CpSolverSolutionCallback,
        CpSolverStatus,
    )

STAGES = (
    (cli, "_gate"),
    (record, "fetch_game"),
    (timeline, "run_header"),
    (calm, "run_calm"),
    (read, "run_read"),
    (observe, "run_observe"),
    (decode, "run_decode"),
    (export, "write_outputs"),
)
OPERATIONS = (
    (Detector, "predict"),
    (Detector, "predict_batch"),
    (Classifier, "posteriors"),
    (video, "frame_at"),
    (timeline, "read_pond_counts"),
    (solver.HandModel, "build"),
    (solver.HandModel, "solve"),
    (solver.HandModel, "certify"),
    (dense, "dense_reads"),
    (dense, "dense_pond_reads"),
)
HAND = "engine.decode.decode_hand"
SOLVE = "engine.solver.HandModel.solve"
CERTIFY = "engine.solver.HandModel.certify"
SEARCH = "CpSolver.solve"
DENSE = {"engine.dense.dense_reads", "engine.dense.dense_pond_reads"}
PACKAGES = (
    "video2tenhou",
    "torch",
    "torchvision",
    "libreyolo",
    "ortools",
    "numpy",
    "opencv-python",
)
NEAR_BUDGET = 0.9


def label(owner: object, attribute: str) -> str:
    """Name a module function or class method without the package prefix."""
    name = getattr(owner, "__name__", str(owner))
    return f"{name.removeprefix('video2tenhou.')}.{attribute}"


@dataclass
class Call:
    """One completed call on the profiler's monotonic clock.

    ``cpu`` is process CPU time, so it includes concurrent worker threads.
    """

    name: str
    stage: str | None
    hand: int | None
    start: float
    wall: float = 0.0
    cpu: float = 0.0
    failed: bool = True
    status: str | None = None
    budget: float | None = None


@dataclass
class Recorder:
    """Collect calls, labelling worker calls with the active stage and hand.

    A conversion runs its stages and hands serially on the calling thread.
    """

    calls: list[Call] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    stage: str | None = None
    hand: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def span(self, name: str) -> Iterator[Call]:
        """Time a call, keeping failed calls and propagating their errors."""
        call = Call(name, self.stage, self.hand, time.perf_counter())
        cpu = time.process_time()
        try:
            yield call
            call.failed = False
        finally:
            call.wall = time.perf_counter() - call.start
            call.cpu = time.process_time() - cpu
            with self.lock:
                self.calls.append(call)

    def wrap[**P, R](
        self,
        function: Callable[P, R],
        name: str,
        *,
        stage: bool = False,
        hand: bool = False,
    ) -> Callable[P, R]:
        """Time a function; stages and hand decodes also label the calls they make."""

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if not (stage or hand):  # Worker threads must not touch the labels.
                with self.span(name):
                    return function(*args, **kwargs)
            previous = self.stage, self.hand
            if stage:
                self.stage = name
            if hand:
                entry = args[0] if args else kwargs["entry"]
                if not isinstance(entry, dict):
                    raise TypeError("A profiled hand decode requires its hand entry")
                self.hand = entry["hand"]
            try:
                with self.span(name):
                    return function(*args, **kwargs)
            finally:
                self.stage, self.hand = previous

        return wrapped

    def identify[T](self, model: type[T]) -> Callable[..., None]:
        """Time model construction and keep the loaded model's recognition identity."""
        construct = model.__init__

        @wraps(construct)
        def wrapped(instance: T, *args: object, **kwargs: object) -> None:
            with self.span(f"{model.__name__}.__init__"):
                construct(instance, *args, **kwargs)
            self.models[model.__name__] = getattr(instance, "id", "")

        return wrapped

    def solve(self) -> Callable[..., CpSolverStatus]:
        """Time every CP-SAT search with its status and configured time limit."""
        original = CpSolver.solve

        @wraps(original)
        def solve(
            instance: CpSolver,
            model: CpModel,
            solution_callback: CpSolverSolutionCallback | None = None,
        ) -> CpSolverStatus:
            with self.span(SEARCH) as call:
                status = original(instance, model, solution_callback)
                call.status = instance.status_name(status)
                call.budget = instance.parameters.max_time_in_seconds
            return status

        return solve


@contextmanager
def instrument(recorder: Recorder) -> Iterator[None]:
    """Patch the pipeline's stage, hand, model and search entry points temporarily."""
    with ExitStack() as patches:
        for owner, attribute in STAGES:
            wrapper = recorder.wrap(
                getattr(owner, attribute), label(owner, attribute), stage=True
            )
            patches.enter_context(patch.object(owner, attribute, wrapper))
        for owner, attribute in OPERATIONS:
            wrapper = recorder.wrap(getattr(owner, attribute), label(owner, attribute))
            patches.enter_context(patch.object(owner, attribute, wrapper))
        hand = recorder.wrap(decode.decode_hand, HAND, hand=True)
        patches.enter_context(patch.object(decode, "decode_hand", hand))
        patches.enter_context(patch.object(CpSolver, "solve", recorder.solve()))
        for model in (Detector, Classifier):
            patches.enter_context(
                patch.object(model, "__init__", recorder.identify(model))
            )
        yield


def summarize(calls: list[Call]) -> dict:
    """Aggregate stage, per-hand and nested-operation timings.

    CPU time is reported for stages and hands only, which run one at a time.
    """
    stages = {label(owner, attribute) for owner, attribute in STAGES}
    totals: dict[tuple[str | None, str], dict] = defaultdict(
        lambda: {"calls": 0, "wall": 0.0, "cpu": 0.0, "failed": 0}
    )
    by_hand: dict[int, list[Call]] = defaultdict(list)
    for call in calls:
        row = totals[(None if call.name in stages else call.stage, call.name)]
        row["calls"] += 1
        row["wall"] += call.wall
        row["cpu"] += call.cpu
        row["failed"] += call.failed
        if call.hand is not None:
            by_hand[call.hand].append(call)
    hands = []
    for hand, hand_calls in sorted(by_hand.items()):
        searches = [call for call in hand_calls if call.name == SEARCH]
        decodes = [call for call in hand_calls if call.name == HAND]
        hands.append(
            {
                "hand": hand,
                "wall": sum(call.wall for call in decodes),
                "cpu": sum(call.cpu for call in decodes),
                "failed": any(call.failed for call in decodes),
                "dense_reads": sum(
                    call.wall for call in hand_calls if call.name in DENSE
                ),
                "solve": sum(call.wall for call in hand_calls if call.name == SOLVE),
                "certify": sum(
                    call.wall for call in hand_calls if call.name == CERTIFY
                ),
                "searches": len(searches),
                "near_budget": sum(
                    1
                    for call in searches
                    if call.budget and call.wall >= NEAR_BUDGET * call.budget
                ),
                "statuses": dict(Counter(call.status for call in searches)),
            }
        )
    return {
        "stages": [
            {"name": name, **row} for (_, name), row in totals.items() if name in stages
        ],
        "hands": hands,
        "operations": [
            {"stage": stage, "name": name, "calls": row["calls"], "wall": row["wall"]}
            for (stage, name), row in sorted(
                totals.items(), key=lambda item: -item[1]["wall"]
            )
            if name not in stages
        ],
    }


def render(summary: dict) -> str:
    """Write the stage table, per-hand solver/dense breakdown and nested work."""
    wall, cpu = summary.get("wall_seconds"), summary.get("process_cpu_seconds")
    lines = ["# Conversion profile", ""]
    if summary["failure"]:
        lines += [f"Run ended with an error: `{summary['failure']}`.", ""]
    if wall:
        cores = summary["metadata"]["logical_cpus"]
        lines += [
            (
                f"Conversion wall time {wall:.1f} s; process CPU {cpu:.1f} s "
                f"({cpu / wall:.2f} logical cores of {cores})."
            ),
            "",
        ]
    files = summary["metadata"]["initial_files"]
    lines += [
        f"Files before conversion: work {files['work']}, output {files['out']}.",
        "",
        "| Stage | Calls | Wall s | CPU s | Share | Ceiling if free |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["stages"]:
        share = row["wall"] / wall if wall else 0.0
        ceiling = f"{1 / (1 - share):.2f}x" if share < 1 else "-"
        lines.append(
            f"| {row['name']} | {row['calls']} | {row['wall']:.1f} | "
            f"{row['cpu']:.1f} | {share:.1%} | {ceiling} |"
        )
    lines += [
        "",
        "## Hands",
        "",
        (
            "Solve and certify are inclusive times of the reconstruction searches "
            "and the confidence checks. Near budget means at least 90% of a "
            "search's time limit."
        ),
        "",
        (
            "| Hand | Wall s | CPU s | Dense reads s | Solve s | Certify s | "
            "Searches | Near budget | Statuses |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines += [
        f"| {row['hand']}{' (failed)' if row['failed'] else ''} | {row['wall']:.1f} "
        f"| {row['cpu']:.1f} | {row['dense_reads']:.1f} | {row['solve']:.1f} "
        f"| {row['certify']:.1f} | {row['searches']} "
        f"| {row['near_budget']} | {row['statuses']} |"
        for row in summary["hands"]
    ]
    lines += [
        "",
        "## Nested operations",
        "",
        "Inclusive totals; nested and concurrent calls overlap.",
        "",
        "| Stage | Operation | Calls | Wall s |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {row['stage']} | {row['name']} | {row['calls']} | {row['wall']:.1f} |"
        for row in summary["operations"]
    ]
    return "\n".join(lines) + "\n"


def run_metadata(args: argparse.Namespace) -> dict:
    """Record code, library versions, input and geometry identity without hashing."""
    git = executable("git")
    revision, status = (
        subprocess.run(  # noqa: S603  fixed read-only git queries
            [git, *command],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        ).stdout.strip()
        for command in (("rev-parse", "HEAD"), ("status", "--porcelain"))
    )
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    source = Path(args.video).resolve()
    stat = source.stat()
    geometry = Calibration.load(args.calib, args.video).data
    initial_files = {
        name: sum(path.is_file() for path in (Path(directory) / source.stem).rglob("*"))
        for name, directory in (("work", args.work), ("out", args.out))
    }
    return {
        "arguments": {key: value for key, value in vars(args).items() if key != "fn"},
        "revision": revision,
        "modified_files": bool(status),
        "versions": versions,
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "video": {
            **vars(video.probe(source)),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "geometry_sha256": hashlib.sha256(
            json.dumps(geometry, sort_keys=True).encode()
        ).hexdigest(),
        "initial_files": initial_files,
    }


def main(argv: list[str] | None = None) -> int | None:
    """Profile ``convert`` and always write summary.json and report.md."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--profile-output", required=True, type=Path)
    options, convert_args = parser.parse_known_args(argv)
    if convert_args[:1] == ["convert"]:
        convert_args = convert_args[1:]
    output = options.profile_output
    output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder()
    summary: dict = {"failure": None}
    original = cli.cmd_convert

    @wraps(original)
    def profiled(args: argparse.Namespace) -> None:
        summary["metadata"] = run_metadata(args)
        with instrument(recorder):
            started, cpu = time.perf_counter(), time.process_time()
            try:
                original(args)
            except BaseException as error:
                # The CLI reports the error as its outcome line; keep it here too.
                summary["failure"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                summary["wall_seconds"] = time.perf_counter() - started
                summary["process_cpu_seconds"] = time.process_time() - cpu
                summary["threads"] = {
                    "torch": torch.get_num_threads(),
                    "torch_interop": torch.get_num_interop_threads(),
                    "opencv": cv2.getNumThreads(),
                }

    try:
        with patch.object(cli, "cmd_convert", profiled):
            return cli.main(["convert", *convert_args])
    except BaseException as error:
        summary["failure"] = summary["failure"] or f"{type(error).__name__}: {error}"
        raise
    finally:
        summary.update(summarize(recorder.calls), models=recorder.models)
        atomic_write_text(
            output / "summary.json", json.dumps(summary, indent=2, default=str)
        )
        if "metadata" in summary:
            atomic_write_text(output / "report.md", render(summary))


if __name__ == "__main__":
    main()
