# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent local recording projects and the workspace's single job slot.

Manifests contain only workflow metadata; the established work/<video>,
labels/<video> and out/<video> contracts remain the pipeline's source of
truth. One job runs at a time across all projects to bound GPU memory: preparation,
analysis, hand updates and calibration measurements share the slot. A server
restart marks an unfinished preparation or analysis interrupted rather than
silently claiming success. Every job is a `video2tenhou.cli` child process whose
final stdout line reports its outcome; HTTP handlers never execute shell strings.
Recording digests are computed off the request path and never under the lock.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from video2tenhou import cache
from video2tenhou.files import atomic_write_json, read_published_text
from video2tenhou.layout import Calibration
from video2tenhou.tenhou6 import editor_url, viewer_url
from video2tenhou.video import format_time, time_range

from .processes import ProcessCancelledError, ProcessOwner
from .review_state import ReviewState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


MAX_PROJECT_NAME_LENGTH = 120
LOG_LINES = 80
DIGEST_PEEK_SECONDS = 0.2
LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
ARTIFACT = re.compile(
    r"(?:g\d+\.(?:json|html|confidence\.json)|review\.json|report\.md)\Z"
)
GAME_LOG = re.compile(r"g(\d+)\.json\Z")
MARKERS = ("calibration.changed", "inputs.changed")
INTERRUPTED = "Interrupted when the app closed. Retry to resume from cached work."
SAVE_FAILED = (
    "Could not save the project. Check that its data folder is writable and"
    " available, then retry."
)

JobKind = Literal["prepare", "analyze", "rebuild", "fit", "check"]
LABELS: dict[str, str] = {
    "prepare": "Preparation",
    "analyze": "Analysis",
    "rebuild": "Updating hands",
    "fit": "Measuring the table",
    "check": "Checking borders",
}
BUSY: dict[str, str] = {
    "prepare": "A recording is being prepared",
    "analyze": "Analysis is running",
    "rebuild": "Hands are being updated",
    "fit": "The table is being measured",
    "check": "Borders are being checked",
}


def _progress_stage(line: str) -> str | None:
    """Translate pipeline progress into stable user actions; keep raw detail in logs."""
    stages = {
        "[0 fit]": "Checking table borders",
        "[1 header]": "Matching hands to the score record",
        "[2 calm]": "Finding clear video frames",
        "[3 read]": "Reading tiles",
        "[4 observe]": "Combining tile observations",
        "[5 decode]": "Reconstructing hands",
        "[6 write]": "Building replay files",
    }
    for prefix, label in stages.items():
        if line.startswith(prefix):
            return label
    if line.startswith("table timing:"):
        return "Finding hands at the table"
    match = re.match(r"(read hand|hand)\s+(\d+):", line)
    if match:
        return (
            f"{('Reading tiles' if match[1] == 'read hand' else 'Reconstructing')} "
            f"· hand {int(match[2]) + 1}"
        )
    return None


class ChildError(Exception):
    """A job's child command failed; `message` is what the studio shows."""

    def __init__(
        self, message: str, result: object = None, *, unexpected: bool = False
    ) -> None:
        """Keep the reported message, any partial result and whether it was a bug."""
        super().__init__(message)
        self.message = message
        self.result = result
        self.unexpected = unexpected


@dataclass
class Job:
    """One child-process job of the workspace; its log is for developers only."""

    kind: JobKind
    project: str
    stage: str
    hands: list[int] | None = None
    started: float = field(default_factory=time.time)
    running: bool = True
    finished: float | None = None
    error: str | None = None
    interrupted: bool = False
    log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES))

    def status(self) -> dict:
        """Describe progress and the user-facing outcome, without the log."""
        return {
            "kind": self.kind,
            "project": self.project,
            "stage": self.stage,
            "hands": self.hands,
            "running": self.running,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
            "log_lines": len(self.log),
        }

    def record(self) -> dict:
        """Persist a preparation or analysis in its project manifest."""
        return {
            "action": self.kind,
            "running": self.running,
            "started": self.started,
            "finished": self.finished,
            "stage": self.stage,
            "error": self.error,
            "log": list(self.log),
        }


class SourceDigests:
    """Recording content digests computed in background threads.

    A digest is keyed by the file's identity and timestamps, so a replaced
    recording starts a new computation. Requests can wait for it or report that
    the recording is still being checked; nothing waits while holding a lock.
    """

    def __init__(self) -> None:
        """Start without remembered computations."""
        self._lock = threading.Lock()
        self._futures: dict[str, tuple[tuple[int, ...], Future[str | None]]] = {}

    def digest(self, path: str) -> Future[str | None]:
        """Return the digest computation for the file as it is now (None: missing)."""
        try:
            st = Path(path).stat()
        except FileNotFoundError:
            future: Future[str | None] = Future()
            future.set_result(None)
            return future
        key = (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)
        with self._lock:
            known = self._futures.get(path)
            if known and known[0] == key and not _failed(known[1]):
                return known[1]
            future = Future()
            self._futures[path] = (key, future)
        threading.Thread(
            target=_identify, args=(path, future), daemon=True, name="source-digest"
        ).start()
        return future


def _failed(future: Future) -> bool:
    return future.done() and future.exception() is not None


def _identify(path: str, future: Future[str | None]) -> None:
    try:
        digest = cache.source_identity(path)
    except FileNotFoundError:
        future.set_result(None)
    except Exception as error:
        LOGGER.exception("Could not identify the recording %s", path)
        future.set_exception(error)
    else:
        future.set_result(digest)


class Workspace:
    """Own persisted projects beneath a user-selected local data directory.

    `runner(args, job)` runs one child command and returns its reported result,
    raising ChildError; tests inject a replacement. Production uses a separate
    Python process per job so model memory is released afterwards.
    """

    def __init__(
        self, root: Path, runner: Callable[[list[str], Job], object] | None = None
    ) -> None:
        """Load projects under an isolated root and configure command ownership."""
        self.root = root.resolve()
        self.projects_dir = self.root / "work" / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runner = runner or self.run_child
        self.processes = ProcessOwner()
        self.digests = SourceDigests()
        self.job: Job | None = None
        self.threads: list[threading.Thread] = []
        self.projects: dict[str, dict] = {}
        self.states: dict[str, ReviewState] = {}
        self._open_items: dict[Path, tuple[tuple[int, int], int]] = {}
        for path in self.projects_dir.glob("*.json"):
            project = json.loads(path.read_text(encoding="utf-8"))
            if project["job"].get("running"):
                project["job"].update(running=False, error=INTERRUPTED)
                project["status"] = "interrupted"
            self.projects[project["id"]] = project

    # -- persistence -----------------------------------------------------------

    def _save(self, project: dict) -> None:
        """Publish one complete manifest, retaining the previous file on failure.

        Windows readers that omit delete sharing can briefly deny replacement
        (including WinError 5). Retry only those replacement errors for at most
        750 ms; a persistent denial is an actionable failure, not a running job.
        """
        path = self.projects_dir / f"{project['id']}.json"
        atomic_write_json(path, project, indent=2, retry_windows=True)

    @staticmethod
    def _save_failed(project: dict, exc: OSError, *, completed: bool = False) -> None:
        """Expose a retryable failure when storage cannot persist a job transition."""
        job = project["job"]
        job.update(
            running=False,
            finished=time.time(),
            stage="Project could not be saved",
            error=SAVE_FAILED,
        )
        job["log"] = [*job.get("log", []), f"Project save failed: {exc}"][-LOG_LINES:]
        project["status"] = "failed"
        if completed:
            project["export_signature"] = None

    def setup(self) -> dict:
        """Report actionable prerequisites without loading large ML models."""
        missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
        weights = (
            "models/detector/weights.pt",
            "models/classifier/weights.pt",
            "models/classifier/meta.json",
        )
        missing += [name for name in weights if not (self.root / name).is_file()]
        return {"ready": not missing, "missing": missing}

    # -- project library -------------------------------------------------------

    def _recording(
        self, body: dict, start: float, end: float | None
    ) -> tuple[str, str, Path, str]:
        """Validate a source and assign an identity distinct for each selected clip."""
        source = str(body.get("source", "")).strip()
        kind = body.get("kind", "local")
        if kind not in ("local", "url"):
            raise ValueError("Choose a local recording or a video URL.")
        if kind == "url":
            if not source:
                raise ValueError("Enter a video URL.")
            # Different sections must never share timestamped labels or cached frames.
            identity = json.dumps([source, start, end])
            stem = "video_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            video = self.root / "samples" / f"{stem}.mp4"
        else:
            if source.startswith(("\\\\", "//")) or "://" in source:
                raise ValueError("Choose a local video file, not a network address.")
            video = Path(source).expanduser()
            if not video.is_absolute():
                video = self.root / video
            video = video.resolve()
            if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError(
                    "The local recording does not exist or is not a supported video"
                    " file."
                )
            source = str(video)
            stem = video.stem
            if start or end is not None:
                identity = json.dumps([source, start, end])
                stem = (
                    stem[:80]
                    + "_clip_"
                    + hashlib.sha256(identity.encode()).hexdigest()[:16]
                )
                video = self.root / "samples" / f"{stem}.mp4"
        if not re.fullmatch(r"[\w .-]{1,120}", stem) or stem in (".", ".."):
            raise ValueError(
                "Rename the recording using letters, numbers, spaces, dots, "
                "underscores or hyphens."
            )
        return kind, source, video, stem

    def create(self, body: dict) -> dict:
        """Validate source/game order and persist a project without starting work.

        Local sources must exist and be video files. URL support is decided
        by yt-dlp when downloading. A basename is a pipeline identity, so two
        recordings cannot accidentally share labels or cached observations.
        """
        games, layout = self._settings(body)
        start, end = time_range(body.get("start"), body.get("end"))
        kind, source, video, stem = self._recording(body, start, end)
        display_name = self._display_name(body.get("display_name", stem))
        digest = self.digests.digest(str(video)).result()
        with self.lock:
            for existing in self.projects.values():
                if Path(existing["video"]).stem.casefold() == stem.casefold():
                    raise ValueError(
                        "This recording already has a project. Open it from Projects."
                    )
            project = {
                "id": uuid.uuid4().hex,
                "name": stem,
                "display_name": display_name,
                "kind": kind,
                "source": source,
                "video": str(video),
                "games": games,
                "layout": layout,
                "start": start,
                "end": end,
                "created": time.time(),
                "status": "new",
                "job": {"running": False},
                "source_sha256": digest,
                "export_signature": None,
            }
            if (self.root / "work" / stem / "hands.json").exists():
                self._invalidate_inputs(project, digest)
            self._save(project)
            self.projects[project["id"]] = project
        return self.snapshot(project["id"])

    @staticmethod
    def _display_name(value: object) -> str:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value.strip()) > MAX_PROJECT_NAME_LENGTH
        ):
            raise ValueError("Enter a project name between 1 and 120 characters.")
        return value.strip()

    def rename(self, key: str, body: dict) -> dict:
        """Change the display name without moving source, labels, caches or exports."""
        name = self._display_name(body.get("display_name"))
        with self.lock:
            stored = self.project(key)
            project = deepcopy(stored)
            project["display_name"] = name
            self._save(project)
            stored["display_name"] = name
        return self.snapshot(key)

    def delete(self, key: str) -> dict:
        """Remove an idle project manifest, retaining videos and human evidence."""
        with self.lock:
            self.project(key)
            running = self.busy()
            if running is not None and running.project == key:
                raise ValueError(
                    "Wait for this project's running job to finish before deleting it."
                )
            (self.projects_dir / f"{key}.json").unlink()
            del self.projects[key]
            state = self.states.pop(key, None)
            if state is not None:
                state.close()
            return {"deleted": key}

    def _settings(self, body: dict) -> tuple[list[int], str]:
        games = body.get("games")
        if (
            not isinstance(games, list)
            or not games
            or any(type(g) is not int or g <= 0 for g in games)
        ):
            raise ValueError(
                "Enter at least one positive scoremj game ID in recording order."
            )
        if len(games) != len(set(games)):
            raise ValueError("Each scoremj game ID must appear only once.")
        layout = str(body.get("layout") or "pml")
        if layout != "pml":
            path = Path(layout).expanduser()
            if not path.is_absolute():
                path = self.root / path
            if not path.is_file() or path.suffix != ".json":
                raise ValueError(
                    "The custom layout must be an existing local JSON file."
                )
            layout = str(path.resolve())
        return games, layout

    def update(self, key: str, body: dict) -> dict:
        """Correct idle project inputs while retaining source video and human labels.

        Game changes require analysis; layout changes require preparation. Existing
        exports become unavailable until successful analysis establishes provenance
        for the new settings.
        """
        games, layout = self._settings(body)
        with self.lock:
            running = self.busy()
            if running is not None:
                raise ValueError(
                    f"{BUSY[running.kind]}. Wait for it to finish before changing "
                    "settings."
                )
            stored = self.project(key)
            if stored["games"] != games or stored["layout"] != layout:
                needs_prepare = bool(
                    stored.get("needs_prepare") or stored["layout"] != layout
                )
                project = deepcopy(stored)
                project.update(
                    games=games,
                    layout=layout,
                    export_signature=None,
                    status="new"
                    if needs_prepare or stored["status"] == "new"
                    else "ready",
                    needs_prepare=needs_prepare,
                )
                self._mark_inputs_changed(
                    project,
                    "Analyze the recording after changing its game IDs or layout.",
                )
                self._save(project)
                stored.update(project)
                state = self.states.pop(key, None)
                if state is not None:
                    state.close()
        return self.snapshot(key)

    def _mark_inputs_changed(self, project: dict, reason: str) -> None:
        """Retire review rebuilds until analysis refreshes the evidence."""
        work = self.root / "work" / project["name"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "inputs.changed").write_text(reason + "\n", encoding="utf-8")

    def _signature(self, project: dict, source: str | None) -> str | None:
        """Bind exports to the recording content, games, layout and geometry."""
        if source is None:
            return None
        cal = Calibration.load(project["layout"], project["video"])
        payload = {
            "games": project["games"],
            "layout": project["layout"],
            "geometry": cal.data,
            "source_sha256": source,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _apply_source(self, project: dict, digest: str | None) -> None:
        """Adopt a first download, or retire evidence when the recording changed.

        The caller holds the lock and computed `digest` without it.
        """
        if project["source_sha256"] == digest:
            return
        work = self.root / "work" / project["name"]
        if (
            project["source_sha256"] is None
            and project["status"] in ("new", "preparing")
            and not (work / "inputs.changed").exists()
        ):
            project["source_sha256"] = digest
        else:
            self._invalidate_inputs(project, digest)
        self._save(project)

    def _invalidate_inputs(self, project: dict, source: str | None) -> None:
        """Retire unverified outputs without changing the saved human answers."""
        project.update(source_sha256=source, export_signature=None, needs_prepare=True)
        running = self.busy()
        if running is None or running.project != project["id"]:
            project["status"] = "new"
        self._mark_inputs_changed(
            project,
            "Prepare and analyze again: recording contents changed or input "
            "provenance is missing.",
        )

    def _record_matches(self, project: dict) -> bool:
        try:
            record = json.loads(
                read_published_text(
                    self.root / "work" / project["name"] / "record.json"
                )
            )
            return [game["id"] for game in record] == project["games"]
        except FileNotFoundError:
            return False

    def project(self, key: str) -> dict:
        """Resolve an opaque project ID; request paths never become disk paths."""
        if key not in self.projects:
            raise KeyError("Project not found.")
        return self.projects[key]

    def _current(self, key: str, *, wait: bool) -> tuple[dict, str | None, bool]:
        """Return a manifest copy synchronized with its recording, and its digest.

        Without `wait`, a digest still being computed is reported as checking
        (third value) and the manifest is returned unsynchronized.
        """
        with self.lock:
            video = self.project(key)["video"]
        future = self.digests.digest(video)
        try:
            digest = future.result(timeout=None if wait else DIGEST_PEEK_SECONDS)
        except FutureTimeoutError:
            with self.lock:
                return deepcopy(self.project(key)), None, True
        with self.lock:
            stored = self.project(key)
            self._apply_source(stored, digest)
            return deepcopy(stored), digest, False

    def _exports_valid(self, project: dict, digest: str | None) -> bool:
        """Require exports built from this recording, these settings and geometry."""
        work = self.root / "work" / project["name"]
        return (
            bool(project["export_signature"])
            and project["export_signature"] == self._signature(project, digest)
            and self._record_matches(project)
            and not any((work / marker).exists() for marker in MARKERS)
        )

    def _artifacts(self, project: dict) -> list[str]:
        out = self.root / "out" / project["name"]
        games = len(project["games"])
        return [
            p.name
            for p in sorted(out.glob("*"))
            if p.is_file()
            and ARTIFACT.fullmatch(p.name)
            and (not p.name.startswith("g") or int(p.name[1:].split(".")[0]) < games)
        ]

    def _has_fit(self, project: dict) -> bool:
        """Whether a prepared, current calibration allows analysis."""
        fit_path = self.root / "labels" / project["name"] / "calib.json"
        fit = json.loads(read_published_text(fit_path)) if fit_path.exists() else {}
        job = project["job"]
        return (
            bool(project["source_sha256"])
            and "overhead" in fit
            and not project.get("needs_prepare")
            and not (job.get("action") == "prepare" and job.get("error"))
        )

    def _count_open_items(self, path: Path) -> int:
        """Count questions in a published review queue, cached by file version."""
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._open_items.get(path)
        if cached is None or cached[0] != key:
            items = json.loads(read_published_text(path))
            self._open_items[path] = cached = (key, len(items))
        return cached[1]

    def snapshot(self, key: str) -> dict:
        """Describe one project for the browser without waiting for a digest."""
        project, digest, checking = self._current(key, wait=False)
        with self.lock:
            running = self.busy()
            job = (
                running.record()
                if running is not None
                and running.project == key
                and running.kind in ("prepare", "analyze")
                else dict(project["job"])
            )
        job.pop("log", None)
        valid = not checking and self._exports_valid(project, digest)
        out = self.root / "out" / project["name"]
        artifacts = self._artifacts(project) if valid else []
        return {
            "id": key,
            "name": project["name"],
            "display_name": project.get("display_name", project["name"]),
            "kind": project["kind"],
            "source": project["source"],
            "games": project["games"],
            "layout": project["layout"],
            "start": project.get("start", 0.0),
            "end": project.get("end"),
            "created": project.get("created"),
            "status": project["status"],
            "job": job,
            "checking": checking,
            "artifacts": artifacts,
            "stale_exports": not checking and not valid and any(out.glob("g*.json")),
            "results_revision": [
                [name, (out / name).stat().st_mtime_ns, (out / name).stat().st_size]
                for name in artifacts
                if GAME_LOG.fullmatch(name)
            ],
            "can_calibrate": bool(project["source_sha256"]),
            "has_fit": not checking and self._has_fit(project),
            "open_items": (
                self._count_open_items(out / "review.json")
                if "review.json" in artifacts
                else None
            ),
        }

    def review_state(self, key: str) -> ReviewState:
        """Lazily bind the review service to exactly one project."""
        with self.lock:
            project = self.project(key)
            if key not in self.states:
                self.states[key] = ReviewState(
                    Path(project["video"]),
                    self.root / "work",
                    project["layout"],
                    self.root / "out",
                    self.processes,
                )
            return self.states[key]

    def artifact(self, key: str, name: str) -> Path:
        """Return only a known generated export, excluding arbitrary workspace files."""
        if not ARTIFACT.fullmatch(name):
            raise ValueError("Unknown export filename.")
        project, digest, _ = self._current(key, wait=True)
        if not self._exports_valid(project, digest) or name not in self._artifacts(
            project
        ):
            raise FileNotFoundError(
                "These exports are out of date. Analyze the recording with its "
                "current settings first."
            )
        return self.root / "out" / project["name"] / name

    def results(self, key: str) -> dict:
        """Read result rows and replay links from provenance-checked exports.

        The browser and CLI share Tenhou URL serialization. HTML export files are never
        parsed or embedded, and stale exports are omitted.
        """
        project, digest, _ = self._current(key, wait=True)
        valid = self._exports_valid(project, digest)
        out = self.root / "out" / project["name"]
        logs = sorted(
            (int(m[1]), m[0])
            for name in (self._artifacts(project) if valid else [])
            if (m := GAME_LOG.fullmatch(name))
        )
        games = []
        for index, filename in logs:
            data = json.loads(read_published_text(out / filename))
            hands = []
            for hi, hand in enumerate(data["log"]):
                kyoku, honba = hand[0][:2]
                wind = ("East", "South", "West", "North")[kyoku // 4 % 4]
                hands.append(
                    {
                        "index": hi,
                        "round": f"{wind} {kyoku % 4 + 1}",
                        "honba": honba,
                        "editor_url": editor_url(data, hi),
                    }
                )
            games.append(
                {
                    "index": index,
                    "record_id": project["games"][index],
                    "names": data.get("name", []),
                    "hands": hands,
                    "viewer_url": viewer_url(data),
                    "download": f"/exports/{key}/{filename}",
                }
            )
        pending: set[int] = set()
        state = self.review_state(key)
        if valid:
            pending = set(state.pending())
        return {
            "games": games,
            "pending_games": sorted(
                {h["game"] for h in state.hands if h["hand"] in pending}
            ),
        }

    # -- jobs ------------------------------------------------------------------

    def busy(self) -> Job | None:
        """Return the running job, if any."""
        job = self.job
        return job if job is not None and job.running else None

    def job_status(self, key: str | None = None) -> dict:
        """Report the workspace job and, for a project, its review revision."""
        job = self.job
        revision = self.review_state(key).revision() if key is not None else None
        return {"job": job.status() if job else None, "revision": revision}

    def log(self, key: str) -> list[str]:
        """Return the developer log of a project's most recent job."""
        with self.lock:
            project = self.project(key)
            if self.job is not None and self.job.project == key:
                return list(self.job.log)
            return list(project["job"].get("log", []))

    def _claim(self, job: Job) -> None:
        """Take the single job slot; the caller holds the lock."""
        if self.processes.closing:
            raise ValueError("The app is closing. Restart it before starting a job.")
        running = self.busy()
        if running is not None:
            raise ValueError(f"{BUSY[running.kind]}. Wait for it to finish.")
        self.job = job

    def _launch(
        self,
        job: Job,
        work: Callable[[], object],
        finish: Callable[[], None] | None = None,
    ) -> None:
        """Run a claimed job in a thread; `finish` runs under the lock afterwards."""

        def run() -> None:
            try:
                work()
            except Exception as error:
                LOGGER.exception("%s for project %s failed", job.kind, job.project)
                job.interrupted = self.processes.closing or isinstance(
                    error, ProcessCancelledError
                )
                job.error = self._failure_message(job, error)
            finally:
                with self.lock:
                    job.running = False
                    job.finished = time.time()
                    if finish is not None:
                        finish()

        thread = threading.Thread(
            target=run, daemon=True, name=f"{job.kind}-{job.project}"
        )
        self.threads.append(thread)
        thread.start()

    @staticmethod
    def _failure_message(job: Job, error: Exception) -> str:
        if job.interrupted:
            return INTERRUPTED
        if isinstance(error, ChildError) and not error.unexpected:
            return error.message
        if isinstance(error, ValueError):
            return str(error)
        detail = error.message if isinstance(error, ChildError) else str(error)
        return (
            f"{LABELS[job.kind]} failed unexpectedly: {detail}. The processing "
            "log has details."
        )

    def start(self, key: str, action: str) -> dict:
        """Start preparation or analysis once, preserving logs and failures for retry.

        Preparation downloads if needed, then measures geometry. Analysis is
        a separate explicit user action after visual calibration review and
        always retains the CLI's geometry gate. Review facts invalidate
        decode caches through the existing pipeline.
        """
        if action not in ("prepare", "analyze"):
            raise ValueError("Unknown project action.")
        project, digest, _ = self._current(key, wait=True)
        self._check_start(project, action, digest)
        with self.lock:
            project = self.project(key)
            job = Job(
                kind="prepare" if action == "prepare" else "analyze",
                project=key,
                stage="Preparing recording"
                if action == "prepare"
                else "Checking recording",
            )
            self._claim(job)
            project.update(
                status="preparing" if action == "prepare" else "analyzing",
                job=job.record(),
            )
            if action == "prepare":
                project["needs_prepare"] = True
            try:
                self._save(project)
            except OSError as exc:
                job.running, job.error = False, SAVE_FAILED
                self._save_failed(project, exc)
                raise ValueError(SAVE_FAILED) from exc
            self._release_review_states()
        updates: dict = {}

        def work() -> None:
            if action == "prepare":
                updates.update(self._prepare(project, job))
            else:
                updates.update(self._analyze(project, job, digest))

        def finish() -> None:
            project.update(updates)
            if job.error:
                project["status"] = "interrupted" if job.interrupted else "failed"
            project["job"] = job.record()
            try:
                self._save(project)
            except OSError as exc:
                self._save_failed(project, exc, completed=True)
                job.error = SAVE_FAILED
                job.log.append(f"Project save failed: {exc}")

        self._launch(job, work, finish)
        return self.snapshot(key)

    def _check_start(self, project: dict, action: str, digest: str | None) -> None:
        """Check source availability and calibration before launch."""
        # Projects created before clip ranges existed have no bounds.
        bounded = bool(project.get("start")) or project.get("end") is not None
        can_trim = (
            project["kind"] == "local"
            and action == "prepare"
            and bounded
            and Path(project["source"]).is_file()
        )
        if digest is None and project["kind"] != "url" and not can_trim:
            raise ValueError(
                "Recording not found. Restore the local video before continuing."
            )
        if action == "analyze" and not self._has_fit(project):
            raise ValueError(
                "Prepare and check the table calibration before analyzing."
            )

    def _release_review_states(self) -> None:
        """Close review states and release their GPU allocations before analysis."""
        # Review prefill models otherwise retain VRAM while a fresh child
        # loads another copy for the full pipeline.
        had_models = any(state.models_loaded for state in self.states.values())
        for state in self.states.values():
            state.close()
        self.states.clear()
        if had_models:
            import torch  # noqa: PLC0415

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _cli(self, *args: str) -> list[str]:
        return [sys.executable, "-u", "-m", "video2tenhou.cli", *args]

    def _calibration_args(self, project: dict, kind: str) -> list[str]:
        return self._cli(
            "calib",
            kind,
            project["video"],
            "--calib",
            project["layout"],
            "--work",
            str(self.root / "work"),
            "--out",
            str(self.root / "work" / "calib"),
        )

    def _prepare(self, project: dict, job: Job) -> dict:
        """Acquire the recording and fit geometry against a stable source identity."""
        video = project["video"]
        if not Path(video).exists():
            operation = "download" if project["kind"] == "url" else "trim"
            job.stage = (
                "Downloading recording"
                if operation == "download"
                else "Selecting recording time range"
            )
            command = self._cli(operation)
            if project.get("start"):
                command += ["--start", format_time(project["start"])]
            if project.get("end") is not None:
                command += ["--end", format_time(project["end"])]
            self.runner([*command, "--", project["source"], video], job)
        fit_source = self.digests.digest(video).result()
        job.stage = "Measuring table layout"
        self._calibration_job(job, self._calibration_args(project, "fit"))
        if fit_source is None or self.digests.digest(video).result() != fit_source:
            raise ValueError(
                "The recording changed during preparation. Restore it, then retry."
            )
        return {"source_sha256": fit_source, "status": "ready", "needs_prepare": False}

    def _analyze(self, project: dict, job: Job, source_before: str | None) -> dict:
        """Convert a calibrated recording and require unchanged input for exports."""
        command = self._cli(
            "convert",
            project["video"],
            "--calib",
            project["layout"],
            "--work",
            str(self.root / "work"),
            "--out",
            str(self.root / "out"),
            "--redo",
            "decode",
        )
        for game in project["games"]:
            command += ["--game", str(game)]
        self.runner(command, job)
        source_after = self.digests.digest(project["video"]).result()
        if source_after != source_before:
            with self.lock:
                self._apply_source(project, source_after)
            raise ValueError(
                "The recording changed during analysis. Prepare it again, then retry."
            )
        return {
            "status": "complete",
            "export_signature": self._signature(project, source_before),
        }

    def _calibration_job(self, job: Job, args: list[str]) -> None:
        """Run a fit or border check and remember its per-region results."""
        state = self.review_state(job.project)
        before = state.cal.data
        result: object = None
        try:
            result = self.runner(args, job)
        except ChildError as failure:
            result = failure.result
            raise
        finally:
            state.geometry_saved(before)
            if isinstance(result, dict):
                state.save_checks(result)

    def calibrate(self, key: str, kind: str) -> dict:
        """Start measuring the table again ("fit") or checking its borders."""
        if kind not in ("fit", "check"):
            raise ValueError("Unknown calibration action.")
        with self.lock:
            project = deepcopy(self.project(key))
            job = Job(
                kind="fit" if kind == "fit" else "check",
                project=key,
                stage="Measuring table layout" if kind == "fit" else "Checking borders",
            )
            self._claim(job)
        args = self._calibration_args(project, kind)
        self._launch(job, lambda: self._calibration_job(job, args))
        return self.job_status()

    def rebuild(self, key: str, which: object) -> dict:
        """Start updating hands with the saved answers and rewriting the exports.

        "pending" updates only hands whose decode or export is not current, reusing
        decodes that already match; "all" or a list of hand indices decodes afresh.
        """
        state = self.review_state(key)
        known = [h["hand"] for h in state.hands]
        if which == "pending":
            hands, force = state.pending(), False
        elif which == "all":
            hands, force = known, True
        elif (
            isinstance(which, list)
            and which
            and all(type(h) is int and h in known for h in which)
        ):
            hands, force = sorted(set(which)), True
        else:
            raise ValueError(
                "Choose pending hands, all hands or known hand numbers to update."
            )
        if not hands:
            return self.job_status()
        with self.lock:
            project = deepcopy(self.project(key))
            job = Job(kind="rebuild", project=key, stage="Updating hands", hands=hands)
            self._claim(job)
        args = self._cli(
            "rebuild",
            project["video"],
            "--calib",
            project["layout"],
            "--work",
            str(self.root / "work"),
            "--out",
            str(self.root / "out"),
            "--hands",
            *map(str, hands),
            *(["--force"] if force else []),
        )
        self._launch(job, lambda: self.runner(args, job))
        return self.job_status()

    def close(self) -> None:
        """Stop owned child process trees and persist interrupted jobs."""
        self.processes.shutdown()
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)
        with self.lock:
            for state in self.states.values():
                state.close()

    def run_child(self, args: list[str], job: Job) -> object:
        """Run one CLI command, streaming its log, and return its reported result.

        Both output streams feed the job's developer log and stage; the final
        stdout JSON document is the outcome. Raises ChildError with the
        command's own message, or ProcessCancelledError when the app closes.
        """
        env = {**os.environ, "VIDEO2TENHOU_HOME": str(self.root)}
        outcomes: list[dict] = []
        with self.processes.spawn(
            args,
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            reader = threading.Thread(
                target=self._follow, args=(process.stdout, job, outcomes), daemon=True
            )
            reader.start()
            self._follow(process.stderr, job, None)
            reader.join()
            returncode = process.wait()
        outcome = outcomes[-1] if outcomes else {}
        if returncode or outcome.get("error"):
            raise ChildError(
                outcome.get("error")
                or (f"the command stopped with exit code {returncode}"),
                outcome.get("result"),
                unexpected=bool(outcome.get("unexpected")) or not outcome,
            )
        return outcome.get("result")

    @staticmethod
    def _follow(stream: Iterable[str] | None, job: Job, outcomes: list | None) -> None:
        """Append lines to the job log; collect stdout outcome documents."""
        for raw in stream or ():
            line = raw.strip()
            if not line:
                continue
            if outcomes is not None and line.startswith("{"):
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    document = None
                if isinstance(document, dict) and (
                    "result" in document or "error" in document
                ):
                    outcomes.append(document)
                    continue
            job.log.append(line)
            stage = _progress_stage(line)
            if stage:
                job.stage = stage
