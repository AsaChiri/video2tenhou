# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent local recording projects and subprocess-backed browser jobs.

Manifests contain only workflow metadata; the established work/<video>,
labels/<video> and out/<video> contracts remain the pipeline's source of
truth. Jobs are serialized across projects to bound GPU memory. A server
restart marks an unfinished job interrupted rather than silently claiming
success. HTTP handlers never execute shell command strings.
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
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from video2tenhou import cache
from video2tenhou.files import atomic_write_json
from video2tenhou.layout import Calibration
from video2tenhou.tenhou6 import editor_url, viewer_url
from video2tenhou.video import format_time, time_range

from .processes import ProcessOwner
from .review_state import ReviewState

if TYPE_CHECKING:
    from collections.abc import Callable


MAX_PROJECT_NAME_LENGTH = 120
LOGGER = logging.getLogger(__name__)


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
ARTIFACT = re.compile(
    r"(?:g\d+\.(?:json|html|confidence\.json)|review\.json|report\.md)\Z"
)


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


class Workspace:
    """Own persisted projects beneath a user-selected local data directory.

    `runner` is an injectable command runner for integration tests. Production
    uses a separate Python process per pipeline phase so model memory is
    released after each run and logs cannot capture unrelated server output.
    """

    def __init__(
        self, root: Path, runner: Callable[[list[str], dict], None] | None = None
    ) -> None:
        """Load projects under an isolated root and configure command ownership."""
        self.root = root.resolve()
        self.projects_dir = self.root / "work" / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runner = runner or self._run_command
        self.processes = ProcessOwner()
        self._threads: list[threading.Thread] = []
        self.projects: dict[str, dict] = {}
        self.states: dict[str, ReviewState] = {}
        for path in self.projects_dir.glob("*.json"):
            project = json.loads(path.read_text(encoding="utf-8"))
            if project["job"]["running"]:
                project["job"].update(
                    running=False,
                    error=(
                        "Interrupted when the app closed. Retry to resume from "
                        "cached work."
                    ),
                )
                project["status"] = "interrupted"
            self.projects[project["id"]] = project

    def _save(self, project: dict) -> None:
        """Publish one complete manifest, retaining the previous file on failure.

        Windows readers that omit delete sharing can briefly deny replacement
        (including WinError 5). Retry only those replacement errors for at most
        750 ms; a persistent denial is an actionable failure, not a running job.
        """
        path = self.projects_dir / f"{project['id']}.json"
        atomic_write_json(path, project, indent=2, retry_windows=True)

    def _save_failed(
        self, project: dict, exc: OSError, *, completed: bool = False
    ) -> None:
        """Expose a retryable failure when storage cannot persist a job transition."""
        job = project["job"]
        message = (
            "Could not save the project. Check that its data folder is writable and"
            " available, then retry."
        )
        job.update(
            running=False,
            finished=time.time(),
            stage="Project could not be saved",
            error=message,
        )
        job["log"] = ([*job.get("log", []), f"Project save failed: {exc}"])[-80:]
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
        return {
            "ready": not missing,
            "missing": missing,
            "data_directory": str(self.root),
        }

    def _recording(
        self, body: dict, start: float, end: float | None
    ) -> tuple[str, str, Path, str]:
        """Validate a source and assign an identity distinct for each selected clip."""
        source = str(body.get("source", "")).strip()
        kind = body.get("kind", "local")
        if kind not in ("local", "url"):
            msg = "Choose a local recording or a video URL."
            raise ValueError(msg)
        if kind == "url":
            if not source:
                msg = "Enter a video URL."
                raise ValueError(msg)
            # Different sections must never share timestamped labels or cached frames.
            identity = json.dumps([source, start, end])
            stem = "video_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            video = self.root / "samples" / f"{stem}.mp4"
        else:
            if source.startswith(("\\\\", "//")) or "://" in source:
                msg = "Choose a local video file, not a network address."
                raise ValueError(msg)
            video = Path(source).expanduser()
            if not video.is_absolute():
                video = self.root / video
            video = video.resolve()
            if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
                msg = (
                    "The local recording does not exist or is not a supported video"
                    " file."
                )
                raise ValueError(msg)
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
            msg = (
                "Rename the recording using letters, numbers, spaces, dots, "
                "underscores or hyphens."
            )
            raise ValueError(msg)
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
        with self.lock:
            for existing in self.projects.values():
                if Path(existing["video"]).stem.casefold() == stem.casefold():
                    msg = "This recording already has a project. Open it from Projects."
                    raise ValueError(msg)
            project = {
                "id": uuid.uuid4().hex,
                "name": stem,
                "display_name": self._display_name(body.get("display_name", stem)),
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
            }
            project["source_sha256"] = self._source(project)
            project["export_signature"] = None
            if (self.root / "work" / stem / "hands.json").exists():
                self._invalidate_inputs(project, project["source_sha256"])
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
            msg = "Enter a project name between 1 and 120 characters."
            raise ValueError(msg)
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
            project = self.project(key)
            state = self.states.get(key)
            if project["job"].get("running") or (
                state is not None
                and any(job.get("running") for job in state.jobs.values())
            ):
                msg = (
                    "Wait for this project's running job to finish before deleting it."
                )
                raise ValueError(msg)
            (self.projects_dir / f"{key}.json").unlink()
            del self.projects[key]
            if state is not None:
                state.close()
                del self.states[key]
            return {"deleted": key}

    def _settings(self, body: dict) -> tuple[list[int], str]:
        games = body.get("games")
        if (
            not isinstance(games, list)
            or not games
            or any(type(g) is not int or g <= 0 for g in games)
        ):
            msg = "Enter at least one positive scoremj game ID in recording order."
            raise ValueError(msg)
        if len(games) != len(set(games)):
            msg = "Each scoremj game ID must appear only once."
            raise ValueError(msg)
        layout = str(body.get("layout") or "pml")
        if layout != "pml":
            path = Path(layout).expanduser()
            if not path.is_absolute():
                path = self.root / path
            if not path.is_file() or path.suffix != ".json":
                msg = "The custom layout must be an existing local JSON file."
                raise ValueError(msg)
            layout = str(path.resolve())
        return games, layout

    def update(self, key: str, body: dict) -> dict:
        """Correct idle project inputs while retaining source video and human labels.

        Game changes rebuild alignment and decode; layout changes require preparation.
        Existing exports become unavailable until successful analysis establishes
        provenance for the new settings.
        """
        games, layout = self._settings(body)
        with self.lock:
            if any(p["job"].get("running") for p in self.projects.values()) or any(
                any(j.get("running") for j in state.jobs.values())
                for state in self.states.values()
            ):
                msg = (
                    "Wait for the running job to finish before changing project "
                    "settings."
                )
                raise ValueError(msg)
            project = self.project(key)
            if project["games"] == games and project["layout"] == layout:
                return self.snapshot(key)
            layout_changed = project["layout"] != layout
            project.update(
                games=games,
                layout=layout,
                export_signature=None,
                status="ready",
                job={"running": False},
                needs_prepare=layout_changed,
                inputs_changed=True,
            )
            work = self.root / "work" / project["name"]
            work.mkdir(parents=True, exist_ok=True)
            (work / "inputs.changed").write_text(
                "Analyze the recording after changing its game IDs or layout.\n",
                encoding="utf-8",
            )
            if layout_changed:
                project["status"] = "new"
            state = self.states.pop(key, None)
            if state is not None:
                state.close()
            self._save(project)
            return self.snapshot(key)

    def _signature(self, project: dict) -> str | None:

        source = self._source(project)
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

    def _source(self, project: dict) -> str | None:

        try:
            # The shared stat-keyed memo avoids rereading a broadcast on each poll.
            return cache.source_identity(project["video"])
        except FileNotFoundError:
            return None

    def _sync_source(self, project: dict) -> None:
        """Require regeneration when the recording contents change."""
        current = self._source(project)
        if project["source_sha256"] != current:
            # A first download has no earlier evidence to invalidate.
            if (
                project["source_sha256"] is None
                and project["status"] in ("new", "preparing")
                and not project.get("inputs_changed")
            ):
                project["source_sha256"] = current
                self._save(project)
            else:
                self._invalidate_inputs(project, current)

    def _invalidate_inputs(self, project: dict, source: str | None) -> None:
        """Retire unverified outputs without changing the saved human answers."""
        project.update(
            source_sha256=source,
            export_signature=None,
            inputs_changed=True,
            needs_prepare=True,
        )
        if not project["job"].get("running"):
            project["status"] = "new"
        work = self.root / "work" / project["name"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "inputs.changed").write_text(
            (
                "Prepare and analyze again: recording contents changed or input "
                "provenance is missing.\n"
            ),
            encoding="utf-8",
        )
        state = self.states.pop(project["id"], None)
        if state is not None:
            state.close()
        self._save(project)

    def _record_matches(self, project: dict) -> bool:
        try:
            record = json.loads(
                (self.root / "work" / project["name"] / "record.json").read_text(
                    encoding="utf-8"
                )
            )
            return [game["id"] for game in record] == project["games"]
        except FileNotFoundError:
            return False

    def project(self, key: str) -> dict:
        """Resolve an opaque project ID; request paths never become disk paths."""
        if key not in self.projects:
            msg = "Project not found."
            raise KeyError(msg)
        return self.projects[key]

    def snapshot(self, key: str) -> dict:
        """Return workflow state with available exports and review counts."""
        with self.lock:
            stored = self.project(key)
            self._sync_source(stored)
            project = deepcopy(stored)
            project["display_name"] = project.get("display_name", project["name"])
            state = self.states.get(key)
            project["review_running"] = bool(
                state and any(job.get("running") for job in state.jobs.values())
            )
        name = Path(project["video"]).stem
        out = self.root / "out" / name
        valid = (
            bool(project["export_signature"])
            and project["export_signature"] == self._signature(project)
            and self._record_matches(project)
        )
        valid = valid and not any(
            (self.root / "work" / name / marker).exists()
            for marker in ("calibration.changed", "inputs.changed")
        )
        project["artifacts"] = [
            p.name
            for p in sorted(out.glob("*"))
            if valid
            and p.is_file()
            and ARTIFACT.fullmatch(p.name)
            and (
                not p.name.startswith("g")
                or int(p.name[1:].split(".")[0]) < len(project["games"])
            )
        ]
        project["stale_exports"] = any(out.glob("g*.json")) and not valid
        project["results_revision"] = [
            [
                filename,
                (out / filename).stat().st_mtime_ns,
                (out / filename).stat().st_size,
            ]
            for filename in project["artifacts"]
            if re.fullmatch(r"g\d+\.json", filename)
        ]
        project["can_calibrate"] = bool(project["source_sha256"])
        fit_path = self.root / "labels" / name / "calib.json"
        fit = (
            json.loads(fit_path.read_text(encoding="utf-8"))
            if fit_path.exists()
            else {}
        )
        project["has_fit"] = (
            bool(project["source_sha256"])
            and "overhead" in fit
            and not project.get("needs_prepare")
            and not (
                project["job"].get("action") == "prepare"
                and project["job"].get("error")
            )
        )
        project["has_hands"] = (
            bool(project["source_sha256"])
            and (self.root / "work" / name / "hands.json").exists()
            and not project.get("inputs_changed")
        )
        if valid:
            items = json.loads((out / "review.json").read_text(encoding="utf-8"))
            project["open_items"] = sum(
                item.get("kind") != "solver_incomplete" for item in items
            )
            project["conflicts"] = sum(i.get("kind") == "conflict" for i in items)
        else:
            project["open_items"] = None
            project["conflicts"] = 0
        project["pending_rebuilds"] = (
            self.review_state(key).pending_rebuilds()
            if valid and project["has_hands"]
            else []
        )
        return project

    def review_state(self, key: str) -> ReviewState:
        """Lazily bind the existing review service to exactly one project."""
        with self.lock:
            project = self.project(key)
            self._sync_source(project)
            signature = self._signature(project)
            if (
                key in self.states
                and self.states[key].configuration_signature != signature
            ):
                self.states[key].reload_calib()
                self.states[key].configuration_signature = signature
            if key not in self.states:
                self.states[key] = ReviewState(
                    Path(project["video"]),
                    self.root / "work",
                    project["layout"],
                    self.root / "out",
                )
                self.states[key].configuration_signature = signature
                if project.get("inputs_changed"):
                    self.states[key].hands = []
            return self.states[key]

    def artifact(self, key: str, name: str) -> Path:
        """Return only a known generated export, excluding arbitrary workspace files."""
        if not ARTIFACT.fullmatch(name):
            msg = "Unknown export filename."
            raise ValueError(msg)
        project = self.project(key)
        if name not in self.snapshot(key)["artifacts"]:
            msg = (
                "These exports are out of date. Analyze the recording with its "
                "current settings first."
            )
            raise FileNotFoundError(msg)
        path = self.root / "out" / Path(project["video"]).stem / name
        if not path.is_file():
            msg = "This export is not available yet."
            raise FileNotFoundError(msg)
        return path

    def results(self, key: str) -> dict:
        """Read result rows and replay links from provenance-checked exports.

        The browser and CLI share Tenhou URL serialization. HTML export files are never
        parsed or embedded, and stale exports are omitted.
        """
        project = self.snapshot(key)
        games = []
        files = sorted(
            (f for f in project["artifacts"] if re.fullmatch(r"g\d+\.json", f)),
            key=lambda f: int(f[1:-5]),
        )
        for filename in files:
            index = int(filename[1:-5])
            data = json.loads(self.artifact(key, filename).read_text(encoding="utf-8"))
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
        pending = set(project["pending_rebuilds"])
        pending_games = (
            sorted(
                {
                    h["game"]
                    for h in self.review_state(key).hands
                    if h["hand"] in pending
                }
            )
            if pending
            else []
        )
        return {
            "games": games,
            "revision": project["results_revision"],
            "pending_rebuilds": sorted(pending),
            "pending_games": pending_games,
        }

    def start(self, key: str, action: str) -> dict:
        """Start preparation or analysis once, preserving logs and failures for retry.

        Preparation downloads if needed, then measures geometry. Analysis is
        a separate explicit user action after visual calibration review and
        always retains the CLI's geometry gate. Review facts invalidate
        decode caches through the existing pipeline.
        """
        if action not in ("prepare", "analyze"):
            msg = "Unknown project action."
            raise ValueError(msg)
        with self.lock:
            if self.processes.closing:
                msg = "The app is closing. Restart it before starting a job."
                raise ValueError(msg)
            project = self.project(key)
            self._sync_source(project)
            self._check_start(key, action, project)
            self._release_review_states()
            job = {
                "running": True,
                "action": action,
                "started": time.time(),
                "stage": "Preparing recording"
                if action == "prepare"
                else "Checking recording",
                "log": [],
                "error": None,
            }
            project.update(
                status="preparing" if action == "prepare" else "analyzing", job=job
            )
            if action == "prepare":
                project["needs_prepare"] = True
            try:
                self._save(project)
            except OSError as exc:
                self._save_failed(project, exc)
                raise ValueError(job["error"]) from exc

        def run() -> None:
            try:
                source_before = self._source(project)
                args = [sys.executable, "-u", "-m", "video2tenhou.cli"]
                if action == "prepare":
                    self._prepare(project, args)
                else:
                    self._analyze(project, args, source_before)
                job["stage"] = (
                    "Ready to check the table"
                    if action == "prepare"
                    else "Analysis complete"
                )
            except Exception as exc:
                LOGGER.exception("Project %s %s failed", key, action)
                project["status"] = (
                    "interrupted" if self.processes.closing else "failed"
                )
                job["error"] = (
                    "Interrupted when the app closed. Retry to resume from cached work."
                    if self.processes.closing
                    else str(exc)
                )
            finally:
                with self.lock:
                    job.update(running=False, finished=time.time())
                    state = self.states.pop(
                        key, None
                    )  # Reload new hands and geometry on the next request.
                    if state is not None:
                        state.close()
                    try:
                        self._save(project)
                    except OSError as exc:
                        self._save_failed(project, exc, completed=True)

        thread = threading.Thread(target=run, daemon=True, name=f"project-{key}")
        self._threads.append(thread)
        thread.start()
        return self.snapshot(key)

    def _check_start(self, key: str, action: str, project: dict) -> None:
        """Check source availability, exclusive jobs and calibration before launch."""
        bounded = bool(project["start"]) or project["end"] is not None
        can_trim = (
            project["kind"] == "local"
            and action == "prepare"
            and bounded
            and Path(project["source"]).is_file()
        )
        if self._source(project) is None and project["kind"] != "url" and not can_trim:
            msg = "Recording not found. Restore the local video before continuing."
            raise ValueError(msg)
        if any(p.get("job", {}).get("running") for p in self.projects.values()):
            msg = (
                "Another job is running. Wait for it to finish before starting "
                "this recording."
            )
            raise ValueError(msg)
        if any(
            any(j.get("running") for j in s.jobs.values()) for s in self.states.values()
        ):
            msg = "A review job is running. Wait for it to finish first."
            raise ValueError(msg)
        if action == "analyze" and not self.snapshot(key)["has_fit"]:
            msg = "Prepare and check the table calibration before analyzing."
            raise ValueError(msg)

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

    def _prepare(self, project: dict, args: list[str]) -> None:
        """Acquire the recording and fit geometry against a stable source identity."""
        job = project["job"]
        if not Path(project["video"]).exists():
            operation = "download" if project["kind"] == "url" else "trim"
            job["stage"] = (
                "Downloading recording"
                if operation == "download"
                else "Selecting recording time range"
            )
            command = [*args, operation]
            if project["start"]:
                command += ["--start", format_time(project["start"])]
            if project["end"] is not None:
                command += ["--end", format_time(project["end"])]
            self.runner(
                [*command, "--", project["source"], project["video"]],
                project,
            )
        fit_source = self._source(project)
        job["stage"] = "Measuring table layout"
        self.runner(
            [
                *args,
                "calib",
                "fit",
                project["video"],
                "--calib",
                project["layout"],
                "--work",
                str(self.root / "work"),
                "--out",
                str(self.root / "work" / "calib"),
            ],
            project,
        )
        if fit_source is None or self._source(project) != fit_source:
            msg = "The recording changed during preparation. Restore it, then retry."
            raise ValueError(msg)
        project["source_sha256"] = fit_source
        project["status"] = "ready"
        project["needs_prepare"] = False

    def _analyze(
        self, project: dict, args: list[str], source_before: str | None
    ) -> None:
        """Convert a calibrated recording and require unchanged input for exports."""
        command = [
            *args,
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
        ]
        for game in project["games"]:
            command += ["--game", str(game)]
        self.runner(command, project)
        if self._source(project) != source_before:
            with self.lock:
                self._sync_source(project)
            msg = "The recording changed during analysis. Prepare it again, then retry."
            raise ValueError(msg)
        project["status"] = "complete"
        project["export_signature"] = self._signature(project)
        project["inputs_changed"] = False
        for marker in ("inputs.changed", "calibration.changed"):
            (self.root / "work" / project["name"] / marker).unlink(missing_ok=True)

    def close(self) -> None:
        """Stop owned pipeline/review process trees and persist interrupted jobs."""
        self.processes.shutdown()
        for state in list(self.states.values()):
            state.close()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)

    def _run_command(self, args: list[str], project: dict) -> None:
        env = {**os.environ, "PYTHONUTF8": "1", "VIDEO2TENHOU_HOME": str(self.root)}
        with self.processes.spawn(
            args,
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            if process.stdout is None:
                msg = "Pipeline process did not provide its requested output pipe"
                raise RuntimeError(msg)
            for output_line in process.stdout:
                line = output_line.strip()
                if not line:
                    continue
                with self.lock:
                    job = project["job"]
                    job["log"] = (job["log"] + [line])[-80:]
                    stage = _progress_stage(line)
                    if stage:
                        job["stage"] = stage
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(
                    returncode, args, output="\n".join(project["job"]["log"])
                )
