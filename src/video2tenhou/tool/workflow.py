"""Persistent local recording projects and subprocess-backed browser jobs.

Manifests contain only workflow metadata; the established work/<video>,
labels/<video> and out/<video> contracts remain the pipeline's source of
truth. Jobs are serialized across projects to bound GPU memory. A server
restart marks an unfinished job interrupted rather than silently claiming
success. HTTP handlers never execute shell command strings.
"""
from __future__ import annotations

from copy import deepcopy
import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .processes import ProcessOwner
from ..files import atomic_write_json

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
ARTIFACT = re.compile(r"(?:g\d+\.(?:json|html|confidence\.json)|review\.json|report\.md)\Z")


def _progress_stage(line: str) -> str | None:
    """Translate pipeline progress into stable user actions; keep raw detail in logs."""
    stages = {"[0 fit]": "Checking table borders", "[1 header]": "Matching hands to the score record",
              "[2 calm]": "Finding clear video frames", "[3 read]": "Reading tiles",
              "[4 observe]": "Combining tile observations", "[5 decode]": "Reconstructing hands",
              "[6 write]": "Building replay files"}
    for prefix, label in stages.items():
        if line.startswith(prefix):
            return label
    if line.startswith("overlay:"):
        return "Finding hands in the recording"
    match = re.match(r"(read hand|hand)\s+(\d+):", line)
    if match:
        return f"{'Reading tiles' if match[1] == 'read hand' else 'Reconstructing'} · hand {int(match[2]) + 1}"
    return None


def _failure_message(lines: list[str]) -> str:
    """Summarize corrective action while retaining subprocess diagnostics in logs."""
    if any(re.match(r"\s*FAIL\s+(overhead|(?:pond|hand|meld):)", line) for line in lines):
        return "Calibration needs adjustment. Adjust the highlighted table regions, then check again."
    if any("video and site record disagree" in line for line in lines):
        return "The recording does not match the score records. Check the game IDs and their order in Settings, then retry."
    detail = "\n".join(lines).lower()
    if "out of memory" in detail and "cuda" in detail:
        return "The GPU ran out of memory. Close other GPU applications and retry, or relaunch with VIDEO2TENHOU_DEVICE=cpu (slower)."
    if any(message in detail for message in ("no kernel image is available", "not compiled with cuda",
                                             "cuda driver version is insufficient", "invalid device function",
                                             "runtime check failed")):
        return "The GPU runtime could not run on this computer. Update uv and the NVIDIA driver, then relaunch Start.cmd or start.sh to repair setup, or use VIDEO2TENHOU_DEVICE=cpu (slower). See docs/QUICKSTART.md."
    if ("filenotfounderror" in detail and any(name in detail for name in ("weights.pt", "meta.json"))
            or "metadata does not match checkpoint" in detail):
        return "The trained models are missing or do not match. Extract the complete model bundle from the same release into the active data directory, then retry. See docs/QUICKSTART.md."
    return "Processing stopped. Open the processing log for details, correct the issue, then retry."


class Workspace:
    """Own persisted projects beneath a user-selected local data directory.

    `runner` is an injectable command runner for integration tests. Production
    uses a separate Python process per pipeline phase so model memory is
    released after each run and logs cannot capture unrelated server output.
    """

    def __init__(self, root: Path, runner=None):
        self.root = root.resolve()
        self.projects_dir = self.root / "work" / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runner = runner or self._run_command
        self.processes = ProcessOwner()
        self._threads: list[threading.Thread] = []
        self.projects: dict[str, dict] = {}
        self.states: dict[str, object] = {}
        for path in self.projects_dir.glob("*.json"):
            try:
                project = json.loads(path.read_text(encoding="utf-8"))
                if project.get("job", {}).get("running"):
                    project["job"].update(running=False, error="Interrupted when the app closed. Retry to resume from cached work.")
                    project["status"] = "interrupted"
                self.projects[project["id"]] = project
            except (OSError, ValueError, KeyError):
                continue  # A broken manifest must not prevent other projects opening.

    def _save(self, project: dict) -> None:
        """Publish one complete manifest, retaining the previous file on failure.

        Windows readers that omit delete sharing can briefly deny replacement
        (including WinError 5). Retry only those replacement errors for at most
        750 ms; a persistent denial is an actionable failure, not a running job.
        """
        path = self.projects_dir / f"{project['id']}.json"
        atomic_write_json(path, project, indent=2, retry_windows=True)

    def _save_failed(self, project: dict, exc: OSError, *, completed: bool = False) -> None:
        """Expose a retryable failure when storage cannot persist a job transition."""
        job = project["job"]
        message = "Could not save the project. Check that its data folder is writable and available, then retry."
        job.update(running=False, finished=time.time(), stage="Project could not be saved", error=message)
        job["log"] = (job.get("log", []) + [f"Project save failed: {exc}"])[-80:]
        project["status"] = "failed"
        if completed:
            project["export_signature"] = None

    def setup(self) -> dict:
        """Report actionable prerequisites without loading large ML models."""
        missing = [name for name in ("ffmpeg", "ffprobe", "tesseract") if not shutil.which(name)]
        weights = ("models/detector/weights.pt", "models/classifier/weights.pt", "models/classifier/meta.json")
        missing += [name for name in weights if not (self.root / name).is_file()]
        return {"ready": not missing, "missing": missing, "data_directory": str(self.root)}

    def sources(self) -> list[dict]:
        """List recordings in samples/videos without copying large existing files."""
        paths = [p for name in ("samples", "videos") for p in (self.root / name).glob("*")]
        return [{"path": str(p), "name": f"{p.parent.name}/{p.name}"} for p in sorted(paths)
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]

    def create(self, body: dict) -> dict:
        """Validate source/game order and persist a project without starting work.

        Local sources must exist and be video files. URL support is decided
        by yt-dlp when downloading. A basename is a pipeline identity, so two
        recordings cannot accidentally share labels or cached observations.
        """
        games, layout = self._settings(body)
        source = str(body.get("source", "")).strip()
        kind = body.get("kind", "local")
        if kind not in ("local", "url"):
            raise ValueError("Choose a local recording or a video URL.")
        if kind == "url":
            if not source:
                raise ValueError("Enter a video URL.")
            # Hash the full URL so query-based video IDs stay distinct and filenames stay safe.
            stem = "video_" + hashlib.sha256(source.encode()).hexdigest()[:16]
            video = self.root / "samples" / f"{stem}.mp4"
        else:
            if source.startswith(("\\\\", "//")) or "://" in source:
                raise ValueError("Choose a local video file, not a network address.")
            video = Path(source).expanduser()
            if not video.is_absolute():
                video = self.root / video
            video = video.resolve()
            if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("The local recording does not exist or is not a supported video file.")
            stem = video.stem
        if not re.fullmatch(r"[\w .-]{1,120}", stem) or stem in (".", ".."):
            raise ValueError("Rename the recording using letters, numbers, spaces, dots, underscores or hyphens.")
        with self.lock:
            for existing in self.projects.values():
                if Path(existing["video"]).stem.casefold() == stem.casefold():
                    raise ValueError("This recording already has a project. Open it below; use a different filename for a different recording.")
            project = {"id": uuid.uuid4().hex, "name": stem, "kind": kind, "source": source,
                       "video": str(video), "games": games, "layout": layout,
                       "created": time.time(), "status": "new", "job": {"running": False}}
            project["source_sha256"] = self._source(project)
            project["export_signature"] = None
            if (self.root / "work" / stem / "hands.json").exists():
                self._invalidate_inputs(project, project["source_sha256"])
            self._save(project)
            self.projects[project["id"]] = project
            return self.snapshot(project["id"])

    def _settings(self, body: dict) -> tuple[list[int], str]:
        games = body.get("games")
        if not isinstance(games, list) or not games or any(type(g) is not int or g <= 0 for g in games):
            raise ValueError("Enter at least one positive scoremj game ID in recording order.")
        if len(games) != len(set(games)):
            raise ValueError("Each scoremj game ID must appear only once.")
        layout = str(body.get("layout") or "pml")
        if layout != "pml":
            path = Path(layout).expanduser()
            if not path.is_absolute():
                path = self.root / path
            if not path.is_file() or path.suffix != ".json":
                raise ValueError("The custom layout must be an existing local JSON file.")
            layout = str(path.resolve())
        return games, layout

    def update(self, key: str, body: dict) -> dict:
        """Correct idle project inputs while retaining source video and human labels.

        Game changes rebuild alignment and decode; layout changes also rescan
        the overlay. Existing exports become unavailable until successful analysis establishes
        provenance for the new settings.
        """
        games, layout = self._settings(body)
        with self.lock:
            if any(p["job"].get("running") for p in self.projects.values()) or any(
                    any(j.get("running") for j in state.jobs.values()) for state in self.states.values()):
                raise ValueError("Wait for the running job to finish before changing project settings.")
            project = self.project(key)
            if project["games"] == games and project["layout"] == layout:
                return self.snapshot(key)
            layout_changed = project["layout"] != layout
            project.update(games=games, layout=layout, export_signature=None, status="ready", job={"running": False},
                           needs_prepare=layout_changed, inputs_changed=True)
            work = self.root / "work" / project["name"]
            work.mkdir(parents=True, exist_ok=True)
            (work / "inputs.changed").write_text("Analyze the recording after changing its game IDs or layout.\n", encoding="utf-8")
            if layout_changed:
                project["status"] = "new"
                (self.root / "work" / project["name"] / "overlay.jsonl").unlink(missing_ok=True)
            state = self.states.pop(key, None)
            if state is not None:
                state.close()
            self._save(project)
            return self.snapshot(key)

    def _signature(self, project: dict) -> str | None:
        try:
            from ..layout import Calibration
            source = self._source(project)
            if source is None:
                return None
            cal = Calibration.load(project["layout"], project["video"])
            payload = {"games": project["games"], "layout": project["layout"], "geometry": cal.data,
                       "source_sha256": source}
            return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        except (OSError, ValueError, KeyError):
            return None

    def _source(self, project: dict) -> str | None:
        from ..cache import source_identity
        try:
            # The shared stat-keyed memo avoids rereading a broadcast on each poll.
            return source_identity(project["video"])
        except OSError:
            return None

    def _sync_source(self, project: dict) -> None:
        """Require regeneration when input provenance is missing or changes."""
        current = self._source(project)
        if "source_sha256" not in project or "export_signature" not in project:
            self._invalidate_inputs(project, current)
        elif project["source_sha256"] != current:
            # A first download has no earlier evidence to invalidate.
            if project["source_sha256"] is None and project["status"] in ("new", "preparing") and not project.get("inputs_changed"):
                project["source_sha256"] = current
                self._save(project)
            else:
                self._invalidate_inputs(project, current)

    def _invalidate_inputs(self, project: dict, source: str | None) -> None:
        """Retire unverified outputs without changing the saved human answers."""
        project.update(source_sha256=source, export_signature=None, inputs_changed=True, needs_prepare=True)
        if not project["job"].get("running"):
            project["status"] = "new"
        work = self.root / "work" / project["name"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "inputs.changed").write_text("Prepare and analyze again: recording contents changed or input provenance is missing.\n", encoding="utf-8")
        state = self.states.pop(project["id"], None)
        if state is not None:
            state.close()
        self._save(project)

    def _record_matches(self, project: dict) -> bool:
        try:
            record = json.loads((self.root / "work" / project["name"] / "record.json").read_text(encoding="utf-8"))
            return [game["id"] for game in record] == project["games"]
        except (OSError, ValueError, KeyError):
            return False

    def project(self, key: str) -> dict:
        """Resolve an opaque project ID; request paths never become disk paths."""
        if key not in self.projects:
            raise KeyError("Project not found.")
        return self.projects[key]

    def snapshot(self, key: str) -> dict:
        """Return workflow state with available exports and review counts."""
        with self.lock:
            stored = self.project(key)
            self._sync_source(stored)
            project = deepcopy(stored)
        name = Path(project["video"]).stem
        out = self.root / "out" / name
        valid = bool(project["export_signature"]) and project["export_signature"] == self._signature(project) and self._record_matches(project)
        valid = valid and not any((self.root / "work" / name / marker).exists() for marker in ("calibration.changed", "inputs.changed"))
        project["artifacts"] = [p.name for p in sorted(out.glob("*")) if valid and p.is_file() and ARTIFACT.fullmatch(p.name)
                                and (not p.name.startswith("g") or int(p.name[1:].split(".")[0]) < len(project["games"]))]
        project["stale_exports"] = any(out.glob("g*.json")) and not valid
        project["results_revision"] = [
            [filename, (out / filename).stat().st_mtime_ns, (out / filename).stat().st_size]
            for filename in project["artifacts"] if re.fullmatch(r"g\d+\.json", filename)
        ]
        project["has_fit"] = bool(project["source_sha256"]) and (self.root / "labels" / name / "calib.json").exists() and not project.get("needs_prepare")
        project["has_hands"] = bool(project["source_sha256"]) and (self.root / "work" / name / "hands.json").exists() and not project.get("inputs_changed")
        try:
            if not valid:
                raise ValueError("Exports do not match current inputs.")
            items = json.loads((out / "review.json").read_text(encoding="utf-8"))
            project["open_items"] = len(items)
            project["conflicts"] = sum(i.get("kind") == "conflict" for i in items)
        except (OSError, ValueError):
            project["open_items"] = None
            project["conflicts"] = 0
        project["pending_rebuilds"] = self.review_state(key).pending_rebuilds() if valid and project["has_hands"] else []
        return project

    def review_state(self, key: str):
        """Lazily bind the existing review service to exactly one project."""
        from .server import State
        with self.lock:
            project = self.project(key)
            self._sync_source(project)
            signature = self._signature(project)
            if key in self.states and self.states[key]._configuration_signature != signature:
                self.states[key].reload_calib()
                self.states[key]._configuration_signature = signature
            if key not in self.states:
                self.states[key] = State(Path(project["video"]), self.root / "work", project["layout"], self.root / "out")
                self.states[key]._configuration_signature = signature
                if project.get("inputs_changed"):
                    self.states[key].hands = []
            return self.states[key]

    def artifact(self, key: str, name: str) -> Path:
        """Return only a known generated export, excluding arbitrary workspace files."""
        if not ARTIFACT.fullmatch(name):
            raise ValueError("Unknown export filename.")
        project = self.project(key)
        if name not in self.snapshot(key)["artifacts"]:
            raise FileNotFoundError("These exports are out of date. Analyze the recording with its current settings first.")
        path = self.root / "out" / Path(project["video"]).stem / name
        if not path.is_file():
            raise FileNotFoundError("This export is not available yet.")
        return path

    def results(self, key: str) -> dict:
        """Return native result rows and replay links from provenance-checked JSON exports.

        The browser and CLI share Tenhou URL serialization. HTML export files
        are never parsed or embedded, and stale exports are omitted.
        """
        from ..tenhou6 import editor_url, viewer_url

        project = self.snapshot(key)
        games = []
        files = sorted((f for f in project["artifacts"] if re.fullmatch(r"g\d+\.json", f)),
                       key=lambda f: int(f[1:-5]))
        for filename in files:
            index = int(filename[1:-5])
            data = json.loads(self.artifact(key, filename).read_text(encoding="utf-8"))
            hands = []
            for hi, hand in enumerate(data["log"]):
                kyoku, honba = hand[0][:2]
                wind = ("East", "South", "West", "North")[kyoku // 4 % 4]
                hands.append({"index": hi, "round": f"{wind} {kyoku % 4 + 1}",
                              "honba": honba, "editor_url": editor_url(data, hi)})
            games.append({"index": index, "record_id": project["games"][index],
                          "names": data.get("name", []), "hands": hands,
                          "viewer_url": viewer_url(data), "download": f"/exports/{key}/{filename}"})
        pending = set(project["pending_rebuilds"])
        pending_games = sorted({h["game"] for h in self.review_state(key).hands if h["hand"] in pending}) if pending else []
        return {"games": games, "revision": project["results_revision"],
                "pending_rebuilds": sorted(pending), "pending_games": pending_games}

    def start(self, key: str, action: str) -> dict:
        """Start preparation or analysis once, preserving logs and failures for retry.

        Preparation downloads if needed, then measures geometry. Analysis is
        a separate explicit user action after visual calibration review and
        always retains the CLI's geometry gate. Review facts invalidate
        decode caches through the existing pipeline.
        """
        if action not in ("prepare", "analyze"):
            raise ValueError("Unknown project action.")
        with self.lock:
            if self.processes.closing:
                raise ValueError("The app is closing. Restart it before starting a job.")
            project = self.project(key)
            self._sync_source(project)
            if self._source(project) is None and project["kind"] != "url":
                raise ValueError("Recording not found. Restore the local video before continuing.")
            if any(p.get("job", {}).get("running") for p in self.projects.values()):
                raise ValueError("Another job is running. Wait for it to finish before starting this recording.")
            if any(any(j.get("running") for j in s.jobs.values()) for s in self.states.values()):
                raise ValueError("A review job is running. Wait for it to finish first.")
            if action == "analyze" and not self.snapshot(key)["has_fit"]:
                raise ValueError("Prepare and check the table calibration before analyzing.")
            # Review prefill models otherwise retain VRAM while a fresh child
            # loads another copy for the full pipeline.
            had_models = any(state._models is not None for state in self.states.values())
            for state in self.states.values():
                state.close()
                state._models = None
            self.states.clear()
            if had_models:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            job = {"running": True, "action": action, "started": time.time(),
                   "stage": "Preparing recording" if action == "prepare" else "Checking recording", "log": [], "error": None}
            project.update(status="preparing" if action == "prepare" else "analyzing", job=job)
            try:
                self._save(project)
            except OSError as exc:
                self._save_failed(project, exc)
                raise ValueError(job["error"]) from exc

        def run():
            try:
                source_before = self._source(project)
                args = [sys.executable, "-u", "-m", "video2tenhou.cli"]
                if action == "prepare":
                    if project["kind"] == "url" and not Path(project["video"]).exists():
                        job["stage"] = "Downloading recording"
                        self.runner(args + ["download", "--", project["source"], project["video"]], project)
                    fit_source = self._source(project)
                    job["stage"] = "Measuring table layout"
                    self.runner(args + ["calib", "fit", project["video"], "--calib", project["layout"],
                                        "--work", str(self.root / "work"), "--out", str(self.root / "work" / "calib")], project)
                    if fit_source is None or self._source(project) != fit_source:
                        raise ValueError("The recording changed during preparation. Restore it, then retry.")
                    project["source_sha256"] = fit_source
                    project["status"] = "ready"
                    project["needs_prepare"] = False
                else:
                    command = args + ["convert", project["video"], "--calib", project["layout"],
                                      "--work", str(self.root / "work"), "--out", str(self.root / "out"), "--redo", "decode"]
                    for game in project["games"]:
                        command += ["--game", str(game)]
                    self.runner(command, project)
                    if self._source(project) != source_before:
                        with self.lock:
                            self._sync_source(project)
                        raise ValueError("The recording changed during analysis. Prepare it again, then retry.")
                    project["status"] = "complete"
                    project["export_signature"] = self._signature(project)
                    project["inputs_changed"] = False
                    for marker in ("inputs.changed", "calibration.changed"):
                        (self.root / "work" / project["name"] / marker).unlink(missing_ok=True)
                job["stage"] = "Ready to check the table" if action == "prepare" else "Analysis complete"
            except Exception as exc:
                project["status"] = "interrupted" if self.processes.closing else "failed"
                job["error"] = "Interrupted when the app closed. Retry to resume from cached work." if self.processes.closing else str(exc)
            finally:
                with self.lock:
                    job.update(running=False, finished=time.time())
                    state = self.states.pop(key, None)  # Reload new hands and geometry on the next request.
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
        with self.processes.spawn(args, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                with self.lock:
                    job = project["job"]
                    job["log"] = (job["log"] + [line])[-80:]
                    stage = _progress_stage(line)
                    if stage:
                        job["stage"] = stage
            if process.wait():
                raise RuntimeError(_failure_message(project["job"]["log"]))
