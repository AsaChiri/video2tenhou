"""Profile the real convert command with bounded wall-time instrumentation.

Usage: uv run python tools/profile_conversion.py --profile-output work/profile \
    video.mp4 --game 123 --work work/cold --out out/cold

No caches are removed. The report records whether the recording's work/output
directories were empty. Timings are inclusive and overlapping: never add nested
spans or parallel counterfactuals to estimate total wall time. GPU operations are
not forcibly synchronized; API timings include their normal synchronization only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


def runtime_threads():
    """Read loaded libraries' effective thread counts without importing models.

    Libraries can change process-wide settings during model initialization;
    startup settings alone do not describe later conversion stages.
    """
    counts = {}
    cv = sys.modules.get("cv2")
    torch = sys.modules.get("torch")
    if cv is not None:
        counts["opencv"] = cv.getNumThreads()
    if torch is not None:
        counts.update(torch=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads())
    return counts


class Recorder:
    """Thread-safe inclusive timing totals and flushed progress events.

    High-frequency calls update aggregates only. Stage, hand and solver events
    are written immediately so a failed/interrupted run retains useful evidence.
    A conversion decodes hands serially; its active hand also labels worker calls.
    """

    def __init__(self, output: Path):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.lock = threading.RLock()
        self.started = time.perf_counter()
        self.totals = {}
        self.active_hand = None
        self.active_stage = None

    def event(self, kind, **data):
        """Write one atomic JSONL event with elapsed time and current thread ID."""
        with self.lock:
            self.stream.write(json.dumps(dict(event=kind, elapsed=time.perf_counter() - self.started,
                                              thread=threading.get_ident(), stage=self.active_stage, **data), default=str) + "\n")
            self.stream.flush()

    def add(self, name, elapsed, *, hand=None, failed=False):
        """Accumulate a completed call; failed calls retain elapsed time."""
        with self.lock:
            key = (name, hand, self.active_stage)
            row = self.totals.setdefault(key, dict(name=name, hand=hand, stage=self.active_stage, calls=0, seconds=0., maximum_seconds=0., failures=0))
            row["calls"] += 1
            row["seconds"] += elapsed
            row["maximum_seconds"] = max(row["maximum_seconds"], elapsed)
            row["failures"] += int(failed)

    @contextmanager
    def span(self, name, *, progress=False, detail=None, record_threads=False):
        """Measure a synchronous call without swallowing errors or changing its result."""
        hand = self.active_hand
        start = time.perf_counter()
        failed = False
        if progress:
            settings = {"runtime_threads": runtime_threads()} if record_threads else {}
            self.event("start", name=name, hand=hand, detail=detail, **settings)
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.add(name, elapsed, hand=hand, failed=failed)
            if progress:
                settings = {"runtime_threads": runtime_threads()} if record_threads else {}
                self.event("end", name=name, hand=hand, seconds=elapsed, failed=failed, **settings)

    def wrap(self, function, name, *, progress=False, hand_call=False, stage_call=False):
        """Wrap ordinary functions; hand_call labels serial decode_hand and its workers."""
        @wraps(function)
        def wrapped(*args, **kwargs):
            previous = self.active_hand
            previous_stage = self.active_stage
            if stage_call:
                self.active_stage = name
            if hand_call:
                self.active_hand = (args[0] if args else kwargs["entry"])["hand"]
            try:
                with self.span(name, progress=progress, record_threads=stage_call):
                    return function(*args, **kwargs)
            finally:
                if hand_call:
                    self.active_hand = previous
                if stage_call:
                    self.active_stage = previous_stage
        return wrapped

    def generator(self, function):
        """Time generator creation/next/close, excluding time spent by its consumer."""
        @wraps(function)
        def wrapped(*args, **kwargs):
            iterator = None
            self.event("sample_start", hand=self.active_hand, arguments=args, keywords=kwargs)
            try:
                with self.span("video.sample.create"):
                    iterator = function(*args, **kwargs)
                while True:
                    start = time.perf_counter()
                    hand = self.active_hand
                    failed = False
                    exhausted = False
                    try:
                        value = next(iterator)
                    except StopIteration:
                        exhausted = True
                    except BaseException:
                        failed = True
                        raise
                    finally:
                        self.add("video.sample.next_wait", time.perf_counter() - start, hand=hand, failed=failed)
                    if exhausted:
                        break
                    yield value
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    with self.span("video.sample.close"):
                        iterator.close()
        return wrapped

    def close(self):
        """Close event output after all workers and resource monitoring have stopped."""
        self.stream.close()


class Resources:
    """Sample this process tree and GPUs every five seconds without CUDA barriers.

    Child CPU totals are a lower bound: short-lived children can exit between
    samples. GPU metrics describe the device, including unrelated processes.
    """

    def __init__(self, recorder):
        self.recorder = recorder
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="profile-resources")
        self.cpu = {}
        self.peak_rss = 0
        self.count = 0
        self.gpu_available = True

    def _run(self):
        try:
            import psutil
            parent = psutil.Process()
        except ImportError:
            parent = None
            self.recorder.event("resource_warning", message="psutil unavailable; process tree samples omitted")
        while not self.stop.is_set():
            row = {}
            if parent is not None:
                rss = 0
                try:
                    processes = [parent, *parent.children(recursive=True)]
                except psutil.Error:
                    processes = [parent]
                for process in processes:
                    try:
                        with process.oneshot():
                            cpu = process.cpu_times()
                            self.cpu[(process.pid, process.create_time())] = cpu.user + cpu.system
                            rss += process.memory_info().rss
                    except psutil.Error:
                        continue
                self.peak_rss = max(self.peak_rss, rss)
                row.update(tree_rss_bytes=rss, observed_tree_cpu_seconds=sum(self.cpu.values()), processes=len(processes))
            if self.gpu_available:
                try:
                    result = subprocess.run([
                        "nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if result.returncode:
                        raise RuntimeError(result.stderr.strip())
                    row["gpu_csv"] = result.stdout.strip().splitlines()
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
                    self.gpu_available = False
                    row["gpu_unavailable"] = str(error)
            self.recorder.event("resources", hand=self.recorder.active_hand, **row)
            self.count += 1
            self.stop.wait(5)

    def start(self):
        """Begin background sampling; call finish before closing the recorder."""
        self.thread.start()

    def finish(self):
        """Stop sampling with a bounded wait and return observed resource peaks."""
        self.stop.set()
        self.thread.join(4)
        return dict(samples=self.count, peak_sampled_tree_rss_bytes=self.peak_rss,
                    observed_tree_cpu_seconds_lower_bound=sum(self.cpu.values()))


@contextmanager
def instrument(recorder):
    """Temporarily instrument actual pipeline methods and their imported aliases."""
    from video2tenhou import cli, record, timeline, calm, read, observe, video, overlay
    from video2tenhou.engine import decode, dense, solver
    from video2tenhou.perception.detector import Detector
    from video2tenhou.perception.classifier import Classifier
    from ortools.sat.python.cp_model import CpSolver

    patches = []

    def patch(owner, attribute, wrapper):
        patches.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, wrapper)

    stages = [(cli, "_gate"), (record, "fetch_game"), (timeline, "run_header"),
              (calm, "run_calm"), (read, "run_read"), (observe, "run_observe"),
              (decode, "run_decode"), (cli, "write_outputs")]
    details = [(Detector, "__init__"), (Detector, "predict"), (Detector, "predict_batch"),
               (Classifier, "__init__"), (Classifier, "classify"), (Classifier, "posteriors"),
               (video, "frame_at"), (overlay, "read_overlay"), (overlay, "_tess"),
               (solver.HandModel, "build"), (solver.HandModel, "solve"), (solver.HandModel, "_resolve"),
               (decode.HandDecoder, "_solve")]
    try:
        for owner, name in stages + details:
            label = owner.__name__.replace("video2tenhou.", "") + "." + name
            patch(owner, name, recorder.wrap(getattr(owner, name), label,
                                            stage_call=(owner, name) in stages,
                                            progress=(owner, name) in stages or owner is solver.HandModel and name == "solve"))
        # timeline imports this function directly; patch its alias to the same
        # wrapper rather than wrapping twice or missing the header's calls.
        patch(timeline, "read_overlay", overlay.read_overlay)
        # CpSolver.Solve delegates to solve; instrument only the implementation.
        cp_solve = CpSolver.solve

        @wraps(cp_solve)
        def measured_solve(instance, *args, **kwargs):
            start = time.perf_counter()
            with recorder.span("CpSolver.solve"):
                result = cp_solve(instance, *args, **kwargs)
            recorder.event("cp_search", hand=recorder.active_hand, status=instance.StatusName(result),
                           seconds=time.perf_counter() - start,
                           time_budget=instance.parameters.max_time_in_seconds,
                           workers=instance.parameters.num_workers,
                           objective=instance.ObjectiveValue(), bound=instance.BestObjectiveBound())
            return result

        patch(CpSolver, "solve", measured_solve)
        patch(video, "sample", recorder.generator(video.sample))
        wrapped_dense = recorder.wrap(read.dense_reads, "read.dense_reads", progress=True)
        patch(read, "dense_reads", wrapped_dense)
        patch(dense, "dense_reads", wrapped_dense)
        patch(decode, "decode_hand", recorder.wrap(decode.decode_hand, "decode.decode_hand", progress=True, hand_call=True))
        yield
    finally:
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)


def file_identity(path):
    """Return complete SHA-256 and stat identity; absent optional files are explicit."""
    path = Path(path).resolve()
    if not path.is_file():
        return dict(path=str(path), exists=False)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), exists=True, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())


def run_metadata(args):
    """Capture inputs, effective geometry, versions and initial cache state before timing."""
    from video2tenhou import video
    from video2tenhou.layout import Calibration, fit_path
    from video2tenhou.paths import DATA_DIR, MODEL_DIR, LABEL_DIR
    cal = Calibration.load(args.calib, args.video)
    versions = {}
    for package in ("video2tenhou", "torch", "torchvision", "libreyolo", "ortools", "numpy", "opencv-python-headless", "opencv-python", "psutil"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    cache = {}
    for name in ("work", "out"):
        directory = Path(getattr(args, name)).resolve() / Path(args.video).stem
        files = list(directory.rglob("*")) if directory.exists() else []
        cache[name] = dict(path=str(directory), empty=not any(p.is_file() for p in files),
                           files=sum(p.is_file() for p in files))
    inputs = [Path(__file__), Path(args.video), MODEL_DIR / "detector/weights.pt", MODEL_DIR / "classifier/weights.pt",
              MODEL_DIR / "classifier/meta.json", MODEL_DIR / "detector/meta.json", fit_path(args.video)]
    label_dir = LABEL_DIR / Path(args.video).stem
    inputs.extend(sorted(label_dir.glob("*.jsonl")))
    git = {}
    for label, command in (("revision", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"]), ("diff", ["diff", "HEAD", "--", "src", "tools/profile_conversion.py"])):
        result = subprocess.run(["git", "-c", "safe.directory=" + str(Path.cwd()).replace("\\", "/"), *command], capture_output=True, timeout=10)
        git[label] = hashlib.sha256(result.stdout).hexdigest() if label == "diff" else result.stdout.decode(errors="replace").strip()
    return dict(arguments=vars(args), data_directory=str(DATA_DIR), platform=platform.platform(),
                python=sys.version, logical_cpus=os.cpu_count(), versions=versions, git=git, cache=cache,
                input_files=[file_identity(p) for p in inputs], video=vars(video.probe(args.video)),
                calibration=cal.data, calibration_sha256=hashlib.sha256(json.dumps(cal.data, sort_keys=True).encode()).hexdigest())


def main(argv=None):
    """Run unchanged CLI conversion arguments and always persist the final timing summary."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--profile-output", required=True, type=Path)
    options, convert_args = parser.parse_known_args(argv)
    if convert_args[:1] == ["convert"]:
        convert_args = convert_args[1:]
    recorder = Recorder(options.profile_output)
    resources = Resources(recorder)
    entry_start = time.perf_counter()
    conversion_started = None
    conversion_cpu = None
    failure = None
    resource_summary = {}
    from video2tenhou import cli
    original = cli.cmd_convert

    @wraps(original)
    def profiled(args):
        nonlocal conversion_started, conversion_cpu
        with recorder.span("profile.preparation", progress=True):
            info = run_metadata(args)
            (recorder.output / "metadata.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        setup_start = time.perf_counter()
        with instrument(recorder):
            recorder.add("profile.instrumentation_setup", time.perf_counter() - setup_start)
            import torch
            import cv2
            info["runtime_threads"] = dict(torch=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
                                           opencv=cv2.getNumThreads())
            (recorder.output / "metadata.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            resources.start()
            conversion_started = time.perf_counter()
            conversion_cpu = time.process_time()
            with recorder.span("cli.cmd_convert", progress=True):
                return original(args)

    cli.cmd_convert = profiled
    try:
        return cli.main(["convert", *convert_args])
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        wall = None if conversion_started is None else time.perf_counter() - conversion_started
        cpu = None if conversion_cpu is None else time.process_time() - conversion_cpu
        cli.cmd_convert = original
        if resources.thread.ident is not None:
            resource_summary = resources.finish()
        summary = dict(conversion_wall_seconds=wall, conversion_process_cpu_seconds=cpu,
                       entry_wall_seconds=time.perf_counter() - entry_start, failure=failure,
                       resources=resource_summary, timings=list(recorder.totals.values()),
                       interpretation="Inclusive overlapping timings are not additive. sample.next_wait includes startup/exhaustion and excludes consumer work; GPU calls are not forcibly synchronized. GPU utilization is device-wide, not this process alone. Process CPU seconds/wall seconds gives occupied logical cores (divide by logical CPU count for whole-machine fraction). Sampled child CPU misses short-lived children. Preparation hashes/imports are outside conversion wall time.")
        (recorder.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        recorder.event("finished", failure=failure, conversion_wall_seconds=wall)
        recorder.close()


if __name__ == "__main__":
    main()
