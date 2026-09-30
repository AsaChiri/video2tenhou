# Maintenance guide

## Contracts and module map

Module docstrings describe the stage and its assumptions; public API docstrings
describe coordinates, seat conventions, effects and failure behavior. Read them
alongside the detailed [design](DESIGN.md), rather than treating cached JSON as
an undocumented source of truth.

Before adding an implementation, check the standard library, installed tools and
existing project functions. Share code when its inputs, behavior and failure
guarantees match; do not combine algorithms merely because their loops look alike.
Remove obsolete callers and wrappers when consolidating, and check CLI entry
points, browser callbacks, saved formats and tests before declaring code dead.
Historical behavior and backward compatibility are not reasons to retain an
implementation. Require a concrete current use; update owned callers and fixtures
to the current contract, and regenerate derived data instead of maintaining
obsolete formats. Keep human evidence intact and reject ambiguous inputs rather
than silently changing their meaning.

| Modules | Responsibility / boundary |
|---|---|
| `paths` | Immutable package assets vs writable `VIDEO2TENHOU_HOME`. |
| `files` | Streaming file digests and atomic UTF-8/JSON publication using standard-library primitives. |
| `video` | Download, probe, normalized BGR frames; external decoder errors propagate. |
| `layout`, `calibfit` | Frame/region transforms and per-video fitting; failing borders block analysis. |
| `record`, `timeline` | Normalize authoritative results and align pond-clearing windows to hands. |
| `calm`, `read`, `observe` | Find still intervals, retain tile posteriors, combine repeated evidence. |
| `perception.detector`, `classifier`, `reader` | Local model inference, geometry and tile-row structure. |
| `engine.ponds`, `melds`, `indicators`, `calls`, `turns` | Persistent visible objects and chronological events. |
| `engine.hand`, `dense` | Hand/seat alignment and targeted closer readings. |
| `engine.rules`, `solver`, `scoring` | Tile conservation, legal hand transitions and result checks. |
| `engine.decode`, `review` | Stage orchestration, human constraints and unresolved evidence. |
| `engine.confidence` | Shared threshold policy for solver certificates and review questions. |
| `engine.assemble`, `tenhou6` | Encode exports and replay every hand before accepting it. |
| `cli`, `tool` | Script interface and local browser project workflow. |
| `logging_setup` | Command-scoped logging: progress on stderr, JSON results on stdout, and restoration of an embedding application's handlers. Library imports do not configure logging. |
| `tool.server`, `http`, `review_routes` | Starlette ASGI routes, loopback and origin policy, bounded streaming and Uvicorn lifecycle. Blocking workspace/model operations run in worker threads. |
| `tool.processes` | Coordinate subprocess shutdown using pywin32 Job Objects and psutil on Windows, and standard-library process groups on POSIX. |
| `tool.rebuild`, `tool.check_calibration` | Named child-process commands for reconstruction and border checks, replacing inline interpreter scripts. |
| `train`, `eval`, `benchmark` | Annotation datasets, training, held-out metrics and throughput comparisons. |

Seats in engine evidence are wind letters for the current hand; site records
and Tenhou arrays use the hanchan's starting-seat order. Conversion belongs in
the explicit seat-mapping helpers. Times are seconds in the source recording.
Tile strings distinguish red fives (`0m/0p/0s`) from ordinary fives; never collapse
them for tile-inventory accounting. A missing observation is not proof of absence.

## Dependency environments

The launchers use uv directly: `uv venv --allow-existing` provides Python 3.12,
then `uv pip install --torch-backend auto --upgrade-package torch
--upgrade-package torchvision --editable .` resolves a matched runtime for the
machine. There is no fixed CUDA index or exact PyTorch version in the starter
setup. uv owns hardware/driver detection, package resolution, downloads and
caching. Keep uv current as new GPU generations are released.

The starter uses `.venv-runtime`, separate from the development `.venv`.
`UV_PROJECT_ENVIRONMENT` and `uv run --no-sync` launch that selected environment
without applying the development lockfile. Every launch checks for a compatible
current Torch/torchvision pair; installed packages and cached downloads are reused
where possible. `UV_OFFLINE=1` uses uv's cache without network access (a successful
initial setup is required). Interrupted installations can be retried by relaunching.

Before opening the studio, the application checks convolution, matrix multiplication
and torchvision NMS. The detector and classifier share the same checked device.
An automatic GPU check failure falls back to CPU with a warning; explicit device
choices fail visibly. This catches unsupported GPU architectures even when CUDA
reports that a GPU is available.

| Package | Allowed range | Boundary |
|---|---|---|
| Torch | `>=2.6,<3` | Supports the checkpoint-loading and inference APIs in use; excludes a new major API. |
| torchvision | `>=0.21,<0.30` | Its dependency selects the matching Torch version. |
| NumPy | `>=2.5.3,<3` | Starts at the tested numerical baseline; excludes a new major ABI. |
| OpenCV | `>=5.0.0.93,<6` | Starts at the tested image-processing baseline; excludes a new major API. |
| LibreYOLO | `>=1.5,<1.6` | Allows patch updates within the validated YOLO9 prediction series; validate inference and graph behavior before widening. |

These ranges express installation compatibility, not a claim that every version
has identical output or speed. Numerical runtime changes invalidate recognition
caches. Validate evidence, exports and workload performance when changing stacks.
The lockfile remains a development/CI snapshot, not the starter's runtime policy.

Set `VIDEO2TENHOU_DEVICE=cpu` before launching to install and use a CPU build.
Advanced users can set `UV_TORCH_BACKEND` to a backend supported by their installed
uv; normal launches default to `auto`. Neither option edits the project or lockfile.
A manual environment can use the same adaptive resolver:

```powershell
uv venv --python 3.12 work/runtime
uv pip install --python work/runtime/Scripts/python.exe --torch-backend auto .
& ./work/runtime/Scripts/video2tenhou.exe web
```

On Linux use `work/runtime/bin/python` and `work/runtime/bin/video2tenhou`.
Keep a separate `VIDEO2TENHOU_HOME` with the same model bundle and copies of the
review inputs for comparisons. Record package/tool versions and follow the
[benchmark procedure](PERFORMANCE.md).

## Data and caches

Public source includes code, built-in layouts/templates, UI art, small curated
fixtures and guides. `samples/`, `videos/`, `models/`, `weights/`, `labels/`,
`work/`, `out/`, logs and build output are local and ignored. Keep backups of
labels; answers are input data. Do not restore private datasets into Git to make
a test pass.

Imported `haipai` and `final_hand` annotations carrying `source` must explicitly
set boolean `soft`: true contributes uncertain evidence, false fixes a
human-confirmed hand. Unsourced review answers default to confirmed. Source names
never determine annotation strength, and ambiguous imported annotations are
rejected without rewriting the saved journal.

Stage data lives under `work/<video>/`: table timing and hand windows, calm intervals,
per-hand reads, aggregated observations, dense reads and decoded hands. Results
live under `out/<video>/`. Stage caches are accelerators, not independently
versioned interchange formats. When changing evidence semantics, invalidate
affected downstream caches or rerun with `--redo`/`--force`; preserve labels.

Observation completion records bind the actual reading files, their manifest,
relevant calm intervals, voting settings and output digest. An interruption
between reading and voting therefore cannot make stale observations reusable
on resume. Review rebuilds verify sparse and voted evidence before writing a
hand; after changing recognition models, use **Analyze recording** to refresh
that evidence before rebuilding with saved answers.
Decoded results also bind the observation and provenance files by content hash,
along with hand metadata, the authoritative result and that hand's saved facts.
An interruption after voting cannot make an older reconstructed hand reusable;
only complete replacement files become visible.

Studio manifests under `work/projects/` bind source paths, ordered game IDs,
layout selection and an export signature. Exports are only offered when the
source content, those inputs and the effective calibration match their provenance.
Changing project settings preserves labels but requires analysis; changing
layout requires preparation. A `calibration.changed` or
`inputs.changed` marker blocks direct review rebuilds until full analysis
refreshes the upstream evidence. Replacing or removing a recording also requires
preparation again, preserving saved answers. Polling reuses a source digest keyed
by file identity, size and timestamps; conversion stages verify content afresh.
Evidence clips include source and geometry identities in their filenames, and
calibration edits clear remembered border checks.

`tool.server` mounts `tool.review_routes` under each project URL. Review endpoints
use individual Starlette routes, typed hand indices and method validation;
there is no review path dispatcher. Synchronous evidence handlers run in the
framework's worker pool. Review writes check job exclusion before reading the
bounded body, then recheck under the workspace lock before applying a change.
The server streams upload bodies and accepts only local, same-origin writes
with its application header. `tool.workflow`
serializes processing jobs and persists interrupted/failed states. Project
manifests are atomically replaced; brief Windows reader contention has a bounded
retry. A failed save retains the prior complete manifest and exposes a retryable,
non-running job rather than claiming durable completion.
`tool.review_routes` handles calibrated evidence and answers; its revision endpoint
lets open review tabs notice externally rebuilt files without discarding
unsaved edits. Model/encoder commands run through `ProcessOwner` so a server
shutdown reaps owned descendants instead of leaving an invisible job running.
Changing calibration must trigger fresh readings for the affected regions.

## Tests

### Required code checks

Install the locked development tools with `uv sync --frozen` and
`npm --prefix frontend ci`. Run these checks before submitting changes:

```console
uv run ruff format --check .
uv run ruff check .
uv run ty check --error-on-warning
npm --prefix frontend run check
```

Use `uv run ruff format .` and `npm --prefix frontend run format` to apply
formatting. Python linting enables Ruff `ALL`; the only approved exceptions
are `D203`, `D213`, and `COM812` (conflicting docstring/formatter conventions),
and `S101`, `PLR2004`, and `SLF001` in tests (pytest assertions, literal numeric
expectations, and direct checks of internal behavior, approved by the maintainer).
`ARG001` and `ARG002` are approved in tests because test doubles retain unused
named parameters to preserve keyword-call and protocol compatibility with the
components they replace. Remove unnecessary fixtures and unused parameters
from ordinary helpers instead of copying this pattern into application code.
`S311` is also approved in `train/data.py` for non-security crop sampling.
Classifier augmentation uses an owned NumPy generator and needs no RNG exception.
`S603` is approved in `video.py`, `cli.py`, and `tool/processes.py` for
application-owned subprocess argument lists with shell execution disabled.
FFmpeg/FFprobe resolve to absolute executable paths; Python commands use the
current interpreter. Keep user values as separate arguments and retain the
process owner's launch-option validation.
`S603` is also approved in `tools/check_dist.py`, `tools/profile_conversion.py`,
and the video, reading, dense-prefetch and launcher tests for fixed tool commands
and repository-owned fixtures. Launcher tests invoke the platform shell with
the repository launcher and fixed test arguments.
`S603` and `PLC0415` exceptions use rule-specific inline `noqa` annotations,
so newly introduced calls and imports remain checked. `PLC0415` is approved
for lazy runtime/model imports in `benchmark.py`, `cli.py`, `engine/decode.py`,
`eval.py`, `perception/detector.py`, `tool/review_state.py`, and `tool/workflow.py`;
runtime diagnosis in `perception/device.py`; training validation before model
loading in `train/train_libreyolo.py`; and profiling after environment setup in
`tools/profile_conversion.py`. Ordinary test and packaging imports stay at module
scope. Ruff also checks for unused `noqa` annotations.
LibreYOLO inference uses public `predict` and `release_graphs` APIs, with no
private-library-access exception. Graph lifecycle tests cover shape changes,
failures and serialized model access.
The checks cover application code,
training code, tools, tests, and frontend configuration. Do not add exclusions,
rule suppressions, type-check skips, or failure baselines without maintainer
approval. Python's existing inline type-ignore comments are not honored.

These commands are mandatory workflow steps, with no warning-only or
continue-on-error paths. The frontend build also runs its checks first.
GitHub `main` requires `test (ubuntu-latest)` and `test (windows-latest)` from
GitHub Actions, with an up-to-date branch, including for administrators. Keep
those protection settings aligned with the workflow job names; workflow files
alone cannot set that repository policy.

The initial Ruff ALL and ty rollout exposes existing Python diagnostics.
Those diagnostics fail CI and must be fixed rather than hidden by a baseline.

### Regression suite

```console
npm --prefix frontend ci
npm --prefix frontend run build
uv run pytest -q --cov --cov-report=term-missing --cov-report=html
uv build
uv run python tools/check_dist.py
uv run python tools/check_source_release.py
```

The suite covers synthetic decoder frames, calibration transforms, pond-based hand timing, observation/cache behavior, rule/solver cases, replay rejection, review
answers and HTTP project workflows. Cross-module tests verify that an authoritative
record becomes a validated export, and illegal reconstructions become explicit
conflicts. The week_11 fixture protects the observed draw-order failure. Use
coverage gaps to identify missing behavior, not to add assertions repeating code.
See the [test guide](../tests/README.md) for subsystem directories and focused runs.

The wheel smoke check imports the extracted distribution outside the repository
and verifies templates, calibration and UI assets. CI runs without local labels,
videos or weights. Optional local-data tests must skip explicitly. GPU benchmarks
are separate from CI: capture hardware, model hashes, warmup, workload and output
parity. Do not compare cached reruns with cold first runs as a speedup claim.
The source-release check runs the tests shipped inside the sdist in an empty
workspace, so accidental dependence on private labels or local caches is visible.

## Frontend

Frontend source is in `frontend/`. Before packaging UI changes, run
`npm --prefix frontend ci`, `npm --prefix frontend test`, and
`npm --prefix frontend run build`. Generated static assets are ignored by Git
and included in release distributions by the package builder. Build them before
running Python web tests or serving a source checkout.
See [frontend development](../frontend/README.md) for the component
and backend module boundaries. The single application is launched with
`video2tenhou web`; there is no separate review server or browser command.


## Training

Label examples in the browser using full-frame coordinates, including difficult
views. Keep complete hand windows so validation can hold out entire hands.
The default held-out hand IDs live in `train/data.py`; revise the split for a
new recording rather than splitting adjacent frames randomly.

```console
uv run python -m video2tenhou.train.data samples/recording.mp4 --out work/datasets
uv run python -m video2tenhou.train.face_data --human work/datasets/detector --video samples/recording.mp4 --out work/datasets/faces
uv run python -m video2tenhou.train.train_libreyolo --data work/datasets/faces/data.yaml --base models/external/LibreYOLO9s.pt --out work/runs/detector
uv run python -m video2tenhou.train.train_classifier
uv run python -m video2tenhou.eval detector
uv run python -m video2tenhou.eval perception samples/recording.mp4
```

These relative-path examples assume the checkout is also the data directory.
With a separate `VIDEO2TENHOU_HOME`, dataset-building paths and the detector's
input paths are still relative to the launch directory; classifier defaults
and model outputs resolve under the data directory. Pass absolute paths to keep
the stages together. For example, from the checkout in PowerShell:

```powershell
$env:VIDEO2TENHOU_HOME = 'D:/Mahjong'
uv run python -m video2tenhou.train.data D:/Mahjong/samples/recording.mp4 --work D:/Mahjong/work --out D:/Mahjong/work/datasets
uv run python -m video2tenhou.train.face_data --human D:/Mahjong/work/datasets/detector --video D:/Mahjong/samples/recording.mp4 --work D:/Mahjong/work --out D:/Mahjong/work/datasets/faces
uv run python -m video2tenhou.train.train_libreyolo --data D:/Mahjong/work/datasets/faces/data.yaml --base D:/Mahjong/models/external/LibreYOLO9s.pt --out D:/Mahjong/work/runs/detector
uv run python -m video2tenhou.train.train_classifier --data D:/Mahjong/work/datasets/classifier --out D:/Mahjong/models/classifier
uv run python -m video2tenhou.eval detector
uv run python -m video2tenhou.eval perception D:/Mahjong/samples/recording.mp4
```

Use equivalent absolute paths on other systems; quote paths containing spaces.

Detector training starts from the specific
[LibreYOLO9-S checkpoint](https://huggingface.co/LibreYOLO/LibreYOLO9s).
The face dataset preserves whole-hand validation groups and maps reviewed
visible-tile boxes to one localization class. Training writes a separate run;
it does not replace deployed models. Follow [detector qualification and
deployment](DETECTOR_BACKENDS.md) to evaluate the complete detector/classifier
pair, export its inference state and retain matching metadata and provenance.
Classifier training initializes ResNet18 from ImageNet; inference instead loads
the complete local checkpoint. Retain classifier `meta.json` with its weights.
New classifier training uses a dataset-owned NumPy `Generator` seeded with zero;
metadata records the seed, bit generator and `numpy-generator-pcg64-v1`
augmentation recipe. This changes the training sample sequence from older
Python/global-NumPy augmentation. Validation performs no random augmentation,
and existing checkpoints remain unchanged. Context refinement uses Python and
PyTorch randomness, so it no longer seeds the unused global NumPy generator.
Evaluate every view, especially hand cameras, before replacing release models.

## Release boundaries

This local server has no remote account system or production hosting layer.
Keep it bound to loopback. Browser writes must validate origins, constrain file
paths and serialize GPU work. Do not let a filename become a command string.
Download and analysis jobs must surface failures rather than presenting partial
files as complete results. See [release preparation](RELEASING.md).
