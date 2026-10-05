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
| `files` | File and text digests, and atomic UTF-8/JSON publication using standard-library primitives. |
| `cache` | One persisted content digest per recording, keyed by file identity. |
| `commands`, `video` | Absolute executable paths; download, probe, normalized BGR frames; external decoder errors propagate. |
| `layout`, `calibfit` | Frame/region transforms and per-video fitting; failing borders block analysis. |
| `record`, `timeline` | Normalize authoritative results and align pond-clearing windows to hands. |
| `calm`, `read`, `observe` | Find still intervals, retain tile posteriors, combine repeated evidence. |
| `perception.detector`, `yolo9_direct`, `classifier`, `reader` | Local model inference, the pinned direct YOLO9 input path, geometry and tile-row structure. |
| `engine.ponds`, `melds`, `indicators`, `calls`, `turns`, `pond_evidence` | Persistent visible objects and chronological events. |
| `engine.hand`, `dense` | Hand/seat alignment and targeted closer readings. |
| `engine.rules`, `solver`, `scoring` | Tile conservation, legal hand transitions and result checks. |
| `engine.events`, `reconstruct`, `score_reconcile` | Decoder stages: discards, calls, dead wall, turn order and riichi; the constraint program, rereads, repair and kan indicators; the site's result. |
| `engine.questions`, `review` | Every question, note and ignored answer; human facts as constraints and the confidence rows. |
| `engine.decode` | Stage orchestration, the decode artifact and workspace decode runs. |
| `engine.confidence` | Shared threshold policy for solver certificates and review questions. |
| `engine.assemble`, `validation`, `tenhou6` | Encode exports and replay every hand before accepting it. |
| `export` | Stage 6: atomic logs, confidence, review queue and report (one-based hand numbers, collapsed Diagnostics), and `export-inputs.json` naming the decode files each export used. |
| `cli`, `tool` | Script interface and local browser project workflow. |
| `logging_setup` | Command-scoped logging: progress on stderr, JSON results on stdout, and restoration of an embedding application's handlers. Library imports do not configure logging. |
| `tool.server`, `http`, `review_routes` | Starlette ASGI routes, loopback and origin policy, bounded streaming and Uvicorn lifecycle. Blocking workspace/model operations run in worker threads. |
| `tool.workflow`, `review_state` | Project manifests and the workspace's single job slot; per-recording evidence, answers and hand freshness. |
| `tool.processes` | Coordinate subprocess shutdown using pywin32 Job Objects and psutil on Windows, and standard-library process groups on POSIX. |
| `train`, `eval` | Annotation datasets, training and held-out metrics. |

Seats in engine evidence are wind letters for the current hand; site records
and Tenhou arrays use the hanchan's starting-seat order. Conversion belongs in
the explicit seat-mapping helpers. Times are seconds in the source recording.
Tile strings distinguish red fives (`0m/0p/0s`) from ordinary fives; never collapse
them for tile-inventory accounting. A missing observation is not proof of absence.

## Dependency environments

The launchers use uv directly. Setup runs `uv venv --allow-existing --python 3.12
.venv-runtime`, then `uv pip install --torch-backend auto --editable .` into that
environment, which resolves a matched runtime for the machine. There is no fixed
CUDA index or exact PyTorch version in the starter setup. uv owns hardware/driver
detection, package resolution, downloads and caching. Keep uv current as new GPU
generations are released.

The starter uses `.venv-runtime`, separate from the development `.venv`. A
completed setup copies `pyproject.toml` to `.venv-runtime/video2tenhou-pyproject.toml`.
While that copy matches, launches skip setup and start the studio with
`uv run --no-sync` under `UV_PROJECT_ENVIRONMENT=.venv-runtime`, without network
access or the development lockfile. Setup runs again when the environment or its
record is missing, including after an interrupted installation, or when
`pyproject.toml` changed; it installs missing requirements and keeps a PyTorch
that already satisfies them. `Start.cmd --update`, `sh start.sh --update` or
`VIDEO2TENHOU_UPDATE=1` also upgrade PyTorch and torchvision. A different PyTorch
build changes recognition identities, so the next analysis of each recording
reads its video again. `UV_OFFLINE=1` limits a setup or update to uv's cache.

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
| LibreYOLO | `>=1.5,<1.6` | The series whose private graph dispatch and postprocessing the direct input path calls; `tests/perception/test_yolo9_direct.py` must pass before widening. |

These ranges express installation compatibility, not a claim that every version
has identical output or speed. Numerical runtime changes invalidate recognition
caches. Validate evidence, exports and workload performance when changing stacks.
The lockfile remains a development/CI snapshot, not the starter's runtime policy.

Set `VIDEO2TENHOU_DEVICE=cpu` to run on CPU; when setup runs (first launch or
`--update`) it also selects a CPU PyTorch build. `UV_TORCH_BACKEND` overrides the
setup backend for a backend supported by the installed uv; setup defaults to
`auto`. Neither option edits the project or lockfile.
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

A recording's content is hashed once. `cache.source_identity` keeps one SHA-256
per resolved path in `work/source-digests.json`, keyed by size, modification and
change times, file index and device, and hashes again only when one of those
changes; contents rewritten in place with all of them restored are outside this
contract. Stages and the studio take digests from it and never hash videos
themselves. The first call after a change can take minutes, so call it outside
locks.

Each stage's completion manifest records digests of the text it published, and
downstream stages bind to that manifest instead of re-reading and hashing
upstream files. `reads/<hand>/done.json` records the source, recognition,
geometry and sampling identities and one digest per region file. The observation
record `obs/provenance/<hand>.json` binds the digest of that reading manifest,
the relevant calm intervals, the class list, the sparse retention policy and the
voting settings, and records the digest of the votes it wrote. A decode binds the
digest of the observation record, the dense retention policy, the hand's
metadata, its site result and its facts (dismissals excluded). A rewrite removes
the old manifest first and writes the new one last, so an interruption cannot
leave a manifest describing files it did not produce. Manifests are trusted: a
hand edit to a cached file that leaves its manifest in place is not detected.
After editing caches by hand, delete the manifest or rerun with `--force` or
`--redo`. Review rebuilds verify sparse and voted evidence before writing a
hand; after changing recognition models, use **Analyze recording** to refresh
that evidence before rebuilding with saved answers.

Studio manifests under `work/projects/` bind source paths, ordered game IDs,
layout selection and an export signature. Exports are only offered when the
source content, those inputs and the effective calibration match their provenance.
Changing project settings preserves labels but requires analysis; changing
layout requires preparation. A `calibration.changed` or `inputs.changed` marker
blocks review rebuilds until a completed `convert` refreshes the upstream
evidence and removes it; the markers are the only record of that state.
Replacing or removing a recording also requires preparation again, preserving
saved answers. The studio computes recording digests in background threads; a
project whose digest is not yet known reports **Checking recording** and offers
no exports. Evidence clips include source and geometry identities in their
filenames and are encoded to a temporary file before publication. Border checks
are kept in `work/<video>/border-checks.json` with the geometry they measured
and are shown only while that geometry is current.

`tool.server` mounts `tool.review_routes` under each project URL. Review endpoints
use individual Starlette routes, typed hand indices and method validation;
there is no review path dispatcher. Synchronous evidence handlers run in the
framework's worker pool; frame seeks, model loading, clip encoding and digests
never run while the workspace or review lock is held. Review writes check job
exclusion before reading the bounded body, then recheck under the workspace lock
before applying a change. The server streams upload bodies and accepts only
local, same-origin writes with its application header. Responses are `no-store`
except content-hashed bundles and tile art, which are cacheable. Evidence
requests never build the plate, run checks or change answers.

`tool.workflow` owns one job slot for the whole workspace: preparation, analysis,
hand updates and calibration fits and checks run `video2tenhou` CLI children
(preparation downloads or trims first when needed) through the shared
`ProcessOwner`, which forces UTF-8 child output and reaps owned descendants when
the server shuts down. CLI commands other than
`web` end with at most one JSON line on stdout: `{"result": ...}`, or
`{"error": ...}` with any partial result and `"unexpected": true` for a failure
the message cannot explain. The studio shows only that message.
`GET /api/job[?project=<key>]` reports the job and the open project's review
revision; the developer log, both output streams of the latest job, is served
only on request from `GET /api/projects/<key>/log`. Preparation and analysis persist interrupted and
failed states. Project manifests are atomically replaced; brief Windows reader
contention has a bounded retry. A failed save retains the prior complete manifest
and exposes a retryable, non-running job rather than claiming durable completion.

A hand is pending when it has no decode by the current decoder version, when its
saved decode context differs from the current hand metadata, site result and
facts (dismissals excluded), or when `out/<video>/export-inputs.json` does not
name its current decode file. A `dismiss` fact is review state: it hides one
question by its stable id and never enters reconstruction or its binding. Saved
`note` facts from earlier versions dismiss a question only when exactly one
question has the same text; the journal is never rewritten. Changing calibration
must trigger fresh readings for the affected regions.

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
formatting. Python linting enables Ruff `ALL`. The maintainer approved these
configured exceptions: `D203`, `D213` and `COM812` (conflicting docstring and
formatter conventions); `D401` (noun-phrase docstring summaries); `EM101`,
`EM102` and `TRY003` (exception messages written inline); and up to eight
arguments per function (`[tool.ruff.lint.pylint] max-args = 8`). Tests also
ignore `S101`, `PLR2004`, `SLF001`, `ARG001`, `ARG002`, `D`, `ANN` and `TC`:
pytest assertions, literal expectations, direct checks of internal behavior,
test doubles that keep unused named parameters to match the interfaces they
replace, and test names that document themselves. Tests use postponed
annotations (`from __future__ import annotations`) with type imports at module
scope. Remove unnecessary fixtures and unused parameters from ordinary helpers
instead of copying the test exceptions into application code. `S311` is also
approved in `train/data.py` for non-security crop sampling. Classifier
augmentation uses an owned NumPy generator and needs no RNG exception.

Any other exception is a rule-specific inline `noqa` annotation, with a short
reason where the rule does not make it obvious, so newly introduced calls and
imports remain checked. Ruff also checks for unused `noqa` annotations. The
approved annotations are:

- `S603` in `video.py`, `cli.py` and `tool/processes.py` for application-owned
  subprocess argument lists with shell execution disabled. FFmpeg and FFprobe,
  including evidence-clip encoding, resolve to absolute executable paths through
  `commands.executable`; Python commands use the current interpreter. Keep user
  values as separate arguments and retain the process owner's launch-option
  validation.
- `S603` in `tools/check_dist.py`, `tools/profile_conversion.py` (read-only git
  queries), `tools/check_source_release.py` (git file listing and the shipped
  smoke tests), and the video, reading, dense-prefetch, engine import-isolation
  and launcher tests for fixed tool commands and repository-owned fixtures.
  Launcher tests invoke the platform shell with the repository launcher and
  fixed test arguments.
- `PLC0415` for lazy runtime and model imports in `cli.py`, `engine/decode.py`,
  `eval.py`, `perception/detector.py`, `tool/review_state.py` and
  `tool/workflow.py`; runtime diagnosis in `perception/device.py`; training
  validation before model loading in `train/train_libreyolo.py`; HTTPX only when
  querying the score site in `record.py`, so the engine imports result types
  without the HTTP client; and yt-dlp only when parsing user time bounds in
  `video.py`. Ordinary test and packaging imports stay at module scope.
- `SLF001` on the two calls in `perception/yolo9_direct.py` to LibreYOLO 1.5's
  private `_forward_graphed` and `_postprocess`. LibreYOLO stays pinned to the
  validated 1.5 series. `tests/perception/test_yolo9_direct.py` fails if their
  signatures change and compares the direct inputs and detections with public
  `predict` (a fresh model on CPU; the installed checkpoint and reference
  recording when present). Each padded input shape owns its own model, so no
  capture is released while predicting; tests cover per-shape ownership,
  replaced weights and serialized model access.

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

Ruff ALL and strict typing pass across the repository. New diagnostics fail CI
and must be fixed rather than hidden by a baseline.

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
Recorded-hand reconstructions are marked `slow`; `uv run pytest -m "not slow"` is
a quick local loop. CI runs everything.

The wheel smoke check imports the extracted distribution outside the repository
and verifies templates, calibration and UI assets. CI runs without local labels,
videos or weights. Optional local-data tests must skip explicitly. GPU benchmarks
are separate from CI: capture hardware, model hashes, warmup, workload and output
parity. Do not compare cached reruns with cold first runs as a speedup claim.
The source-release check requires the sdist to contain exactly the tracked files
(except the repository-only `.github/`, `.editorconfig`, `.gitattributes` and
`CLAUDE.md`), the built UI and `PKG-INFO`. It then runs a few shipped tests from
the extracted archive with an empty data directory, so accidental dependence on
private labels or local caches is visible; the full suite runs from the checkout.

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
visible-tile boxes to one localization class; human detector labels of any
other class are rejected. Regions with reviewed face-down (`X`) tiles stay out
of the detector dataset; their classifier crops are kept. Training writes a separate run;
it does not replace deployed models. Follow [detector qualification and
deployment](DETECTOR_BACKENDS.md) to evaluate the complete detector/classifier
pair, export its inference state and retain matching metadata and provenance.
Classifier training initializes ResNet18 from ImageNet; inference instead loads
the complete local checkpoint. Retain classifier `meta.json` with its weights.
Classifier training uses a dataset-owned NumPy `Generator` seeded with zero;
metadata records the seed, bit generator and `numpy-generator-pcg64-v1`
augmentation recipe. Validation performs no random augmentation. Context
refinement uses Python and PyTorch randomness.
Evaluate every view, especially hand cameras, before replacing release models.

## Release boundaries

This local server has no remote account system or production hosting layer.
Keep it bound to loopback. Browser writes must validate origins, constrain file
paths and serialize GPU work. Do not let a filename become a command string.
Download and analysis jobs must surface failures rather than presenting partial
files as complete results. See [release preparation](RELEASING.md).
