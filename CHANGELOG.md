# Changelog

Changes are recorded here for every release, newest first.

## Unreleased

## 0.2.0 — 2026-09-30

Download **video2tenhou-0.2.0-starter.zip**, extract it into a writable folder,
and run **Start.cmd** on Windows or `sh start.sh` on Linux/macOS. Install uv
and FFmpeg first; the launcher installs Python and selects a compatible
PyTorch runtime. The starter includes the trained models and compiled browser
application, so Node is not needed. See the [quick start](docs/QUICKSTART.md).

### Changes

- One browser studio now owns the project library, recording import,
  calibration, analysis, guided review and exports. Projects can be renamed
  and removed without deleting recordings or human answers.
- The local backend uses Starlette and Uvicorn, with streamed uploads and
  exports, bounded request bodies, same-origin checks and coordinated shutdown.
- Command progress uses Python logging on stderr; JSON reports remain on stdout
  for scripts. Library imports do not install console handlers.
- LibreYOLO inference uses public APIs throughout. CUDA graphs are released
  before input dimensions change; the updated graph policy invalidates earlier
  graph-enabled recognition caches and may increase mixed-shape processing time.
- Guided review prioritizes questions across hands, saves answers immediately
  and rebuilds affected hands in the background. Advanced review provides the
  hand inspector, saved facts and training labels.
- Local recordings and downloaded videos support a selected time range.
  Each clip has its own evidence and saved-answer identity.
- Hand timing uses observed table clearings and scoremj records. The obsolete
  broadcast-text reader, digit/wind templates and separate review pages are removed.
- Reconstruction distinguishes close competing tile choices from unfinished
  confidence checks, shares a bounded checking budget, and replays each hand
  before accepting it for export. Processing-limit notes do not become tile
  questions or confidence proofs.
- Model, decoder, storage and subprocess failures surface explicitly. Cache
  identities track source, calibration, recognition policy, observations,
  hand metadata, score records and saved human facts.
- Calibration replacement is atomic, keeping the previous complete fit readable
  during browser polling and interrupted preparation.
- Release packages include the compiled Vue application and its license notice.
  Starter creation rejects missing frontend assets.
- Training augmentation uses an owned NumPy generator with a recorded seed and
  recipe. Score requests use HTTPX. Large pipeline functions and test fixtures
  are decomposed, and Python lint, formatting and strict type checks pass.

### Updating

Back up `labels/` before updating; it contains human answers and calibration.
Keep the data directory and model bundle together, or set `VIDEO2TENHOU_HOME`
to the existing data directory. Open the project and run **Analyze recording**
to refresh derived evidence under the current contracts. Check calibration
and game order if table timing reports a mismatch.

`video2tenhou web` is the browser entry point. The separate `review` command
and old review-page URLs are removed. Source-checkout users must build the
frontend before serving or packaging; see [frontend development](frontend/README.md).

### Validation

- 750 Python tests pass with 81% coverage. The isolated source archive passes
  749 tests, with one optional local reference-plate test skipped.
- Frontend checks and build, 60 unit tests and 11 browser tests pass.
- Wheel resources, archive contents, public LibreYOLO prediction parity and
  CPU model inference plus the ASGI studio in an extracted starter are checked.
- Fresh-machine CPU/GPU launcher qualification and a representative full-recording
  comparison against independent human annotations were not repeated for this
  release. Existing reconstruction, scoring and replay regressions pass.

## 0.1.3 — 2026-09-28

- Delegated PyTorch/torchvision and CUDA selection to uv's automatic backend
  selection, replacing the starter's fixed CUDA build and development lockfile.
- Added inference-kernel checks, automatic CPU fallback, explicit-device errors
  and runtime repair on relaunch.
- Improved Windows prerequisite discovery and setup diagnostics.
- Consolidated hashing, atomic writes, detector metadata, evidence loading and
  review-job scheduling; removed obsolete paths and duplicate implementations.
- Required explicit strength for imported starting/final hand annotations and
  fixed review questions for melds with unknown sources.
- Kept model files unchanged from 0.1.2.

## 0.1.2 — 2026-09-27

- Accepted video URLs supported by yt-dlp, replacing Twitch-only source checks.
- Derived stable recording filenames from complete URLs, including query parameters.
- Passed download URLs as literal command arguments and updated workflow tests.
- Removed the legacy `twitch` source kind; API clients use `kind: "url"`.
- Kept models and locked dependencies unchanged.

## 0.1.1 — 2026-09-27

- Reduced detector input copies and prepared the next input during GPU processing.
- Widened compatible Torch/torchvision, NumPy, OpenCV and LibreYOLO version ranges.
- Recorded external tool paths and versions in conversion profiles.
- Kept model files unchanged; reanalysis refreshed optimized recognition caches
  while preserving saved human answers.
