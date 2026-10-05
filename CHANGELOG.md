# Changelog

Changes are recorded here for every release, newest first.

## Unreleased

## 0.3.0 — 2026-10-05

Download **video2tenhou-0.3.0-starter.zip**, extract it into a writable folder,
and run **Start.cmd** on Windows or `sh start.sh` on Linux/macOS. Install uv
and FFmpeg first; the first launch installs Python and a compatible PyTorch
runtime, and later launches start without network access. The starter includes
the trained models and compiled browser application, so Node is not needed. See
the [quick start](docs/QUICKSTART.md).

Conversion is several times faster, every reconstructed tile decision is
checked, and the studio shows reviewers only what they can act on.

### Changes

- Recordings are hashed once. `work/source-digests.json` keeps each recording's
  SHA-256 keyed by its path, size, timestamps, file index and device; stages,
  rebuilds and the studio reuse it instead of rehashing the broadcast in every
  process. Stage manifests record digests of what they write, and downstream
  stages bind to those manifests instead of hashing files again; a rewrite
  removes the old manifest first and writes the new one last. A hand edit to a
  cached file that leaves its manifest in place is no longer detected.
- Recognition does less work for the same evidence. The detector feeds YOLO9
  directly through two private LibreYOLO 1.5 members (pinned and covered by
  tests); boxes, confidences and readings are bit-identical to the public API on
  2,193 recorded calls. With CUDA graphs each padded input shape keeps its own
  model and only replays, instead of releasing and recapturing on every shape
  change. The sampler decodes up to 16 frames ahead, detector crops are
  letterboxed while the GPU works, calm scoring uses single-threaded OpenCV
  calls and the classifier normalizes crops on the device through a lookup
  table. `convert` loads one detector and classifier; rebuilds load models only
  when a hand rereads the video, and engine imports no longer load PyTorch,
  HTTPX or yt-dlp. Recognition identities are unchanged.
- Confidence checks are grouped: one CP-SAT search certifies every pending
  decision at once or finds a close witness, replacing one search per decision
  under a shared 10 s budget. A hand has at most two certification passes of up
  to 60 s each, on eight workers. Far fewer decisions end unresolvable, more
  draws, starting hands and discards are reported as close alternatives, and
  ambiguous draws again trigger dense hand rereads. Easy hands finish sooner;
  hard hands can use both full passes.
- The decoder runs as stages (`events`, `reconstruct`, `score_reconcile`,
  `questions`). Decode artifacts carry `notes` for the reviewer,
  `ignored_facts` for answers the decoder could not apply and `diagnostics`
  shown only in a collapsed section of `report.md`; `problems` and the generic
  processing-limit note are removed. Questions are short prompts with
  structured fields, and a seen meld without a following discard is an `order`
  question. Discard, missed-discard and Can't-tell answers that match nothing
  are reported as ignored answers. Five-swap and missing-riichi tie-breaks are
  deterministic. The decoder version is 20.
- The replayer rejects calls whose tiles do not form their meld and reports
  violations as structured rows; a rejected hand gets one conflict question
  listing them with seat and time. Reviewer-added missed discards stay virtual
  after dense reads insert tiles, the dora question's kans follow the indicator
  reconciliation, and yaku are stored as name and han ("Dora (4)" in the log).
- Questions have stable ids. **Dismiss**, **Leave as conflict** and **Nothing is
  missing** save a `dismiss` answer that hides the question without a rebuild;
  **It is right** on a call saves the shown meld. `note` answers are no longer
  created; saved ones dismiss a question only when exactly one has the same
  text. A hand is pending only when it has no current decode, its inputs or
  answers changed, or the exports were not built from its decode
  (`export-inputs.json`).
- The studio has one job slot for preparation, analysis, hand updates and
  calibration fits and checks. Recording digests are computed in the background
  (**Checking recording**), and no slow work runs under a lock. The browser polls
  one `/api/job` status (every second while a job runs, every five seconds
  otherwise) and fetches the processing log only when it is open. Jobs run CLI
  commands whose final JSON line is their outcome; the studio shows only that
  message, and child processes write UTF-8. `rebuild` and `calib check` replace
  the private rebuild and border-check scripts. Border checks are kept with the
  geometry they measured, evidence clips are encoded atomically with the
  resolved FFmpeg, hashed bundles and tile art are cacheable, the calibration
  preview never builds a plate during a request (an unprepared recording asks
  for preparation), and label prefill runs without CUDA graphs.
- Launchers install only when the runtime is missing, incomplete or older than
  `pyproject.toml`, or on `Start.cmd --update`, `sh start.sh --update` or
  `VIDEO2TENHOU_UPDATE=1`. Ordinary launches use no network and no longer pick
  up a PyTorch release that would invalidate recognition caches. A failed GPU
  check names the update command.
- `tools/package_release.py models|starter` replaces `package_models.py` and
  `package_starter.py`; classifier provenance is required, and its ZIPs record
  Unix file attributes, so checksums differ from earlier bundles of the same
  files. `tools/check_reader_parity.py` replaces `python -m video2tenhou.benchmark`.
  `tools/profile_conversion.py` hashes nothing and writes `summary.json` and a
  `report.md` with stage shares, speedup ceilings and per-hand solver and dense
  timing, replacing `tools/report_profile.py`. The source-release check verifies
  the sdist inventory and runs a few shipped tests, so CI runs the suite once
  per platform. Recorded-hand tests are marked `slow`.
- Detector datasets write only face labels: regions with reviewed face-down
  tiles stay out of them but keep their classifier crops, and non-face human
  labels are rejected. Draft exports record the archive hash and recognition
  identity instead of duplicate groups. Raw pond structured-slot metrics also
  require the annotated orientation.
- Python API: `run_read`, `run_observe`, `run_decode`, perception evaluation and
  `build_context_dataset` take their options as keyword arguments, and
  `run_decode` logs through the module logger instead of a `log` callback;
  `Detector`, `detector_config` and `Det` lose `backend` and `back`; `source_identity` loses
  `refresh`; `turns.merge` returns typed findings; `write_outputs` lives in
  `video2tenhou.export`.
- Lint policy allows `D401`, `EM101`, `EM102`, `TRY003` and up to eight
  arguments; tests skip docstring, annotation and type-checking-import rules and
  share `tests/builders.py`. Rule-specific `noqa` annotations with reasons cover
  the pinned LibreYOLO calls and lazy imports listed in the maintenance guide.
  JSON fixtures are stored with the LF line endings `.gitattributes` declares.

### Updating

Back up `labels/`. The first launch of an existing runtime runs setup once to
record the installation; setup keeps an installed PyTorch that satisfies the
requirements. Run the launcher with `--update` to choose a newer build, which
makes the next analysis read each recording again. The first analysis after
updating hashes each recording once more to fill `work/source-digests.json`,
keeps existing readings and calm scores, and recomputes observations and decodes
because their bindings changed.
Opening Review on an analyzed project starts an update of its hands, since
earlier exports lack `export-inputs.json` and decodes by older decoder versions
are not shown. Saved `note` answers stay in the journal.

### Performance

A 72-minute PML recording (one hanchan, 10 hands) was converted from empty
caches with the same calibration and saved answers by 0.2.0 and 0.3.0, one after
the other with no other conversion running, on Windows 11 with an Intel Core
i9-13900KF, an NVIDIA RTX 4090 (driver 591.86), PyTorch 2.6.0+cu124, LibreYOLO
1.5.0 and OR-Tools 9.15, using detector weights `ee2e1f99…ebc28` and classifier
weights `32bd602b…321ce`:

| Stage | 0.2.0 | 0.3.0 |
|---|---:|---:|
| Geometry check | 43 s | 28 s |
| Table timing, including calm scoring | 487 s | 221 s |
| Tile reading | 3,825 s | 194 s |
| Voting | 5 s | 5 s |
| Reconstruction | 319 s | 262 s |
| **Complete conversion** | **4,680 s** | **715 s** |

Of the recording's 510 draws, 0.3.0 certified 490 and reported 20 as close
alternatives; 0.2.0 certified 55 and left 455 unchecked. 0.3.0 reread 1,739 s of
video closely, against 82 s. Both versions exported every hand with a legal
replay and the site's scores. They chose different tiles for 22 draws and one
starting hand; neither choice was compared with independent human annotations.

### Validation

- 730 Python tests pass with 86% coverage on the locked PyTorch 2.14.0+cpu;
  without local reference data, the two tests that need it skip. Ruff, ty and
  the frontend format, lint, type and build checks pass, as do 68 frontend unit
  tests and 9 browser tests in Microsoft Edge.
- Detector boxes and confidences, classifier posteriors, crops and structured
  readings are bit-identical to 0.2.0 on 2,193 recorded calls covering all 12
  regions and 10 input shapes on the RTX 4090; recognition identities are
  unchanged.
- The staged decoder reproduces the previous decoder's reconstructions exactly
  when given the same recorded solver answers (28 hand artifacts). Grouped
  certification agrees with a reference that searched each decision separately
  until it was decided, on the two adjudicated reference hands.
- An extracted starter passed a first setup, an offline relaunch and an
  `--update` launch on the RTX 4090 workstation (PyTorch 2.8.0+cu129 selected)
  and with `VIDEO2TENHOU_DEVICE=cpu` (PyTorch 2.14.1+cpu, with detector and
  classifier inference on recorded frames); every launch served the studio.
- The starter's installed runtime converted the 72-minute recording: every hand
  was exported with a legal replay and the site's scores, with the same seven
  questions and certification results as the source run and 2 of 510 draws
  chosen differently.
- Not repeated for this release: launcher qualification on a separate CPU-only
  machine, a fresh machine without uv's download cache and Linux or macOS, and a
  full-recording comparison against independent human annotations.

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
