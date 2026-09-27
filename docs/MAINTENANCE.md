# Maintenance guide

## Contracts and module map

Module docstrings describe the stage and its assumptions; public API docstrings
describe coordinates, seat conventions, effects and failure behavior. Read them
alongside the detailed [design](DESIGN.md), rather than treating cached JSON as
an undocumented source of truth.

| Modules | Responsibility / boundary |
|---|---|
| `paths` | Immutable package assets vs writable `VIDEO2TENHOU_HOME`. |
| `video` | Download, probe, normalized BGR frames; external decoder errors propagate. |
| `layout`, `calibfit` | Frame/region transforms and per-video fitting; failing borders block analysis. |
| `overlay`, `record`, `timeline` | Read header pixels, normalize authoritative results, align hand windows. |
| `calm`, `read`, `observe` | Find still intervals, retain tile posteriors, combine repeated evidence. |
| `perception.detector`, `classifier`, `reader` | Local model inference, geometry and tile-row structure. |
| `engine.ponds`, `melds`, `indicators`, `calls`, `turns` | Persistent visible objects and chronological events. |
| `engine.hand`, `dense` | Hand/seat alignment and targeted closer readings. |
| `engine.rules`, `solver`, `scoring` | Tile conservation, legal hand transitions and result checks. |
| `engine.decode`, `review` | Stage orchestration, human constraints and unresolved evidence. |
| `engine.assemble`, `tenhou6` | Encode exports and replay every hand before accepting it. |
| `cli`, `tool` | Script interface and local browser project workflow. |
| `tool.processes` | Own child process trees so stopping the server also stops downloads, decoding and analysis. |
| `train`, `eval`, `benchmark` | Annotation datasets, training, held-out metrics and throughput comparisons. |

Seats in engine evidence are wind letters for the current hand; site records
and Tenhou arrays use the hanchan's starting-seat order. Conversion belongs in
the explicit seat-mapping helpers. Times are seconds in the source recording.
Tile strings distinguish red fives (`0m/0p/0s`) from ordinary fives; never collapse
them for tile-inventory accounting. A missing observation is not proof of absence.

## Data and caches

Public source includes code, built-in layouts/templates, UI art, small curated
fixtures and guides. `samples/`, `videos/`, `models/`, `weights/`, `labels/`,
`work/`, `out/`, logs and build output are local and ignored. Keep backups of
labels; answers are input data. Do not restore private datasets into Git to make
a test pass.

Stage data lives under `work/<video>/`: header and hand windows, calm intervals,
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
Decoded results also bind the observation and provenance files by content hash.
An interruption after voting cannot make an older reconstructed hand reusable;
only complete replacement files become visible.

Studio manifests under `work/projects/` bind source paths, ordered game IDs,
layout selection and an export signature. Exports are only offered when the
source content, those inputs and the effective calibration match their provenance.
Existing PML outputs can be adopted when their record IDs and every hand's source and
crop signatures agree.
Changing project settings preserves labels but requires analysis; changing
layout also removes the obsolete overlay cache. A `calibration.changed` or
`inputs.changed` marker blocks direct review rebuilds until full analysis
refreshes the upstream evidence. Replacing or removing a recording also requires
preparation again, preserving saved answers. Polling reuses a source digest keyed
by file identity, size and timestamps; conversion stages verify content afresh.
Evidence clips include source and geometry identities in their filenames, and
calibration edits clear remembered border checks.

`tool.app` scopes review APIs by project URL, streams upload bodies and accepts
only local, same-origin writes with its application header. `tool.workflow`
serializes processing jobs and persists interrupted/failed states. Project
manifests are atomically replaced; brief Windows reader contention has a bounded
retry. A failed save retains the prior complete manifest and exposes a retryable,
non-running job rather than claiming durable completion.
`tool.server` handles calibrated evidence and answers; its revision endpoint
lets open review tabs notice externally rebuilt files without discarding
unsaved edits. Model/encoder commands run through `ProcessOwner` so a server
shutdown reaps owned descendants instead of leaving an invisible job running.
Changing calibration must trigger fresh readings for the affected regions.

## Tests

```console
uv run pytest -q --cov --cov-report=term-missing --cov-report=html
uv build
uv run python tools/check_dist.py
uv run python tools/check_source_release.py
```

The suite covers synthetic decoder frames, calibration transforms, real overlay
images, observation/cache behavior, rule/solver cases, replay rejection, review
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
Evaluate every view, especially hand cameras, before replacing release models.

## Release boundaries

This local server has no remote account system or production hosting layer.
Keep it bound to loopback. Browser writes must validate origins, constrain file
paths and serialize GPU work. Do not let a filename become a command string.
Download and analysis jobs must surface failures rather than presenting partial
files as complete results. See [release preparation](RELEASING.md).
