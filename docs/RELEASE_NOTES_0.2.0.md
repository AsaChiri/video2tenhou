# video2tenhou 0.2.0

Download **video2tenhou-0.2.0-starter.zip**, extract it into a writable folder,
and run **Start.cmd** on Windows or `sh start.sh` on Linux/macOS. Install uv
and FFmpeg first; the launcher installs Python and selects a compatible
PyTorch runtime. The starter includes the trained models and compiled browser
application, so Node is not needed. See the [quick start](QUICKSTART.md).

## Changes

- One browser studio now owns the project library, recording import,
  calibration, analysis, guided review and exports. Projects can be renamed
  and removed without deleting recordings or human answers.
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

## Updating

Back up `labels/` before updating; it contains human answers and calibration.
Keep the data directory and model bundle together, or set `VIDEO2TENHOU_HOME`
to the existing data directory. Open the project and run **Analyze recording**
to refresh derived evidence under the current contracts. Check calibration
and game order if table timing reports a mismatch.

`video2tenhou web` is the browser entry point. The separate `review` command
and old review-page URLs are removed. Source-checkout users must build the
frontend before serving or packaging; see [frontend development](../frontend/README.md).

## Preparation status

This release is prepared locally and is not yet approved for publication.
The mandatory Python Ruff and ty gates expose a repository-wide diagnostic
backlog. They remain enabled and must pass before tagging or publishing 0.2.0.
The release checklist's fresh-machine CPU/GPU launcher checks and a representative
full-recording comparison also remain required. See [release preparation](RELEASING.md).
