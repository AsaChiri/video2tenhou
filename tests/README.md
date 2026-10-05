# Tests

Before running the suite from a source checkout, build the frontend with `npm --prefix frontend ci` and `npm --prefix frontend run build`. The generated assets are ignored by Git and required by the Python web tests. Then run `uv run pytest` from the repository root. Each folder follows the production subsystem it protects:

| Folder | Responsibility |
|---|---|
| `engine/` | Turn reconstruction, pond/meld tracking, legal replay, confidence and targeted acquisition |
| `perception/` | Detector/classifier adapters, crop reading, evidence policy and graph ownership |
| `pipeline/` | Video sampling, calibration, headers, caches, observations and export |
| `web/` | Workspace jobs, review persistence, server APIs and packaged frontend assets |
| `training/` | Human-data provenance, split isolation, training recipes, checkpoint export and evaluation |
| `tooling/` | Launchers, release packaging and source-archive checks, profiling, reader parity and unreviewed draft-label export |
| `integration/` | Pipeline boundaries and compact recorded regressions with human-grounded expectations |

For a focused check, run a folder or individual test, for example `uv run pytest tests/perception` or `uv run pytest tests/integration/test_recorded_hand_integration.py`. Run coverage with `uv run pytest --cov=video2tenhou --cov-branch`. `uv run pytest -m "not slow"` skips the recorded-hand reconstructions and their CP-SAT searches for a quick loop; CI runs everything.

Frontend behavior lives in `frontend/tests/` and runs with `npm --prefix frontend test` after `npm --prefix frontend ci`. Vitest mounts Vue components in a DOM environment and tests shared domain functions directly. Python tests do not extract JavaScript from HTML or emulate browser nodes. Both suites are required in CI.

Keep curated evidence in `tests/data/`; its README records fixture provenance and scope. Recorded integration tests use saved numerical evidence instead of downloading videos or loading production checkpoints. A synthetic plate drawn with the layout checks the overhead fit everywhere; the reference-plate check reads `work/full_1080p/plate.png` from the data directory (`VIDEO2TENHOU_HOME`) and skips with that reason when absent. Video lifecycle tests need FFmpeg; individual tests report missing optional executables as skips.

Use `tests.paths.DATA` for fixtures and `tests.paths.ROOT` for source/tool files. Do not derive repository paths from a test module's nesting depth. Shared synthetic posteriors, voted slots, observations, solver-order tile counts and published reading sets live in `tests.builders`; `tests.engine.factories` builds small decoder-stage inputs, `tests.recognition` stands in for the models and `tests.web.analysis` writes studio analysis outputs. Tests should not import other test modules. Add regressions to the narrowest applicable domain, and reserve integration cases for behavior spanning real module boundaries.
