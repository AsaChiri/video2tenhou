# Tests

Before running the suite from a source checkout, build the frontend with `npm --prefix frontend ci` and `npm --prefix frontend run build`. The generated assets are ignored by Git and required by the Python web tests. Then run `uv run pytest` from the repository root. Each folder follows the production subsystem it protects:

| Folder | Responsibility |
|---|---|
| `engine/` | Turn reconstruction, pond/meld tracking, legal replay, confidence and targeted acquisition |
| `perception/` | Detector/classifier adapters, crop reading, evidence policy and graph ownership |
| `pipeline/` | Video sampling, calibration, headers, caches, observations and export |
| `web/` | Workspace jobs, review persistence, server APIs and packaged frontend assets |
| `training/` | Human-data provenance, split isolation, training recipes, checkpoint export and evaluation |
| `tooling/` | Model packaging, profiling reports and unreviewed draft-label export |
| `integration/` | Pipeline boundaries and compact recorded regressions with human-grounded expectations |

For a focused check, run a folder or individual test, for example `uv run pytest tests/perception` or `uv run pytest tests/integration/test_recorded_hand_integration.py`. Run coverage with `uv run pytest --cov=video2tenhou --cov-branch`.

Frontend behavior lives in `frontend/tests/` and runs with `npm --prefix frontend test` after `npm --prefix frontend ci`. Vitest mounts Vue components in a DOM environment and tests shared domain functions directly. Python tests do not extract JavaScript from HTML or emulate browser nodes. Both suites are required in CI.

Keep curated evidence in `tests/data/`; its README records fixture provenance and scope. Recorded integration tests use saved numerical evidence instead of downloading videos or loading production checkpoints. The calibration reference-plate check additionally uses a local work artifact when available and skips when absent. Video lifecycle tests need FFmpeg; individual tests report missing optional executables as skips.

Use `tests.paths.DATA` for fixtures and `tests.paths.ROOT` for source/tool files. Do not derive repository paths from a test module's nesting depth. Shared synthetic pond builders live in `tests.engine.helpers`; tests should not import other test modules. Add regressions to the narrowest applicable domain, and reserve integration cases for behavior spanning real module boundaries.
