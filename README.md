# video2tenhou

Turn a physical-table riichi mahjong broadcast into a Tenhou game log, with
video evidence and human review for anything the cameras cannot settle.
Analysis runs locally. The included layout supports Pacific Mahjong League
(PML) recordings and uses scoremj.com game records to check the results.

## Start

1. Download **video2tenhou-0.2.0-starter.zip** from the
   [latest release](https://github.com/AsaChiri/video2tenhou/releases/latest).
   It includes the application, PML layout and trained models.
2. Follow the [one-time setup](docs/QUICKSTART.md#one-time-setup) to install
   uv and FFmpeg on Windows or Linux.
3. Extract the ZIP into a folder you can write to. On Windows, double-click
   **Start.cmd**. On Linux, open a terminal in that folder and run **`bash start.sh`**.

The launcher installs Python and selects a compatible PyTorch/torchvision build
for your hardware and driver, checks that inference works, then opens the browser
studio. Later launches reuse that runtime without network access; run the
launcher with `--update` to select a newer build. CPU mode is available when GPU
acceleration cannot run.
No Git checkout or separate Python installation is
needed. Leave the launcher running while processing recordings. An NVIDIA
CUDA GPU is recommended for practical analysis times.

The browser studio covers Video, Calibrate, Analyze, Review and Results.
Choose a **Video file** from your computer or paste a **Video URL** supported by yt-dlp.
Supply the scoremj game IDs in broadcast order. Calibration, progress,
review questions, replay links and file downloads stay in the browser.

Using a source checkout instead? Follow [source setup and models](docs/QUICKSTART.md#source-setup).

## Guides

- [PML quick start](docs/QUICKSTART.md): setup, one-video workflow and recovery.
- [Other layouts](docs/LAYOUTS.md): coordinates, fitting, validation and limits.
- [Maintenance](docs/MAINTENANCE.md): module contracts, caches, tests and training.
- [Performance](docs/PERFORMANCE.md): execution, cache behavior and benchmarking.
- [Model training](docs/DETECTOR_BACKENDS.md): detector, classifier and deployment recipes.
- [Design](docs/DESIGN.md): reconstruction rules and evidence model.
- [Third-party notices](THIRD_PARTY_NOTICES.md): tile art and model dependencies.
- [Release checklist](docs/RELEASING.md): artifact, model and distribution checks.
- [Changelog](CHANGELOG.md): changes and upgrade notes for each release.

## Development

```console
uv sync --frozen
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build
uv run pytest -q --cov --cov-report=term-missing
uv build
```

Tests use small committed fixtures and synthetic legal hands without downloaded
VODs or model weights. See the [test guide](tests/README.md) for focused runs.
The UI uses Vue and Vite; see [frontend development](frontend/README.md) for
components, local development and asset packaging. Node is needed to build
the UI from source; releases include the built assets. Generated frontend files
are ignored by Git. `video2tenhou web` is the single browser command.

The CLI remains available for scripts (`uv run video2tenhou --help`). Development
uses the lockfile; the starter uses its own `.venv-runtime` environment and
selects PyTorch within the supported package ranges at setup or on `--update`.
Generated recordings, labels, work caches and outputs stay local and ignored.
Set `VIDEO2TENHOU_HOME` before launch to use another data directory.

## License

Licensed under the [Apache License 2.0](LICENSE).
