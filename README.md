# video2tenhou

Turn a physical-table riichi mahjong broadcast into a Tenhou game log, with
video evidence and human review for anything the cameras cannot settle.
Analysis runs locally. The included layout supports Pacific Mahjong League
(PML) recordings and uses scoremj.com game records to check the results.

## Start

1. Download **video2tenhou-0.1.2-starter.zip** from the
   [latest release](https://github.com/AsaChiri/video2tenhou/releases/latest).
   It includes the application, PML layout and trained models.
2. Follow the [one-time setup](docs/QUICKSTART.md#one-time-setup) to install
   uv, FFmpeg and Tesseract on Windows or Linux.
3. Extract the ZIP into a folder you can write to. On Windows, double-click
   **Start.cmd**. On Linux, open a terminal in that folder and run **`bash start.sh`**.

The launcher installs Python and the locked dependencies on first use, then
opens the browser studio. No Git checkout or separate Python installation is
needed. Leave the launcher running while processing recordings. An NVIDIA
CUDA GPU is recommended for practical analysis times.

The browser studio covers Video, Calibrate, Analyze, Review and Results.
Upload a recording, enter a local video path, or paste a video URL supported by yt-dlp.
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

## Development

```console
uv sync --frozen
uv run pytest -q --cov --cov-report=term-missing
uv build
```

Tests use small committed fixtures and synthetic legal hands without downloaded
VODs or model weights. See the [test guide](tests/README.md) for focused runs.

The CLI remains available for scripts (`uv run video2tenhou --help`).
Generated recordings, labels, work caches and outputs stay local and ignored.
Set `VIDEO2TENHOU_HOME` before launch to use another data directory.

## License

Licensed under the [Apache License 2.0](LICENSE).
