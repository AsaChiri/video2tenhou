# video2tenhou

Turn a physical-table riichi mahjong broadcast into a Tenhou game log, with
video evidence and human review for anything the cameras cannot settle.
Analysis runs locally. The included layout supports Pacific Mahjong League
(PML) recordings and uses scoremj.com game records to check the results.

## Start

On Windows or Linux, install [uv](https://docs.astral.sh/uv/), FFmpeg (including
`ffprobe`) and Tesseract with English data. Python 3.12 is managed by uv. An NVIDIA CUDA GPU
is recommended for practical analysis times; the locked environment uses
PyTorch's CUDA 12.4 builds on Windows/Linux.

From this checkout:

```console
uv sync --frozen
uv run video2tenhou web
```

The browser studio covers Video, Calibrate, Analyze, Review and Results.
Upload a recording, enter a local video path, or paste a Twitch VOD URL.
Supply the scoremj game IDs in broadcast order. Calibration, progress,
review questions, replay links and file downloads stay in the browser.

**Trained models are required.** Put the project's detector and classifier
bundle in `models/` before analyzing. Keep its detector metadata and classifier
metadata with the matching weights; a generic YOLO checkpoint cannot replace
this trained pair. See [model setup](docs/QUICKSTART.md#models).
The source archive intentionally excludes trained weights and personal video.

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
