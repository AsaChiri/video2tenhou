# PML quick start

## Install on Windows or Linux

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install [FFmpeg](https://ffmpeg.org/download.html), with `ffmpeg` and `ffprobe`
   on PATH, and [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
   with English language data. Restart the terminal after changing PATH.
3. Open a terminal in the downloaded repository and run `uv sync --frozen`.
   This installs Python 3.12 and the locked dependencies. The initial download
   includes PyTorch. Keep an NVIDIA driver compatible with its CUDA runtime.
4. Install the trained models described below.
5. Run `uv run video2tenhou web`. Leave the terminal running while you use the studio.

## Models

The application needs these files in its data directory:

```text
models/
  detector/weights.pt
  detector/meta.json
  detector/provenance.json
  detector/LICENSE
  classifier/weights.pt
  classifier/meta.json
  classifier/provenance.json
```

The detector metadata selects its matching backend and detection settings.
The classifier metadata records the exact class ordering and temperature.
Keep each metadata file with its matching weights, and retain the provenance
and upstream license supplied in the bundle. Inference uses local weights
without downloading an ImageNet checkpoint.

Model weights are distributed separately from the source. Use a compatible
project bundle and verify its checksums before extracting it into the data
directory, or [train your own models](MAINTENANCE.md#training).
Use the model ZIP and its `.sha256` file from the same release as the source.
For example, with both files in the data directory:

```powershell
$modelHash = ((Get-Content ./video2tenhou-pml-models.sha256) -split '\s+')[0]
if ((Get-FileHash ./video2tenhou-pml-models.zip -Algorithm SHA256).Hash -ieq $modelHash) {
    Expand-Archive ./video2tenhou-pml-models.zip -DestinationPath .
} else { throw 'The model ZIP checksum does not match.' }
```

On Linux, use `sha256sum -c video2tenhou-pml-models.sha256 && unzip video2tenhou-pml-models.zip`.

To distribute a trained set, `uv run python tools/package_models.py` creates
a matching bundle with checksums; see the [release checklist](RELEASING.md).

## Convert a recording

1. Choose **Saved video**, **Upload** or **Twitch URL**. Saved video lists
   recordings in `samples/` and `videos/`; **Enter a video path → Video path**
   accepts a local path without copying it. Upload copies a recording into
   `samples/`. Enter scoremj game IDs in broadcast order; a URL ending in
   `/games/21938` has ID `21938`. Use the complete recording where possible
   so its hands agree with the site record, then select **Prepare recording**.
2. Let the tool measure the table and player panels. Inspect
   the preview. Each pond must contain only its owner's discards; borders
   must leave entire tiles visible. Drag incorrect regions, then select
   **Check borders**, which saves your edits before validating them. Use
   **Discard changes** to return to the saved geometry. Overhead position, rotation and crop
   previews are under **Overhead adjustment and crop previews**. The measured
   fit is specific to this video.
3. Select **Analyze recording**. Progress and errors appear in the project.
   Geometry and video/site round alignment are checked before reconstruction.
   Keep the server running while a job runs.
4. **Review** opens after processing. Inspect the full video frame alongside
   each question and its answer controls. Questions with the weakest evidence
   appear first. Answer with the tile, hand or meld
   shown, or use **Can't tell** where offered when the evidence is not visible.
   Select **Rebuild changes**
   after saving answers. It rebuilds only hands with unapplied changes, including
   answers saved across several hands. Results for an affected hanchan become
   available again after rebuilding incorporates its saved changes. Crop diagnostics are available in the collapsed advanced
   controls when needed.
5. Open **Results** and select a hanchan. Use **Open replay** or **Download
   JSON** for the full game, **Copy link** for one hand, or **Copy all hand
   links** for the hanchan. Expand **Reports** for the report and review data.

The header's recording selector switches projects. Processed recordings have
**Review**, **Results** and **Settings** tabs, with Review selected by default.
To correct game IDs or layout, use **Settings → Recording settings**. Saved
answers and the recording stay in place; exports are hidden until analysis
succeeds with the new settings. To adjust table geometry, use **Settings →
Processing → Calibration**, then **Analyze again** to refresh video readings
before rebuilding.

In the hand report, `complete` means no open questions. `review` means a legal
reconstruction has questions to resolve. `conflict` means reconstruction or replay validation
failed, and that hand is excluded. Check the report before treating an export
as a complete hanchan.

**Uncertain tiles** groups draws and starting hands the solver could not certify.
Ambiguous discards also appear as individual review questions.
A close alternative may fit, or the search may have stopped before proving the
choice. With no saved changes waiting, **Rebuild hand** retries that hand's search.
**Can't tell** applies only to that particular draw or starting hand. A hand stays
in review while any of these choices remain unresolved.

**Search incomplete** means a legal result is still provisional because the
solver ran out of search time; use **Rebuild hand** to retry, with no tile answer required.

## Resume and storage

The studio retains projects in the data directory. Reopen it to return to the
recording and answers. Analysis reuses valid stage caches. Restart an interrupted
job from the project. Stopping the server stops its processing jobs and their
video/download subprocesses; closing a browser tab leaves the server running.
Keep `labels/` if you clear caches: it holds human answers
and calibration. `work/` can be rebuilt; `out/` holds exports.

By default the launch directory is the data directory. To put data elsewhere,
set `VIDEO2TENHOU_HOME` before starting:

```powershell
$env:VIDEO2TENHOU_HOME = 'D:/Mahjong'
uv run video2tenhou web
```

```sh
export VIDEO2TENHOU_HOME="$HOME/Mahjong"
uv run video2tenhou web
```

The server is for your own machine, not a multi-user hosted service. Videos
stay local; Twitch downloads, scoremj records and external Tenhou replays
require network access.

## Troubleshooting

| Symptom | Action |
|---|---|
| Model missing or metadata mismatch | Extract the complete matching bundle into the active data directory; do not mix files from different releases. |
| FFmpeg/Tesseract missing | Fix PATH, restart the terminal and relaunch. |
| Geometry check fails | Open Calibration, inspect the named region and save corrected borders. |
| Video/site mismatch | Verify IDs/order, recording completeness and overlay recognition. |
| CUDA unavailable | Check the driver and `uv run python -c "import torch; print(torch.cuda.is_available())"`. |
| Correction not in Results | Rebuild after saving review answers, then refresh Results. |
| Project could not be saved | Check that the active data directory is writable and available, then retry the action. The previous project file is retained. |

For scripts, commands remain `download`, `calib fit`, `convert --game ID`,
and `review`; see `--help` on each command.
