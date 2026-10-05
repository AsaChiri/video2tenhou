# PML quick start

Download **video2tenhou-0.3.0-starter.zip** from the
[latest release](https://github.com/AsaChiri/video2tenhou/releases/latest).
The starter includes the application, PML layout and trained models.

## One-time setup

### Windows

Open PowerShell and install the two prerequisites:

```powershell
winget install --id astral-sh.uv --exact
winget install "FFmpeg (Essentials Build)"
```

These use [uv's installer](https://docs.astral.sh/uv/getting-started/installation/)
and [Gyan's FFmpeg build](https://www.gyan.dev/ffmpeg/builds/) (including `ffprobe`).
Close and reopen PowerShell after installation so it receives the updated PATH.

Extract the starter ZIP, then double-click **Start.cmd** inside the extracted folder.

### Linux

On Ubuntu or Debian, install the prerequisites:

```sh
sudo apt update
sudo apt install -y curl ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, extract the starter ZIP, and run this from its folder:

```sh
bash start.sh
```

On another Linux distribution, install [uv](https://docs.astral.sh/uv/getting-started/installation/),
[FFmpeg](https://ffmpeg.org/download.html)
using that distribution's packages. `uv`, `ffmpeg` and `ffprobe`
must be available on PATH.

### First launch

The launcher installs Python 3.12 and creates a local `.venv-runtime` environment.
It asks uv to select a matched PyTorch/torchvision build for your hardware and
driver, without a fixed CUDA or PyTorch version. You do not need to install a
separate CUDA toolkit. It tests inference kernels before opening the studio;
if the automatically selected GPU cannot run, it uses CPU mode and tells you
that processing will be slower.

The first launch needs internet access and enough disk space for large PyTorch
downloads. Later launches reuse the installed environment and start without
network access; setup runs again only if it did not finish or the application's
requirements changed. Keep uv and your graphics driver up to date. Leave the
launcher running while using the studio.

To repair a failed or outdated installation, update your NVIDIA driver, then
open PowerShell in the extracted folder and run:

```powershell
winget upgrade --id astral-sh.uv --exact
.\Start.cmd --update
```

`--update` selects a current compatible PyTorch build. A new PyTorch build makes
the next analysis of each recording read its video again; saved answers are kept.

To use CPU mode explicitly:

```powershell
$env:VIDEO2TENHOU_DEVICE = 'cpu'
.\Start.cmd
```

Add `--update` to replace an installed GPU build with the CPU build. These
settings apply to that terminal session. Close it and launch normally to return
to automatic selection. On Linux update uv using your installation method and
rerun `sh start.sh --update`, or use `VIDEO2TENHOU_DEVICE=cpu sh start.sh`.
The launch window prints the chosen GPU, device and package versions for support.
After a successful setup, launches do not use the network. `UV_OFFLINE=1` makes
setup or an update use only uv's local cache.

## Convert a recording

The studio opens to **Projects**, where you can search saved recordings by name,
source or game ID. Select **Open project** to resume, or **Settings** to edit game
IDs and layout. **Rename** changes the display name without moving files.
**Delete** asks for confirmation and removes the project from the library while
keeping videos, saved review data and outputs on disk. Projects with a running
analysis or review job cannot be deleted. Use **All projects** to return here.
While the app identifies a new or changed recording file, which can take a
while for a long broadcast, the project shows **Checking recording**.

1. Select **New project**, then choose **Video file** to pick a recording from your computer, or **Video URL**
   to paste its video link. Local files are copied into `samples/` for processing.
   Enter scoremj game IDs in broadcast order; a URL ending in
   `/games/21938` has ID `21938`. Use the complete recording where possible
   so its hands agree with the site record, then select **Prepare recording**.
   To use part of a recording, optionally enter **Start time** and **End time**
   in `HH:MM:SS`, `MM:SS` or seconds (for example, `01:30:00` or `90.5`).
   A blank start means the beginning; a blank end means the end of the recording.
   Include complete games and enter the game IDs for that range. Local originals
   are kept intact; the app creates a separate clip. Review timestamps start at
   zero in that clip. Add a recording again to use a different range.
2. Let the tool measure the table and player panels. Inspect
   the preview. Each pond must contain only its owner's discards; borders
   must leave entire tiles visible. Drag incorrect regions, then select
   **Check borders**, which saves your edits before validating them. Use
   **Discard changes** to return to the saved geometry. Overhead position, rotation and crop
   previews are under **Overhead adjustment and crop previews**. The measured
   fit is specific to this video.
   If preparation fails, open **Settings → Calibration** to correct table,
   hand, pond or meld positioning even before a fit exists. Select **Save
   changes**, then return to Settings and click **Prepare recording** again.
   Saving does not start processing. Calibration checks visible table tiles
   and does not require scores, names or wind labels from the broadcast.
3. Select **Analyze recording**. Progress and errors appear in the project; an
   error says what to do, and **Processing log** shows the full output when you
   open it. Geometry and video/site round alignment are checked before reconstruction.
   Keep the server running while a job runs.
4. **Review** opens after processing with one question beside its video.
   Questions run from most uncertain to least uncertain across all hands.
   Answer with the tile, hand or meld shown, or use **Can't tell** where offered.
   Answers save immediately and update affected hands in the background. Continue
   with other hands while the update runs; further questions from an answered hand
   wait for its new result so inferred answers do not need separate input.
   Answers collected during an update are applied together in the next batch.
   A question that needs no tile answer can be closed with **Dismiss**, **Leave
   as conflict** or **Nothing is missing**; this does not update the hand. For a
   call, **It is right** saves the shown meld as your answer. Unfinished
   confidence checks remain in the confidence data and the report's diagnostics;
   they are not questions and do not block review completion. **Skip for now**
   leaves a question unresolved; you can return to skipped
   questions later. Additional camera views are under **Additional evidence**.
   **Advanced review** is optional: it lets you inspect every action in every
   hand, edit saved answers and apply those changes. Normal review does not
   require navigating hands or manually rebuilding them.
5. Open **Results** and select a hanchan. Use **Open replay** or **Download
   JSON** for the full game, **Copy link** for one hand, or **Copy all hand
   links** for the hanchan. Expand **Reports** for the report and review data.

Use **All projects** to switch recordings. Processed recordings have
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

**Uncertain tiles** groups draws and starting hands with close competing answers.
Ambiguous discards also appear as individual review questions.
A close alternative also fits the available evidence. Review the video and save
the observed tiles to constrain the next reconstruction.
**Can't tell** applies only to that particular draw or starting hand. A hand stays
in review while any of these choices remain unresolved.

When an automatic check reaches its time limit, the report lists it under
**Diagnostics**. This is not a tile question or a confidence proof. A legal
reconstruction can still be exported; a hand with no legal reconstruction
remains excluded. The hand page shows short notes worth checking (for example a
tile that left a pond without a call), and an answer the reconstruction could
not use shows the reason on its row so you can delete or correct it.

## Resume and storage

The studio retains projects in the data directory. Reopen it to return to the
recording and answers. Analysis reuses valid stage caches. Restart an interrupted
job from the project. Stopping the server stops its processing jobs and their
video/download subprocesses; closing a browser tab leaves the server running.
Keep `labels/` if you clear caches: it holds human answers
and calibration. `work/` can be rebuilt; `out/` holds exports.

The starter stores data in its extracted folder. To put data elsewhere,
copy its `models/` directory there and set `VIDEO2TENHOU_HOME` before starting
from the application folder:

```powershell
$env:VIDEO2TENHOU_HOME = 'D:/Mahjong'
.\Start.cmd
```

```sh
export VIDEO2TENHOU_HOME="$HOME/Mahjong"
sh start.sh
```

The server is for your own machine, not a multi-user hosted service. Videos
stay local; video downloads, scoremj records and external Tenhou replays
require network access.

## Troubleshooting

| Symptom | Action |
|---|---|
| Model missing or metadata mismatch | Extract the complete matching bundle into the active data directory; do not mix files from different releases. |
| FFmpeg missing | Fix PATH, restart the terminal and relaunch. |
| Geometry check fails | Open Calibration, inspect the named region and save corrected borders. |
| Video/site mismatch | Verify IDs/order, recording completeness and pond calibration. |
| GPU incompatible / no kernel image / CUDA unavailable | Update the NVIDIA driver and uv, then run `.\Start.cmd --update` (or `sh start.sh --update`). The launcher chooses and checks a compatible build; CPU mode is available as a slower fallback. |
| Setup download interrupted or disk full | Free disk space or restore internet access and relaunch. Incomplete setup is retried automatically. |
| xFormers not available | This optional acceleration message alone is not a setup failure. Read the final error in the log. |
| Correction not in Results | Rebuild after saving review answers, then refresh Results. |
| Project could not be saved | Check that the active data directory is writable and available, then retry the action. The previous project file is retained. |

For scripts, the commands are `download`, `trim`, `calib fit`, `calib check`,
`convert --game ID` and `rebuild`; see `--help` on each command. Results and
errors that need action are reported as a final JSON line on stdout.

## Source setup

For a Git checkout, install Node 22 and build the frontend before launching:

```console
npm --prefix frontend ci
npm --prefix frontend run build
```

Generated frontend assets are excluded from Git. Published source archives
already include them and do not require Node to run.

The source archive and Git checkout exclude trained weights. Install the
prerequisites above and the matching models below, then use **Start.cmd** or
`sh start.sh`, just as with the starter. For development and tests,
`uv sync --frozen` creates a separate `.venv` using the development lockfile.
It does not choose a GPU build for your machine; use the starter launcher for
automatic runtime selection. For advanced runtime choices, see
[dependency environments](MAINTENANCE.md#dependency-environments).

## Models

The starter already contains the required models. Source users can download
**video2tenhou-pml-models.zip** and **video2tenhou-pml-models.sha256** from the
[same release as their source](https://github.com/AsaChiri/video2tenhou/releases/tag/v0.3.0),
or [train their own models](MAINTENANCE.md#training).

With both downloaded files in the application folder, verify and extract them:

```powershell
$modelHash = ((Get-Content ./video2tenhou-pml-models.sha256) -split '\s+')[0]
if ((Get-FileHash ./video2tenhou-pml-models.zip -Algorithm SHA256).Hash -ieq $modelHash) {
    Expand-Archive ./video2tenhou-pml-models.zip -DestinationPath .
} else { throw 'The model ZIP checksum does not match.' }
```

On Linux:

```sh
sha256sum -c video2tenhou-pml-models.sha256 && unzip video2tenhou-pml-models.zip
```

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
and upstream license supplied in the bundle. A generic YOLO checkpoint cannot
replace this trained pair. Inference uses local weights without downloading an
ImageNet checkpoint.

To distribute a trained set, `uv run python tools/package_release.py models`
creates a matching bundle with checksums; see the [release checklist](RELEASING.md).
