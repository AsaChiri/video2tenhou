# Detector and classifier training

The LibreYOLO YOLO9 detector locates face-up tiles. A separate ResNet18 classifier
identifies their faces. Train and evaluate them together: detector boundaries
determine the pixels the classifier receives.

## Runtime contract

The adapter uses LibreYOLO 1.5.0 and standard YOLO9 checkpoints with exactly
`0: face`. It receives upright BGR region crops and returns region-coordinate
boxes. Preprocessing uses a 1024-pixel long side and top-left rectangular padding
rounded to the 32-pixel stride; default confidence is 0.15 and IoU is 0.5.

Adjacent `detector/meta.json` binds configuration to the checkpoint. Schema 1
requires `backend`, `architecture` (for example `yolo9-s`), a string-keyed
`classes` map and `weights_sha256`. Its optional `inference` object contains
`imgsz`, `confidence`, `iou` and boolean `cuda_graph`. Explicit constructor
settings override metadata defaults. Invalid settings, incompatible classes
and mismatched hashes fail during loading.

Inference takes the direct path in `perception/yolo9_direct.py`: it letterboxes
uint8 pixels exactly as LibreYOLO's public `predict()` does, uploads bytes and
maps them through the same float32 division, then calls two private LibreYOLO 1.5
members, the model's graph dispatch and its postprocessing. Inputs and
detections are bit-identical to `predict()`; `tests/perception/test_yolo9_direct.py`
checks the private signatures and compares both paths. CUDA graphs are enabled
on CUDA through metadata or `cuda_graph=True`. YOLO9 replaces its anchor/stride
grids when the padded input shape changes, so each padded shape gets its own
model instance that captures once and then only replays; the first shape uses
the validated model and later instances load from the verified checkpoint,
failing if its file changed. CPU uses eager inference, and so does the studio's
label prefill.

Recognition identities include weights, preprocessing, class mapping, inference
settings, device and numerical runtime. Set runtime flags before constructing
models. See [cache contracts](PERFORMANCE.md#execution-and-cache-contracts).

### Evidence retention and cache reuse

Optional `evidence_policy` metadata uses schema 1 with complete `sparse` and
`dense` maps containing `hand`, `pond` and `meld` confidence floors. Defaults
are `.2/.2/.35` for sparse and `.2/.2/.2` for dense, in that order. Sparse
evidence also requires `p(none) < .5`. Filtering uses inclusive comparisons on
raw detection scores, then recomputes region structure.

Choose floors using the filtered, restructured readings of the exact model
pair. Check rejected rows, ignored roles, pond slots and extra detections.
Keep the selected policy with the deployed checkpoint.

Raw sparse readings are independent of retention policy. Changing a sparse
floor reuses those readings, rebuilds observations and invalidates decoding.
Dense caches store filtered evidence and bind the dense policy. Stage-specific
fingerprints let each stage reuse evidence when only the other policy changes.

## Draft boxes from an archive

Generate face-box drafts with the deployed detector:

```powershell
uv run python tools/pseudolabel_images.py --archive path/to/images.zip --weights models/detector/weights.pt --out work/drafts/faces --device cuda:0
```

The destination must be new. The exporter preserves image bytes under generated
names and writes predictions, normalized YOLO files under `draft_labels/`, a
manifest with each image hash and a summary with the archive hash, the
detector's recognition identity and confidence threshold. Archive paths never
become output paths. A failed run keeps its partial output marked incomplete.

Drafts require review for missed objects, extra boxes and boundaries. An empty
draft is not a reviewed negative. The exporter keeps drafts separate from accepted
annotations and does not create a training split. Use only completed exports.

## Face detector training

Create an isolated environment for training:

```powershell
$trainingPython = "work/train-env/Scripts/python.exe"
$savedEnvironment = $env:UV_PROJECT_ENVIRONMENT
try {
    $env:UV_PROJECT_ENVIRONMENT = "work/train-env"
    uv sync --frozen
} finally {
    $env:UV_PROJECT_ENVIRONMENT = $savedEnvironment
}
```

Build upright crops from human annotations with the
[dataset commands](MAINTENANCE.md#training), then assemble the face dataset and train:

```powershell
uv run --no-project --python $trainingPython python -m video2tenhou.train.face_data --human work/datasets/detector --drafts work/drafts/faces --out work/datasets/faces --video samples/recording.mp4 --work work
uv run --no-project --python $trainingPython python -m video2tenhou.train.train_libreyolo --data work/datasets/faces/data.yaml --base path/to/LibreYOLO9s.pt --out work/runs/detector --epochs 60 --batch 8 --imgsz 1024 --device cuda:0
```

Dataset and run destinations must be new. The builder preserves whole-hand
validation groups and reviewed negatives, rejects exact duplicates across
splits, and places archive drafts only in training. It requires a completed
draft export, excludes drafts with no detections and rejects human labels other
than class 0 (face). Manifests distinguish human and machine labels
and retain group identities and image/label hashes. Split unknown-session
archives conservatively; exact hashes do not identify near duplicates.

Training disables flips, permits small rotations and reserves label capacity
for dense mosaics. Windows defaults to `--workers 0` to avoid spawned workers
duplicating framework memory; other platforms default to four workers.

For camera-domain refinement, build a human-only dataset by omitting `--drafts`
and train from a one-class checkpoint with `--refine-human`. The preset uses
12 epochs, a lower learning rate and no mosaic, with checkpoints every two epochs.
It retains the human holdout and rejects pseudo-labeled rows.

## Deployment

Export the inference tensors and metadata:

```powershell
uv run --no-project --python $trainingPython python -m video2tenhou.train.export_detector --source path/to/best.pt --out path/to/deployment/detector/weights.pt --metadata --confidence .15 --evidence-policy path/to/evidence-policy.json
```

The exporter verifies exact tensor equality and writes checkpoint hashes.
`--evidence-policy` requires `--metadata`; omitting it records the default floors.
Use the confidence and policy qualified for the model pair. Add `--cuda-graph`
after validating boxes, crop pixels and posteriors against eager inference.
Keep `provenance.json` and the base checkpoint's `LICENSE` with the exported files;
[third-party notices](../THIRD_PARTY_NOTICES.md) describes bundle provenance.

## Classifier crop robustness

The context refinement helper varies crop boundaries around human tile quads
while retaining supervised anchor crops. It requires lossless cached frames,
the annotation journal and hand table, and a human-only face-data manifest.
Build that manifest without archive drafts, or reuse an existing human-only
dataset. The mixed detector-training manifest is not accepted here.

```powershell
uv run --no-project --python $trainingPython python -m video2tenhou.train.face_data --human work/datasets/detector --out work/datasets/faces-human --video samples/recording.mp4 --work work
uv run --no-project --python $trainingPython python -m video2tenhou.train.classifier_context build --anchors work/datasets/classifier/train --human-manifest work/datasets/faces-human/manifest.jsonl --video samples/recording.mp4 --work work --calib pml --out work/datasets/classifier-context
uv run --no-project --python $trainingPython python -m video2tenhou.train.classifier_context train --data work/datasets/classifier-context --validation work/datasets/classifier/val --base models/classifier --out work/runs/classifier-context
```

Both destinations must be new. Annotation timestamps determine held-out hands.
Background anchors remain unchanged; new contexts require known human identities
and receive 0–20% margins and ±6% center shifts. The three-epoch recipe uses AdamW
at `1e-5`, frozen batch-normalization statistics and distillation from the supplied
reference classifier on anchor crops only. Distillation uses temperature 2 and
weight 4; inference retains the supplied calibration temperature. Checkpoints
record model/data hashes, settings and individual validation gains and losses.

## Qualification

Evaluate localization and complete region readings on the human holdout:

```powershell
uv run --no-project --python $trainingPython python -m video2tenhou.eval detector --weights path/to/deployment/detector/weights.pt --dataset work/datasets/detector --predictions work/eval/boxes.jsonl
uv run --no-project --python $trainingPython python -m video2tenhou.eval perception samples/recording.mp4 --weights path/to/deployment/detector/weights.pt --classifier path/to/deployment/classifier --predictions work/eval/readings.jsonl
```

Inference defaults come from checkpoint metadata; `--confidence` can override
the detection threshold for evaluation. Cover every camera view, occlusion,
red fives, sideways tiles, empty regions and partial faces.

`usable_correct_of_gt` counts accepted identities, roles, orientations and pond
slots, including missed targets. `retained.sparse` and `retained.dense` apply
their retention policies and recompute structure. Per-reading
`correct_gt_indices` identify individual losses; `raw_reading` preserves boxes
before the human standing-hand scope filter.

Use independent complete-hand evidence to verify draw order, discards, calls,
replay, scores and review behavior. Keep checkpoint-selection data distinct from
final evaluation data. Measure throughput using the
[benchmark workloads](BENCHMARKS.md).
