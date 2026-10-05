# Performance and benchmarking

Profile a complete conversion before choosing an optimization. Analysis combines
video decoding, tile recognition, evidence aggregation and constrained reconstruction.
The [benchmark workloads](BENCHMARKS.md) define representative inputs and checks.

## Execution and cache contracts

- Each sampled window is one FFmpeg decode. A reader thread keeps up to 16
  decoded frames queued, so FFmpeg does not wait while each frame is processed.
  Closing a sampler stops FFmpeg and joins the reader; decoder failures raise.
- Sparse and dense reads prepare at most one CPU crop batch ahead while inference
  runs on the calling thread. Batches contain up to 48 upright regions; classifier
  transfers contain up to 512 tile crops, including sideways orientations. Detector
  calls process images individually, while a worker thread letterboxes the
  batch's crops. Sampling windows, pixels, ordering and batch boundaries stay
  stable; cancellation joins the producer and closes its decoder.
- Detector inputs take the direct YOLO9 path in `perception/yolo9_direct.py`,
  not LibreYOLO's public prediction API: uint8 pixels are letterboxed as
  `predict` does, uploaded as bytes and mapped through the same float32 division,
  then passed to the model's own graph dispatch and postprocessing. Inputs and
  detections are bit-identical to `predict`. One lock per detector covers model
  loading, letterboxing, inference and result materialization of each complete
  batch. With CUDA graphs, each padded input shape owns a model that captures
  once and then only replays, so YOLO9's shape-specific decode grids stay valid
  and no capture is released or recaptured. Each shape model holds about
  200–240 MiB of GPU memory; ten padded shapes reserve about 2.4 GiB in PyTorch,
  with as much again in Windows commit. The graph policy is part of recognition
  identity. Studio label previews run eagerly and keep no shape models.
- The classifier stacks resized uint8 crops, uploads bytes and normalizes them
  on the device through a per-channel lookup table built by its own `to_tensor`,
  so model inputs and posteriors are bit-identical to CPU normalization.
  Inference loads the complete local checkpoint without downloading
  initialization weights.
- Calm scoring runs the small per-region resize, blur, colour conversion and
  skin mask with one OpenCV thread and restores the process setting afterwards,
  so identities that record thread counts do not change; the full-frame
  de-rotation keeps OpenCV's pool.
- `convert` constructs one detector and one classifier for the geometry check,
  table timing and reading. Decoding resolves model identities from metadata
  and checkpoint digests without loading weights; a model pair loads only when a
  hand rereads the video. Importing the engine loads neither PyTorch, LibreYOLO,
  HTTPX nor yt-dlp.
- A recording is hashed once: `work/source-digests.json` keeps its SHA-256 until
  its size, timestamps, file index or device change. Completion manifests record
  digests of the text they publish, and downstream stages bind to those manifests
  instead of hashing files again.
- Count-change checks and votes reuse filtered, structured readings.
- Recognition identities bind weights, preprocessing, device, numerical-library
  versions, math flags and thread settings. Observation caches additionally bind
  the reading manifest, calm intervals, voting settings and sparse retention
  policy. Dense caches bind dense retention policy. Policy-only changes can reuse
  raw sparse readings.
- Dense cache keys include the exact sampling rate and millisecond window bounds.
  Stored timestamps are nominal sampling positions; filtering a longer scan by
  timestamp does not establish equivalent frame selection at shorter endpoints.
- Table timing caches bind source, region geometry, calm intervals and detector
  identity. Calm scores bind source and region geometry; matching scores can
  produce new intervals after a threshold change. Refreshes publish atomically
  and retain complete files until replacement succeeds.
- Scoremj supplies scores, rounds, players, starting seats and riichi sticks.
  Pond clearings determine hand timing; East 1's dealer occupies the TL chair.

Reading several hands in parallel was measured and not adopted: two workers
sped up the read stage 1.21×, and a third ran out of host memory, because each
worker needs its own model pair and shape models (about 2.5 GiB of GPU memory
and 6.5 GiB of Windows commit).

See [maintenance](MAINTENANCE.md#data-and-caches) for invalidation and data ownership.

## Reconstruction and confidence

A main solve has a 60-second ceiling and returns when optimality is proved.
Certification is grouped: each CP-SAT search asks for the cheapest
reconstruction that changes any pending decision and stops at the first proof
above the review gap or witness within it. A hand has at most two certification
passes, the first fit only when the video can be reread and the final solution
once, each with its own 60-second budget and ending as soon as every decision
is classified. Main solves and certification use eight CP-SAT workers. Only
ambiguous draws, and open-kan replacements without direct evidence, trigger
dense hand rereads. A hand can require several solves and video rereads.
Decisions left at the deadline remain unresolvable in confidence data and in
the report's diagnostics without creating retry questions.
The [design](DESIGN.md#48-engine) describes the reconstruction policy.

## Profile a complete conversion

```powershell
uv run python tools/profile_conversion.py --profile-output work/profile-run videos/broadcast.mp4 --game 123 --work work/fresh-run --out out/fresh-run
```

All ordinary `convert` arguments remain available. For a cold conversion, use
empty recording-specific work/output directories; the profiler records their
initial state and never removes caches. Existing calibration and answers remain
inputs. Download and automatic calibration fitting are outside this command.

A forced refresh of a prepared project is a separate workload. Use `--force` to
recompute header, motion, recognition, observation and reconstruction. Retain
the preparation inputs, including the calibration plate used for pond ownership
checks. Neither workload guarantees a cold operating-system file cache.

The profile output directory must be new. `summary.json` records the source
revision, package versions, the recording's size, modification time and stream
properties, a geometry digest, the detector and classifier recognition
identities, thread counts, how many files the recording's work and output
directories held, and every timing; no file is hashed for the profile.
`report.md` gives each stage's wall and process-CPU time, its share and the
speedup ceiling if it took no time; per hand, the decode's wall and CPU time,
dense reads, main solves, certification, the number of CP-SAT searches with
their statuses and how many came within 10% of their time limit; and nested
model, frame and search operations. A conversion that fails after it starts
still writes both files.

Spans are inclusive: nested calls and concurrent work overlap and must not be
added. Process CPU covers all threads of the conversion process, not FFmpeg
children; CPU divided by wall estimates occupied logical cores. GPU work is not
synchronized for timing. The ceiling column, 1/(1 − share), bounds the total
speedup from accelerating one stage with other work unchanged. Status alone does
not establish a timeout.

## Perception comparisons

```powershell
uv run python tools/check_reader_parity.py --images work/datasets/detector/images
```

The check reads every region image alone and in production batches. It exits
nonzero when counts, identities, roles, slots, orientation or rejection differ,
reports the largest posterior and box differences, and measures no speed. For
trained model changes, use [model qualification](DETECTOR_BACKENDS.md#qualification).
