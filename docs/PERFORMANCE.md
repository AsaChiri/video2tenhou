# Performance and benchmarking

Profile a complete conversion before choosing an optimization. Analysis combines
video decoding, tile recognition, evidence aggregation and constrained reconstruction.
The [benchmark workloads](BENCHMARKS.md) define representative inputs and checks.

## Execution and cache contracts

- Sparse and dense reads prepare at most one CPU crop batch ahead while inference
  runs on the calling thread. Batches contain up to 48 upright regions; classifier
  transfers contain up to 512 tile crops, including sideways orientations. Detector
  calls process images individually. Sampling windows, pixels, ordering and batch
  boundaries stay stable; cancellation joins the producer and closes its decoder.
- The supported LibreYOLO9 CUDA-graph path prepares one detector input ahead on
  the CPU. It converts upright BGR arrays directly to the backend's RGB recipe,
  avoiding image copies used for visualization. Resize, padding, normalization,
  tensor layout, single-image inference and postprocessing remain unchanged.
  A shared model lock covers inference and result materialization; preparation
  workers are joined on failure. Other backend versions and eager execution use
  the public prediction path. The selected path is part of recognition identity.
- Count-change checks and votes reuse filtered, structured readings. Classifier
  inference loads the complete local checkpoint without downloading initialization
  weights.
- Recognition identities bind weights, preprocessing, device, numerical-library
  versions, math flags and thread settings. Observation caches additionally bind
  reading contents, calm intervals, voting settings and sparse retention policy.
  Dense caches bind dense retention policy. Policy-only changes can reuse raw
  sparse readings. Completion records verify output digests.
- Dense cache keys include the exact sampling rate and millisecond window bounds.
  Stored timestamps are nominal sampling positions; filtering a longer scan by
  timestamp does not establish equivalent frame selection at shorter endpoints.
- Header caches bind source, overlay geometry, frame dimensions and sampling
  settings. Table-camera edits can reuse unchanged overlay readings. Calm scores
  bind source and region geometry; matching scores can produce new intervals after
  a threshold change. Refreshes publish atomically and retain complete files until
  replacement succeeds.
- Header recognition precomputes normalized templates and caches up to 2,048 exact
  normalized digit signatures. Tesseract results are reused only for identical
  prepared pixels, dimensions, type, configuration and executable. Numeric scans
  stop once a required field is missing or a repeated wind makes a row unusable;
  diagnostic reads examine every corner.
- Scoremj supplies the expected score trajectory. Observed headers still determine
  hand timing, seat mapping and riichi-stick changes.
- Decoder failures propagate as errors. Closing a sampler stops its decoder.

See [maintenance](MAINTENANCE.md#data-and-caches) for invalidation and data ownership.

## Reconstruction and confidence

A main solve has a 60-second ceiling and returns when optimality is proved.
Alternative searches have four-second ceilings, with four searches running
concurrently and eight CP-SAT workers each. A hand can require several solves and
video rereads. Incomplete main searches remain provisional in review.

Draw alternatives clone the built constraint model while preserving variable
indices, constraint order, objective and exclusion placement. Starting-hand
alternatives rebuild because they introduce variables. Reusing a confidence
certificate requires identical model contents, baseline objective and selected
choice.

Alternative searches can stop when the proven lower bound exceeds the review
threshold. Review confidence uses this lower bound. Evidence acquisition also uses
the cost gap of a feasible alternative: a gap at or below 0.5, or a search without
a candidate, requests closer evidence. A distant candidate with a weak proof stays
in review without forcing another reread solely because the proof is weak.
The [design](DESIGN.md#48-engine) describes the reconstruction policy.

## Profile a complete conversion

```powershell
uv run python tools/profile_conversion.py --profile-output work/profile-run videos/broadcast.mp4 --game 123 --work work/fresh-run --out out/fresh-run
uv run python tools/report_profile.py work/profile-run
```

All ordinary `convert` arguments remain available. For a cold conversion, use
empty recording-specific work/output directories; the profiler records their
initial state and never removes caches. Existing calibration and answers remain
inputs. Download and automatic calibration fitting are outside this command.

A forced refresh of a prepared project is a separate workload. Use `--force` to
recompute header, motion, recognition, observation and reconstruction. Alignment
and geometry checks share the authenticated header scan. Retain the preparation
inputs, including the calibration plate used for pond ownership checks. Neither
workload guarantees a cold operating-system file cache.

The profile records input hashes, geometry, library versions, effective thread
counts, stage/hand events and five-second resource samples. It also records
resolved executable paths and versions for FFmpeg, ffprobe, Tesseract and uv;
unavailable or failed version probes are reported without stopping conversion.
`summary.json`
contains conversion wall time, process CPU time and inclusive timing totals.
Hashing and instrumentation imports are reported separately. Existing profile
output is never overwritten; failed calls remain visible.

Nested model calls and concurrent searches overlap. Use the union of alternative
search intervals for active wall time, rather than summing worker durations.
Frame-sampling time measures waiting and cleanup, excluding consumer work. GPU
utilization is device-wide; child CPU sampling misses short-lived processes.
Process CPU seconds divided by wall seconds estimates occupied logical cores.

To estimate the effect of accelerating one complete stage:

```powershell
uv run python tools/report_profile.py work/profile-run --stage read.run_read --factor 2
```

The report gives the hypothetical total speedup and the ceiling if that stage
became free. Other work is assumed unchanged. Solver statuses and searches near
their budgets are reported separately: status alone does not establish a timeout.

## Perception comparisons

```powershell
uv run python -m video2tenhou.benchmark --images work/datasets/detector/images --output work/perception-benchmark.json
```

The benchmark warms scalar and batched reader paths, excludes JPEG loading from
timing, and reports probability and geometry drift. Changed detection counts,
identities, roles or rejection status cause a nonzero exit. For trained model
changes, use [model qualification](DETECTOR_BACKENDS.md#qualification).

## Header comparisons

```powershell
uv run python -m video2tenhou.benchmark --video videos/broadcast.mp4 --window 450 750 --window 1800 2100 --calib pml --output work/header-benchmark.json
```

Windows use seconds; `--window` is repeatable and `--fps` defaults to 1. `--calib`
also accepts a custom layout. This mode requires FFmpeg, packaged templates and
Tesseract for fallback, but no detector or classifier weights.

The scalar reference and production matcher receive the same windows. Template
initialization is outside timing. The report separates pass time, frame wait and
recognition time; serialized fields, missing readings and checksums must match.
Run in a separate process because the benchmark temporarily swaps the matcher.
Repeat with representative score changes and unreadable overlays; the second pass
can benefit from filesystem caching.
