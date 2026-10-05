# Benchmark workloads

Use a workload that exercises the behavior being changed, then measure a complete
conversion. Commands and timing interpretation are in [Performance](PERFORMANCE.md).

| Workload | What to retain | Acceptance checks |
|---|---|---|
| Table timing | Complete games, table clearings, empty spans and occluded ponds | Hand boundaries, hand count and alignment to site records |
| Region recognition | Every camera, empty regions, occlusion, sideways tiles and red fives | Box geometry, crop pixels, probabilities, roles and retained observations |
| Reconstruction | Complete hand evidence, result record and saved answers | Draw/discard order, calls, tile inventory, confidence bounds and legal replay |
| Complete recording | Video, geometry, model/runtime identities, initial caches and answers | Stage times, exports, scores and independent human annotations |

Keep benchmark outputs under `work/`, together with the profile's source
revision, recognition identities and geometry digest. Reports belong to the
measured run; this guide defines the workloads.

## Representative evidence

Hold out whole hands or recordings. Adjacent frames share nearly identical tiles
and cannot serve as independent training and evaluation examples. Include narrow
margins, ambiguous starting hands, terminal discards, calls that remove pond tiles,
and replacement draws after kan. Committed numeric fixtures in `tests/data/`
protect specific reconstruction failures; video workloads exercise recognition.

Record answers used as reconstruction inputs separately from annotations reserved
for evaluation. Compare individual retained observations and corrected choices as
well as totals: equal accuracy counts can conceal different errors.

## Comparing executions

Use the same inputs, model pair, geometry, sampling windows and initial cache
state. Keep the workload otherwise idle and repeat timing-sensitive comparisons
in alternating order. Preserve reports from both executions.

For changes intended to preserve recognition exactly, compare coordinates, crop
bytes, probabilities and structured readings before downstream results. For model
changes, use the [qualification checks](DETECTOR_BACKENDS.md#qualification).
Time-limited search can return different proof bounds even for equal chosen tiles,
so confidence and review behavior are part of the comparison.

For an end-to-end claim, include the full conversion and identify preparation,
download and human review separately. Nested model calls and overlapping searches
are components of their enclosing stage, not extra elapsed time.
