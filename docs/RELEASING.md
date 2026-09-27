# Release checklist

A release consists of tested source and Python distributions, a compatible
model bundle, its provenance and checksums, and installation instructions.
Use the same source revision for all validation and published artifacts.

1. Verify that the source license text and package license metadata describe
   the intended distribution. Check dependency and model redistribution terms
   and include required [third-party notices](../THIRD_PARTY_NOTICES.md).
2. Verify distribution rights for included broadcast-derived calibration
   templates and regression images. Keep private recordings, full annotation
   collections, credentials and generated outputs out of both archives.
3. Run the [validation commands](MAINTENANCE.md#tests), including coverage,
   wheel resource checks and the source archive's isolated test suite. Inspect
   the resulting wheel and source archive for unintended local data.
4. Build the model bundle with `uv run python tools/package_models.py`.
   The ignored ZIP contains per-file SHA-256 metadata and has a separate ZIP
   checksum. Retain the base checkpoint identity, training configuration,
   dataset provenance, classifier class ordering and calibration metadata.
   Detector metadata must match its weights digest. The bundle preserves
   adjacent model provenance and license files; LibreYOLO bundles require
   both `detector/provenance.json` and the upstream `detector/LICENSE` notice.
   Identical model files produce the same ZIP checksum.
5. Install from the artifacts in a fresh environment. Convert a representative
   full recording, inspect review questions, replay every exported hand and
   compare scores and independent human annotations.
6. Record the hardware, model hashes, cache state, workload and quality checks
   for performance claims using the [benchmark protocol](BENCHMARKS.md).
7. Put the actual model download location and checksum instructions in the
   quick start. Tag the tested revision and publish its code, model bundle
   and checksums together.

## Repository history

Ignoring or untracking a file does not remove earlier versions from Git
history. Audit history before making a repository public. If historical
datasets must be excluded, distribute a clean source snapshot or plan a
history migration with collaborators, including replacement of old clones
and references.
