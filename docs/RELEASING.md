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
   Build the starter download from that model bundle and the source distribution:

   ```console
   uv run python tools/package_starter.py --source dist/video2tenhou-0.1.3.tar.gz --models dist/video2tenhou-pml-models.zip --out dist/video2tenhou-0.1.3-starter.zip
   ```

   Use the matching version in the filenames. The starter contains the application,
   launchers, adaptive runtime setup and trained models in one folder.
   Its checksum accompanies the ZIP. Extract it into a new directory and verify
   that the launcher selects a compatible runtime and opens the browser studio.
   Check NVIDIA and CPU-only machines, an existing installation, and a repair
   launch. Record the versions selected on each machine; runtime selection is
   intentionally independent of the development lockfile.
5. Install from the artifacts in a fresh environment. Convert a representative
   full recording, inspect review questions, replay every exported hand and
   compare scores and independent human annotations.
6. Record the hardware, model hashes, cache state, workload and quality checks
   for performance claims using the [benchmark protocol](BENCHMARKS.md).
7. Put the starter download first in the release notes, with prerequisite and
   launcher instructions from the quick start. Tag the tested revision and
   publish the starter, source, wheel, model bundle and checksums together.

## Repository history

Ignoring or untracking a file does not remove earlier versions from Git
history. Audit history before making a repository public. If historical
datasets must be excluded, distribute a clean source snapshot or plan a
history migration with collaborators, including replacement of old clones
and references.
