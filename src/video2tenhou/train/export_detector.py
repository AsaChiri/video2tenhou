# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a face-only YOLO9 inference checkpoint without training-resume state.

The model tensors and checkpoint metadata are preserved. Prediction parity must
also be checked with the deployed adapter before promoting the resulting file.
The full source checkpoint remains available for retraining and provenance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from video2tenhou.files import sha256_file as file_hash
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.perception.detector_metadata import (
    InferenceOptions,
    inference_settings,
)
from video2tenhou.perception.evidence_policy import resolve_policy

if TYPE_CHECKING:
    from video2tenhou.perception.evidence_policy import EvidencePolicy


def export_inference_checkpoint(source: Path, target: Path) -> dict:
    """Write a new checkpoint, verify exact model tensors, and report both hashes.

    Only optimizer/raw-training/EMA-resume entries are removed; inference uses
    the existing ``model`` state, which already contains the trainer's selected
    EMA weights. No dtype conversion, fusion or parameter rewriting occurs.
    Existing destinations are rejected and a failed export removes its own file.
    """
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    source_digest = file_hash(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = _model_state(checkpoint)
    resume_keys = {
        "optimizer",
        "train_model",
        "ema",
        "ema_updates",
        "scheduler",
        "scaler",
    }
    exported = {
        key: value for key, value in checkpoint.items() if key not in resume_keys
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            torch.save(exported, stream)
        _verify_export(source, source_digest, target, state)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return {
        "source_sha256": source_digest,
        "target_sha256": file_hash(target),
        "source_bytes": source.stat().st_size,
        "target_bytes": target.stat().st_size,
        "model_tensors": len(state),
        "removed_keys": sorted(set(checkpoint) & resume_keys),
        "exact_tensor_equality": True,
    }


def _model_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Require a face-only YOLO9 detection checkpoint with a nonempty tensor state."""
    if not isinstance(checkpoint, dict) or checkpoint.get("model_family") != "yolo9":
        raise ValueError("Expected a standard LibreYOLO9 checkpoint")
    if (
        checkpoint.get("names") not in ({0: "face"}, {"0": "face"})
        or checkpoint.get("nc") != 1
    ):
        raise ValueError("Expected the single face class")
    if checkpoint.get("task") != "detect":
        raise ValueError("Expected a detection checkpoint")
    state = checkpoint.get("model")
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError("Checkpoint must contain a tensor model state")
    return state


def _verify_export(
    source: Path, source_digest: str, target: Path, state: dict[str, torch.Tensor]
) -> None:
    """Verify tensor identity and ensure the source did not change during export."""
    verified = torch.load(target, map_location="cpu", weights_only=True)
    if set(verified["model"]) != set(state):
        raise RuntimeError("Export changed model state keys")
    for name, value in state.items():
        other = verified["model"][name]
        if (
            other.dtype != value.dtype
            or other.shape != value.shape
            or not torch.equal(other, value)
        ):
            raise RuntimeError(f"Export changed model tensor: {name}")
    if file_hash(source) != source_digest:
        raise RuntimeError("Source checkpoint changed during export")


def write_runtime_metadata(
    weights: Path,
    settings: InferenceOptions,
    *,
    evidence_policy: EvidencePolicy | dict | None = None,
) -> Path:
    """Write new adjacent metadata for a qualified standard YOLO9 face checkpoint.

    Confidence is explicit because its operating point must come from evaluation. This
    records checkpoint identity and settings, not training-data provenance or a license;
    those accompanying files must be supplied separately. Graph execution and evidence
    retention must pass the downstream quality gate. Missing evidence policy records the
    default evidence floors, not a new qualification.
    """
    weights = Path(weights)
    target = weights.with_name("meta.json")
    if target.exists():
        raise FileExistsError(target)
    overrides = asdict(settings)
    confidence = overrides.pop("confidence")
    inference = inference_settings({"confidence": confidence}, **overrides)
    policy = resolve_policy(evidence_policy)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("model_family") != "yolo9"
        or checkpoint.get("size") not in ("t", "s", "m", "c")
        or checkpoint.get("nc") != 1
        or checkpoint.get("task") != "detect"
        or checkpoint.get("names") not in ({0: "face"}, {"0": "face"})
    ):
        raise ValueError(
            "Metadata requires a standard YOLO9 face checkpoint with a known size"
        )
    metadata = {
        "schema_version": 1,
        "backend": "libreyolo",
        "architecture": f"yolo9-{checkpoint['size']}",
        "classes": {"0": "face"},
        "weights_sha256": file_hash(weights),
        "inference": inference,
        "evidence_policy": policy.to_dict(),
    }
    if not inference["cuda_graph"]:
        del inference["cuda_graph"]
    with target.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return target


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Export a new inference file while preserving its full training checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        action="store_true",
        help=(
            "Also create adjacent backend metadata using the evaluated operating point"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Required with --metadata; use the qualified threshold",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Record qualified bounded LibreYOLO CUDA graph execution",
    )
    parser.add_argument(
        "--evidence-policy",
        type=Path,
        help="Qualified retention-policy JSON; requires --metadata",
    )
    args = parser.parse_args(argv)
    if args.evidence_policy and not args.metadata:
        parser.error("--evidence-policy requires --metadata")
    policy = resolve_policy(
        json.loads(args.evidence_policy.read_text(encoding="utf-8"))
        if args.evidence_policy
        else None
    )
    if args.metadata and args.confidence is None:
        parser.error("--metadata requires an explicit --confidence")
    if args.metadata and args.out.with_name("meta.json").exists():
        raise FileExistsError(args.out.with_name("meta.json"))
    report = export_inference_checkpoint(args.source, args.out)
    if args.metadata:
        report["metadata"] = str(
            write_runtime_metadata(
                args.out,
                settings=InferenceOptions(
                    confidence=args.confidence,
                    imgsz=args.imgsz,
                    iou=args.iou,
                    cuda_graph=args.cuda_graph,
                ),
                evidence_policy=policy,
            )
        )
    RESULT.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
