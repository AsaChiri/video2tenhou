"""Export a face-only YOLO9 inference checkpoint without training-resume state.

The model tensors and checkpoint metadata are preserved. Prediction parity must
also be checked with the deployed adapter before promoting the resulting file.
The full source checkpoint remains available for retraining and provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..files import sha256_file as file_hash
from ..perception.evidence_policy import resolve_policy
from ..perception.detector_metadata import inference_settings


def export_inference_checkpoint(source: Path, target: Path) -> dict:
    """Write a new checkpoint, verify exact model tensors, and report both hashes.

    Only optimizer/raw-training/EMA-resume entries are removed; inference uses
    the existing ``model`` state, which already contains the trainer's selected
    EMA weights. No dtype conversion, fusion or parameter rewriting occurs.
    Existing destinations are rejected and a failed export removes its own file.
    """
    import torch

    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    source_digest = file_hash(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_family") != "yolo9":
        raise ValueError("Expected a standard LibreYOLO9 checkpoint")
    if checkpoint.get("names") not in ({0: "face"}, {"0": "face"}) or checkpoint.get("nc") != 1:
        raise ValueError("Expected the single face class")
    if checkpoint.get("task", "detect") != "detect":
        raise ValueError("Expected a detection checkpoint")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint must contain a tensor model state")
    resume_keys = {"optimizer", "train_model", "ema", "ema_updates", "scheduler", "scaler"}
    exported = {key: value for key, value in checkpoint.items() if key not in resume_keys}
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            torch.save(exported, stream)
        verified = torch.load(target, map_location="cpu", weights_only=True)
        if set(verified["model"]) != set(state):
            raise RuntimeError("Export changed model state keys")
        for name, value in state.items():
            other = verified["model"][name]
            if other.dtype != value.dtype or other.shape != value.shape or not torch.equal(other, value):
                raise RuntimeError(f"Export changed model tensor: {name}")
        if file_hash(source) != source_digest:
            raise RuntimeError("Source checkpoint changed during export")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return dict(source_sha256=source_digest, target_sha256=file_hash(target),
                source_bytes=source.stat().st_size, target_bytes=target.stat().st_size,
                model_tensors=len(state), removed_keys=sorted(set(checkpoint) & resume_keys),
                exact_tensor_equality=True)


def write_runtime_metadata(weights: Path, *, confidence: float, imgsz: int = 1024, iou: float = .5,
                           cuda_graph: bool = False, evidence_policy=None) -> Path:
    """Write new adjacent metadata for a qualified standard YOLO9 face checkpoint.

    Confidence is explicit because its operating point must come from evaluation.
    This records checkpoint identity and settings, not training-data provenance
    or a license; those accompanying files must be supplied separately.
    Graph execution and evidence retention must pass the downstream quality gate.
    Missing evidence policy records the default evidence floors, not a new qualification.
    """
    import torch

    weights = Path(weights)
    target = weights.with_name("meta.json")
    if target.exists():
        raise FileExistsError(target)
    inference = inference_settings(dict(imgsz=imgsz, confidence=confidence, iou=iou, cuda_graph=cuda_graph))
    policy = resolve_policy(evidence_policy)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("model_family") != "yolo9" or
            checkpoint.get("size") not in ("t", "s", "m", "c") or checkpoint.get("nc") != 1 or
            checkpoint.get("task", "detect") != "detect" or
            checkpoint.get("names") not in ({0: "face"}, {"0": "face"})):
        raise ValueError("Metadata requires a standard YOLO9 face checkpoint with a known size")
    metadata = dict(schema_version=1, backend="libreyolo", architecture=f"yolo9-{checkpoint['size']}",
                    classes={"0": "face"}, weights_sha256=file_hash(weights),
                    inference=inference,
                    evidence_policy=policy.to_dict())
    if not cuda_graph:
        del inference["cuda_graph"]
    with target.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return target


def main(argv=None):
    """Export a new inference file while preserving its full training checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", action="store_true", help="Also create adjacent backend metadata using the evaluated operating point")
    parser.add_argument("--confidence", type=float, help="Required with --metadata; use the qualified threshold")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--cuda-graph", action="store_true", help="Record qualified bounded LibreYOLO CUDA graph execution")
    parser.add_argument("--evidence-policy", type=Path, help="Qualified retention-policy JSON; requires --metadata")
    args = parser.parse_args(argv)
    if args.evidence_policy and not args.metadata:
        parser.error("--evidence-policy requires --metadata")
    policy = resolve_policy(json.loads(args.evidence_policy.read_text(encoding="utf-8")) if args.evidence_policy else None)
    if args.metadata and args.confidence is None:
        parser.error("--metadata requires an explicit --confidence")
    if args.metadata and args.out.with_name("meta.json").exists():
        raise FileExistsError(args.out.with_name("meta.json"))
    report = export_inference_checkpoint(args.source, args.out)
    if args.metadata:
        report["metadata"] = str(write_runtime_metadata(args.out, confidence=args.confidence, imgsz=args.imgsz, iou=args.iou, cuda_graph=args.cuda_graph, evidence_policy=policy))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
