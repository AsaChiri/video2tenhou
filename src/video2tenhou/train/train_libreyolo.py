# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Train an isolated face-only YOLO9-S candidate from a provenance-tracked dataset.

This never replaces the production detector. A candidate requires held-out box,
crop/classifier and complete-recording validation before deployment.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib import metadata
from pathlib import Path

from video2tenhou.files import atomic_write_text
from video2tenhou.files import sha256_file as file_hash


def _check_refinement_checkpoint(names: dict, *, refine_human: bool) -> None:
    if refine_human and names != {0: "face"}:
        raise ValueError(
            "Human refinement requires an existing one-class face checkpoint"
        )


def main(argv: list[str] | None = None) -> None:
    """Train the installed LibreYOLO on recorded inputs; record success or failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--epochs", type=int, help="Defaults to 60, or 12 for human-only refinement"
    )
    parser.add_argument(
        "--refine-human",
        action="store_true",
        help="Refine a face checkpoint on human-only data at a lower learning rate",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if platform.system() == "Windows" else 4,
        help=(
            "Data-loader processes; defaults to 0 on Windows to avoid duplicated "
            "framework memory, otherwise 4"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    epochs = (
        args.epochs if args.epochs is not None else (12 if args.refine_human else 60)
    )
    args.out.mkdir(parents=True, exist_ok=False)
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from libreyolo import LibreYOLO  # noqa: PLC0415

    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    manifest_rows = [
        json.loads(line)
        for line in (args.data.parent / "manifest.jsonl").read_text().splitlines()
    ]
    if args.refine_human and any(
        row.get("origin") != "human" or row.get("reviewed") is not True
        for row in manifest_rows
    ):
        raise ValueError(
            "Human refinement requires a dataset containing only reviewed human "
            "annotations"
        )
    max_labels = max(200, 4 * max(row["boxes"] for row in manifest_rows))
    config = {
        "data": str(args.data.resolve()),
        "epochs": epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "project": str(args.out.resolve()),
        "name": "training",
        "optimizer": "adamw",
        "lr0": 0.001,
        "patience": 20,
        "eval_interval": 2,
        "save_period": 10,
        "no_aug_epochs": 10,
        "flip_prob": 0.0,
        "flipud": 0.0,
        "degrees": 5.0,
        "mosaic_scale": (0.7, 1.3),
        "mosaic_prob": 1.0,
        "mixup_prob": 0.0,
        "cache": "disk",
        "max_labels": max_labels,
        "max_det": 300,
        "amp": True,
        "cuda_graph": False,
        "faster_coco_eval": False,
        "save_plots": False,
    }
    if args.refine_human:
        config.update(
            lr0=0.0002,
            warmup_epochs=1,
            no_aug_epochs=epochs,
            mosaic_prob=0.0,
            eval_interval=1,
            save_period=2,
            patience=epochs,
        )
    report = {
        "complete": False,
        "config": config,
        "base": str(args.base.resolve()),
        "base_sha256": file_hash(args.base),
        "data_sha256": file_hash(args.data),
        "manifest_sha256": file_hash(args.data.parent / "manifest.jsonl"),
        "dataset_provenance": json.loads(
            (args.data.parent / "provenance.json").read_text()
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "libreyolo": metadata.version("libreyolo"),
        },
        "phase": "human_refinement" if args.refine_human else "mixed_training",
        "error": None,
    }
    started = time.perf_counter()
    try:
        model = LibreYOLO(str(args.base.resolve()), device=args.device)
        _check_refinement_checkpoint(model.names, refine_human=args.refine_human)
        result = model.train(**config)
        report.update(complete=True, result=result)
        for key in ("best_checkpoint", "last_checkpoint"):
            if result.get(key):
                report[key + "_sha256"] = file_hash(Path(result[key]))
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_text(
            args.out / "provenance.json", json.dumps(report, indent=2, default=str)
        )


if __name__ == "__main__":
    main()
