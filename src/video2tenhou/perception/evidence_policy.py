# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Model-bound retention of raw detections before deriving observation structure.

Policies do not rescale scores or change recognition. Sparse reads remain reusable
when retention changes; caches of filtered evidence must include the relevant
stage fingerprint. Missing metadata selects the default filtering contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from video2tenhou.files import json_digest
from video2tenhou.paths import MODEL_DIR

from .reader import Box, structure
from .tiles import CLASS_INDEX

SPARSE_NONE_MAX = 0.5
PREPARATION_VERSION = 1

type Stage = Literal["sparse", "dense"]


@dataclass(frozen=True)
class Floors:
    """Inclusive detector-confidence floors per region kind."""

    hand: float
    pond: float
    meld: float


@dataclass(frozen=True)
class EvidencePolicy:
    """Confidence floors for sparse and dense readings.

    Floors act on stored confidence without renormalization. Sparse preparation
    also keeps only boxes with ``p(none) < SPARSE_NONE_MAX``; dense preparation has
    no none filter.
    """

    sparse: Floors
    dense: Floors

    def to_dict(self) -> dict:
        """Return schema-1 JSON metadata."""
        return {
            "schema_version": 1,
            "sparse": asdict(self.sparse),
            "dense": asdict(self.dense),
        }

    def fingerprint_for(self, stage: Stage) -> str:
        """Identify one derived-evidence stage without invalidating the other."""
        return json_digest(
            {
                "schema_version": 1,
                "preparation_version": PREPARATION_VERSION,
                "stage": stage,
                "floors": asdict(getattr(self, stage)),
                "none_max_exclusive": SPARSE_NONE_MAX if stage == "sparse" else None,
            }
        )


DEFAULT_POLICY = EvidencePolicy(Floors(0.2, 0.2, 0.35), Floors(0.2, 0.2, 0.2))


def _floors(values: object) -> Floors:
    if not isinstance(values, dict) or set(values) != {"hand", "pond", "meld"}:
        raise ValueError(
            "Evidence policy must specify exactly hand, pond and meld floors"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values.values()
    ):
        raise ValueError("Evidence floors must be finite numbers in [0, 1]")
    return Floors(**{kind: float(value) for kind, value in values.items()})


def resolve_policy(value: object = None) -> EvidencePolicy:
    """Validate schema-1 evidence metadata, or use the default for absent metadata.

    Unknown fields, partial maps, nonfinite values and unsupported schemas fail closed.
    Explicit metadata does not itself establish that a policy has passed downstream
    quality evaluation for its detector/classifier pair.
    """
    if value is None:
        return DEFAULT_POLICY
    if isinstance(value, EvidencePolicy):
        return value
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "sparse", "dense"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("Unsupported or incomplete evidence policy metadata")
    return EvidencePolicy(_floors(value["sparse"]), _floors(value["dense"]))


def load_policy(metadata_path: str | Path | None = None) -> EvidencePolicy:
    """Read retention metadata without importing models or hashing checkpoints.

    Missing metadata selects the default policy. Malformed metadata fails closed;
    this lightweight cache check does not verify checkpoint contents. ``Detector``
    separately verifies the checkpoint hash before inference.
    """
    if metadata_path is None:
        metadata_path = MODEL_DIR / "detector" / "meta.json"
    path = Path(metadata_path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_POLICY
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
    ):
        raise ValueError("Unsupported detector metadata schema for evidence policy")
    return resolve_policy(metadata.get("evidence_policy"))


def prepare_reading(
    kind: str,
    reading: dict,
    *,
    stage: Stage,
    policy: EvidencePolicy | None = None,
) -> dict:
    """Return a filtered copy with recomputed structure; cached raw readings are kept.

    Sparse voting also excludes classifier background predictions. Comparisons use
    serialized scores as stored; this function never rounds, rescales, reclassifies
    or votes. Box roles, rows, columns, groups and order are recomputed with the
    reader's geometry rules, which also reject impossible pond layouts. Probability
    lists are shared with the input and must be treated as read-only.
    """
    floor = getattr(getattr(resolve_policy(policy), stage), kind)
    none = CLASS_INDEX["none"]
    kept = [
        box
        for box in reading["boxes"]
        if box["conf"] >= floor
        and (stage == "dense" or box["p"][none] < SPARSE_NONE_MAX)
    ]
    boxes = [
        Box(tuple(b["xyxy"]), b["conf"], sideways=b["sideways"], p=np.asarray(b["p"]))
        for b in kept
    ]
    source = {id(box): raw for box, raw in zip(boxes, kept, strict=True)}
    rejected = structure(kind, boxes, reading["size"])
    return {
        **reading,
        "rejected": rejected,
        "boxes": [_restructured(source[id(box)], box) for box in boxes],
    }


def _restructured(raw: dict, box: Box) -> dict:
    out = {
        key: value for key, value in raw.items() if key not in ("row", "col", "group")
    }
    out["role"], out["sideways"] = box.role, box.sideways
    for key in ("row", "col", "group"):
        if getattr(box, key) is not None:
            out[key] = getattr(box, key)
    return out
