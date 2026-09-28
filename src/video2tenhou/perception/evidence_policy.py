"""Model-bound retention of raw detections before deriving observation structure.

Policies do not rescale scores or change recognition. Sparse reads remain reusable
when retention changes; caches of filtered evidence must include the relevant
stage fingerprint. Missing metadata selects the default filtering contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


KINDS = ("hand", "pond", "meld")
PREPARATION_VERSION = 1


def _floors(values) -> tuple[float, float, float]:
    if not isinstance(values, dict) or set(values) != set(KINDS):
        raise ValueError("Evidence policy must specify exactly hand, pond and meld floors")
    result = tuple(values[kind] for kind in KINDS)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or not 0 <= value <= 1 for value in result):
        raise ValueError("Evidence floors must be finite numbers in [0, 1]")
    return tuple(float(value) for value in result)


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EvidencePolicy:
    """Immutable confidence floors, ordered hand/pond/meld, for both consumers.

    Prefer ``resolve_policy`` for metadata input. Floors act on stored confidence
    without renormalization. Sparse preparation also retains the
    ``p(none) < .5`` condition; dense preparation has no additional none filter.
    """
    sparse: tuple[float, float, float]
    dense: tuple[float, float, float]

    def __post_init__(self):
        for stage in ("sparse", "dense"):
            values = getattr(self, stage)
            if not isinstance(values, tuple) or len(values) != len(KINDS):
                raise ValueError("Evidence policy requires three immutable floors per stage")
            object.__setattr__(self, stage, _floors(dict(zip(KINDS, values))))

    def minimum(self, stage: str, kind: str) -> float:
        """Return the inclusive detection floor; unknown stages/kinds are errors."""
        if stage not in ("sparse", "dense") or kind not in KINDS:
            raise ValueError(f"Unknown evidence stage/kind: {stage}/{kind}")
        return getattr(self, stage)[KINDS.index(kind)]

    def to_dict(self) -> dict:
        """Return independent JSON metadata; callers cannot mutate this policy."""
        return {"schema_version": 1, **{stage: dict(zip(KINDS, getattr(self, stage)))
                                       for stage in ("sparse", "dense")}}

    @property
    def fingerprint(self) -> str:
        """Identify the full policy for model provenance, excluding recognition."""
        return _digest({"preparation_version": PREPARATION_VERSION, **self.to_dict()})

    def fingerprint_for(self, stage: str) -> str:
        """Identify one derived-evidence stage without invalidating the other."""
        self.minimum(stage, "hand")
        return _digest({"schema_version": 1, "preparation_version": PREPARATION_VERSION,
                        "stage": stage, "floors": self.to_dict()[stage],
                        "none_max_exclusive": .5 if stage == "sparse" else None})


DEFAULT_POLICY = EvidencePolicy((.2, .2, .35), (.2, .2, .2))


def resolve_policy(value=None) -> EvidencePolicy:
    """Validate complete schema-1 metadata, or resolve absent metadata to the default policy.

    Unknown fields, partial maps, nonfinite values and unsupported schemas fail
    closed. Explicit metadata does not itself establish that a policy has passed
    downstream quality evaluation for its detector/classifier pair.
    """
    if value is None:
        return DEFAULT_POLICY
    if isinstance(value, EvidencePolicy):
        return value
    if (not isinstance(value, dict) or set(value) != {"schema_version", "sparse", "dense"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("Unsupported or incomplete evidence policy metadata")
    return EvidencePolicy(_floors(value["sparse"]), _floors(value["dense"]))


def load_policy(metadata_path: str | Path | None = None) -> EvidencePolicy:
    """Read retention metadata without importing models or hashing checkpoints.

    Missing metadata selects the default policy. Malformed metadata fails closed;
    this lightweight cache check does not verify checkpoint contents. ``Detector``
    separately verifies the checkpoint hash before inference.
    """
    if metadata_path is None:
        from ..paths import MODEL_DIR
        metadata_path = MODEL_DIR / "detector" / "meta.json"
    path = Path(metadata_path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_POLICY
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read detector evidence metadata: {path}") from error
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1):
        raise ValueError("Unsupported detector metadata schema for evidence policy")
    return resolve_policy(metadata.get("evidence_policy"))


def restructure(kind: str, reading: dict) -> None:
    """Recompute stored box roles/order in place using the reader's geometry rules.

    Raw probabilities and confidence are unchanged. Invalid pond layouts are
    rejected after filtering, using the same geometry rules as live readings.
    """
    import numpy as np
    from .reader import Box, assign_hand, assign_meld, assign_pond
    if kind not in ("pond", "meld", "hand"):
        return
    reading["rejected"] = False
    if not reading["boxes"]:
        return
    boxes = [Box(tuple(b["xyxy"]), b["conf"], b["sideways"], np.asarray(b["p"])) for b in reading["boxes"]]
    if kind == "pond":
        reading["rejected"] = assign_pond(boxes, region_h=reading["size"][1], region_w=reading["size"][0])
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i].role != "tile", boxes[i].row if boxes[i].row is not None else 99,
                                                          boxes[i].col if boxes[i].col is not None else 99, boxes[i].cx))
    else:
        (assign_meld if kind == "meld" else assign_hand)(boxes)
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i].group if boxes[i].group is not None else 99, boxes[i].cx))
    new = []
    for i in order:
        d = dict(reading["boxes"][i])
        for k in ("row", "col", "group"):
            d.pop(k, None)
        b = boxes[i]
        d["role"] = b.role
        d["sideways"] = b.sideways
        for k in ("row", "col", "group"):
            if getattr(b, k) is not None:
                d[k] = getattr(b, k)
        new.append(d)
    reading["boxes"] = new


def prepare_reading(kind: str, reading: dict, *, stage: str,
                    policy: EvidencePolicy | None = None, none_index: int | None = None) -> dict:
    """Return a filtered/restructured copy without changing cached raw readings.

    Sparse voting excludes classifier background predictions. Dense event
    searches apply their separate detector-confidence floors. Comparisons use serialized scores
    as stored; this function never rounds, rescales, reclassifies or votes.
    Probability arrays/lists are shared without mutation; callers should treat
    them as read-only rather than editing the retained output's probabilities.
    """
    policy = resolve_policy(policy)
    floor = policy.minimum(stage, kind)
    if stage == "sparse" and (type(none_index) is not int or none_index < 0):
        raise ValueError("Sparse evidence preparation requires the none class index")
    prepared = dict(reading)
    prepared["boxes"] = [dict(box) for box in reading["boxes"] if box["conf"] >= floor
                         and (stage != "sparse" or box["p"][none_index] < .5)]
    restructure(kind, prepared)
    return prepared
