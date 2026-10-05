# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Held-out metrics per stage (docs/DESIGN.md section 7).

uv run python -m video2tenhou.eval detector              # face boxes per view
uv run python -m video2tenhou.eval perception <video>    # detector + classifier
uv run python -m video2tenhou.eval observations <video>  # voted observations
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from video2tenhou.logging_setup import RESULT, command_logging

from .layout import Calibration, quad_to_box
from .layout import box_iou as iou
from .paths import DATA_DIR as ROOT
from .perception import detector, reader
from .perception.crops import region_upright
from .perception.evidence_policy import prepare_reading, resolve_policy
from .train import data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from video2tenhou.perception.evidence_policy import EvidencePolicy, Stage
    from video2tenhou.perception.reader import RegionDetector

CONFIRMED_SIDEWAYS_FRACTION = 0.5


def match(
    pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]], thr: float = 0.5
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by IoU; returns (pred index, gt index) pairs."""
    pairs = sorted(
        ((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
        reverse=True,
    )
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j))
    return out


def read_face_boxes(
    path: Path, width: int, height: int
) -> list[tuple[float, float, float, float]]:
    """Read face boxes in pixels; a missing file is an error, an empty one negative."""
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, cx, cy, bw, bh = (float(value) for value in line.split())
        if not np.isfinite([cx, cy, bw, bh]).all() or label != 0 or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid face annotation in {path}")
        boxes.append(
            (
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
        )
    return boxes


def evaluate_detector_images(
    det: RegionDetector, dataset: Path, *, predictions_path: Path | None = None
) -> dict:
    """Measure held-out face-box precision and recall by view, including negatives.

    ``det`` follows Detector's BGR-in, region-box-out interface. Boxes are matched
    one-to-one at IoU 0.5. The optional JSONL keeps every prediction for paired
    model or crop comparisons; predictions are not ground truth.
    """
    counts = defaultdict(lambda: defaultdict(float))
    manifest = dataset / "manifest.jsonl"
    views = (
        {
            Path(row["image"]).name: row["view"]
            for row in (json.loads(line) for line in manifest.read_text().splitlines())
        }
        if manifest.exists()
        else {}
    )
    stream = None
    try:
        if predictions_path is not None:
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            stream = predictions_path.open("x", encoding="utf-8")
        for image_path in sorted((dataset / "images" / "val").iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read evaluation image: {image_path}")
            height, width = image.shape[:2]
            gt = read_face_boxes(
                dataset / "labels" / "val" / (image_path.stem + ".txt"), width, height
            )
            predictions = det.predict(image)
            matched = match([box.xyxy for box in predictions], gt)
            view = views.get(image_path.name, image_path.stem.split("_")[0])
            row = counts[view]
            row["images"] += 1
            row["gt"] += len(gt)
            row["predicted"] += len(predictions)
            row["true_positive"] += len(matched)
            row["iou_sum"] += sum(iou(predictions[i].xyxy, gt[j]) for i, j in matched)
            if stream is not None:
                stream.write(
                    json.dumps(
                        {
                            "image": str(image_path.resolve()),
                            "view": view,
                            "width": width,
                            "height": height,
                            "ground_truth": gt,
                            "predictions": [
                                {"xyxy": box.xyxy, "conf": box.conf}
                                for box in predictions
                            ],
                            "matches": matched,
                        }
                    )
                    + "\n"
                )
                stream.flush()
    finally:
        if stream is not None:
            stream.close()
    return {
        view: {
            "images": int(row["images"]),
            "gt": int(row["gt"]),
            "predicted": int(row["predicted"]),
            "true_positive": int(row["true_positive"]),
            "precision": round(row["true_positive"] / max(1, row["predicted"]), 6),
            "recall": round(row["true_positive"] / max(1, row["gt"]), 6),
            "mean_matched_iou": round(row["iou_sum"] / max(1, row["true_positive"]), 6),
        }
        for view, row in sorted(counts.items())
    }


@dataclass
class ReadingScore:
    """Human matches for one reading.

    ``kept`` indexes the scored reading boxes and ``matches`` pairs a kept
    position with a ground-truth index; the other lists hold ground-truth
    indices. ``identified`` boxes name the human tile; ``usable`` ones also come
    from an accepted reading with the human role, annotated orientation and,
    for pond tiles, slot. ``slots`` are usable pond tile slots. ``slot_checks``
    holds (row and column agree, role agrees) for each matched slot annotation.
    """

    kept: list[int]
    matches: list[tuple[int, int]]
    labelled: int = 0
    identified: list[int] = field(default_factory=list)
    usable: list[int] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)
    slot_checks: list[tuple[bool, bool]] = field(default_factory=list)
    confusions: list[str] = field(default_factory=list)


def score_reading(
    kind: str, reading: dict, ground_truth: list[dict], classes: list[str]
) -> ReadingScore:
    """Match a reading's boxes to human boxes and judge each match.

    Hand labels cover the standing row only, so revealed melds beside it and
    sideways boxes are left out before matching.
    """
    kept = [
        i
        for i, box in enumerate(reading["boxes"])
        if kind != "hand" or (box.get("role", "tile") == "tile" and not box["sideways"])
    ]
    boxes = [reading["boxes"][i] for i in kept]
    score = ReadingScore(
        kept,
        match([box["xyxy"] for box in boxes], [box["xyxy"] for box in ground_truth]),
    )
    rejected = bool(reading.get("rejected", False))
    for i, j in score.matches:
        box, annotation = boxes[i], ground_truth[j]
        role = annotation.get("role", "tile")
        role_ok = box.get("role", "tile") == role
        slot_ok = (box.get("row"), box.get("col")) == (
            annotation.get("row"),
            annotation.get("col"),
        )
        if "row" in annotation:
            score.slot_checks.append((slot_ok, role_ok))
        orientation_ok = "sideways" not in annotation or box["sideways"] == bool(
            annotation["sideways"]
        )
        usable = (
            not rejected
            and role_ok
            and role != "other"
            and orientation_ok
            and (kind != "pond" or role != "tile" or ("row" in annotation and slot_ok))
        )
        if usable and kind == "pond" and role == "tile" and "row" in annotation:
            score.slots.append(j)
        if annotation["tile"] == "?":
            continue
        score.labelled += 1
        top = classes[int(np.argmax(box["p"]))]
        if top != annotation["tile"]:
            score.confusions.append(f"{kind}:{annotation['tile']}->{top}")
        else:
            score.identified.append(j)
            if usable:
                score.usable.append(j)
    return score


def _usable_labelled(ground_truth: list[dict]) -> int:
    """Count human boxes with a known tile that can be reader evidence."""
    return sum(
        box["tile"] != "?" and box.get("role", "tile") != "other"
        for box in ground_truth
    )


def _human_slots(ground_truth: list[dict]) -> int:
    """Count human pond tiles annotated with a slot."""
    return sum(
        box.get("role", "tile") == "tile" and "row" in box for box in ground_truth
    )


def _predicted_slots(boxes: list[dict]) -> int:
    """Count read boxes placed in a pond tile slot."""
    return sum(
        box.get("role", "tile") == "tile"
        and box.get("row") is not None
        and box.get("col") is not None
        for box in boxes
    )


def evaluate_retained_reading(
    reading: dict,
    ground_truth: list[dict],
    classes: list[str],
    *,
    stage: Stage,
    policy: EvidencePolicy | dict | None = None,
) -> dict:
    """Measure human matches after actual retention and geometry reassignment.

    Ground-truth boxes use region-coordinate ``xyxy`` and human ``tile``/role
    annotations. Returned indices refer to that unchanged list, supporting
    paired per-example retention gates. All boxes take part in reassignment
    before the standing-hand scope applies. This is a single-reading gate, not
    a temporal-vote or whole-hand accuracy claim.
    """
    kind = reading["region"].split(":", 1)[0]
    prepared = prepare_reading(
        kind,
        reading,
        stage=stage,
        policy=resolve_policy(policy),
    )
    score = score_reading(kind, prepared, ground_truth, classes)
    rejected = bool(prepared.get("rejected", False))
    labelled = _usable_labelled(ground_truth)
    pond = kind == "pond"
    return {
        "reading": prepared,
        "correct_gt_indices": sorted(score.usable),
        "correct": len(score.usable),
        "gt_labelled": labelled,
        "correct_of_gt": len(score.usable) / max(1, labelled),
        "predicted": len(score.kept),
        "matched": len(score.matches),
        "rejected": rejected,
        "structured_matched_slots": len(score.slots),
        "structured_gt_slots": _human_slots(ground_truth) if pond else 0,
        "structured_pred_slots": _predicted_slots(prepared["boxes"])
        if pond and not rejected
        else 0,
    }


@dataclass
class PerceptionMetrics:
    """Accumulate identity, structure and retention metrics over reviewed regions."""

    classes: list[str]
    policy: EvidencePolicy
    stats: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    retained_stats: dict = field(
        default_factory=lambda: {
            stage: defaultdict(lambda: defaultdict(int))
            for stage in ("sparse", "dense")
        }
    )
    conf_pairs: Counter = field(default_factory=Counter)
    audit: list[dict] = field(default_factory=list)

    def consume(
        self, d: dict, rd: reader.Reading, ground_truth: list[dict], hand: int | None
    ) -> None:
        """Count misses and rejected readings against the human ground truth."""
        kind = d["kind"]
        raw_reading = rd.to_dict()
        retained = {
            stage: evaluate_retained_reading(
                raw_reading, ground_truth, self.classes, stage=stage, policy=self.policy
            )
            for stage in ("sparse", "dense")
        }
        for stage, result in retained.items():
            aggregate = self.retained_stats[stage][kind]
            aggregate["images"] += 1
            for key in (
                "correct",
                "gt_labelled",
                "predicted",
                "matched",
                "rejected",
                "structured_matched_slots",
                "structured_gt_slots",
                "structured_pred_slots",
            ):
                aggregate[key] += result[key]
        # Score the reader's unrounded output; serialized readings round geometry.
        boxes = [asdict(box) for box in rd.boxes]
        score = score_reading(
            kind, {"rejected": rd.rejected, "boxes": boxes}, ground_truth, self.classes
        )
        s = self.stats[kind]
        s["images"] += 1
        s["gt"] += len(ground_truth)
        s["pred"] += len(score.kept)
        s["matched"] += len(score.matches)
        s["rejected"] += rd.rejected
        s["gt_labelled"] += sum(box["tile"] != "?" for box in ground_truth)
        s["usable_gt_labelled"] += _usable_labelled(ground_truth)
        s["labelled"] += score.labelled
        s["correct"] += len(score.identified)
        s["usable_correct"] += len(score.usable)
        self.conf_pairs.update(score.confusions)
        if kind == "pond":
            s["structured_gt_slots"] += _human_slots(ground_truth)
            s["structured_pred_slots"] += 0 if rd.rejected else _predicted_slots(boxes)
            s["structured_matched_slots"] += len(score.slots)
            s["rc_total"] += len(score.slot_checks)
            s["rc_ok"] += sum(rowcol for rowcol, _ in score.slot_checks)
            s["role_ok"] += sum(role for _, role in score.slot_checks)
        self.audit.append(
            {
                "t": d["t"],
                "region": raw_reading["region"],
                "hand": hand,
                "ground_truth": ground_truth,
                "raw_reading": raw_reading,
                "retained": retained,
                "reading": raw_reading
                | {"boxes": [raw_reading["boxes"][i] for i in score.kept]},
                "top_tiles": [
                    self.classes[int(np.argmax(boxes[i]["p"]))] for i in score.kept
                ],
                "matches": score.matches,
            }
        )

    def report(self) -> dict:
        """Normalize counters while retaining their denominators and policy."""
        out: dict = {}
        for k, s in self.stats.items():
            out[k] = {
                "images": s["images"],
                "gt": s["gt"],
                "predicted": s["pred"],
                "matched": s["matched"],
                "rejected": s["rejected"],
                "recall": round(s["matched"] / max(1, s["gt"]), 4),
                "precision": round(s["matched"] / max(1, s["pred"]), 4),
                "identity": round(s["correct"] / max(1, s["labelled"]), 4),
                "labelled": s["labelled"],
                "correct": s["correct"],
                "gt_labelled": s["gt_labelled"],
                "correct_of_gt": round(s["correct"] / max(1, s["gt_labelled"]), 4),
                "usable_correct": s["usable_correct"],
                "usable_gt_labelled": s["usable_gt_labelled"],
                "usable_correct_of_gt": round(
                    s["usable_correct"] / max(1, s["usable_gt_labelled"]), 4
                ),
            }
            if k == "pond":
                out[k]["rowcol"] = round(s["rc_ok"] / max(1, s["rc_total"]), 4)
                out[k]["role"] = round(s["role_ok"] / max(1, s["rc_total"]), 4)
                out[k]["structured_slot_recall"] = round(
                    s["structured_matched_slots"] / max(1, s["structured_gt_slots"]), 4
                )
                out[k]["structured_slot_precision"] = round(
                    s["structured_matched_slots"] / max(1, s["structured_pred_slots"]),
                    4,
                )
        out["confusions"] = self.conf_pairs.most_common(20)
        out["evidence_policy"] = self.policy.to_dict()
        out["retained"] = {}
        for stage, by_kind in self.retained_stats.items():
            out["retained"][stage] = {}
            for kind, counters in by_kind.items():
                metrics: dict[str, int | float] = dict(counters)
                metrics["correct_of_gt"] = counters["correct"] / max(
                    1, counters["gt_labelled"]
                )
                if kind == "pond":
                    metrics["structured_slot_recall"] = counters[
                        "structured_matched_slots"
                    ] / max(1, counters["structured_gt_slots"])
                    metrics["structured_slot_precision"] = counters[
                        "structured_matched_slots"
                    ] / max(1, counters["structured_pred_slots"])
                out["retained"][stage][kind] = metrics
        return out


def eval_perception(
    video_path: str,
    work: Path = ROOT / "work",
    *,
    weights: Path = detector.DEFAULT_WEIGHTS,
    conf: float | None = None,
    predictions_path: Path | None = None,
    classifier_dir: Path | None = None,
    evidence_policy: EvidencePolicy | dict | None = None,
) -> dict:
    """Evaluate region structure and tile identity on human-labeled held-out hands.

    Uses cached lossless source frames and the production classifier by default;
    ``classifier_dir`` selects an isolated candidate without replacing local models.
    Matched identity accuracy excludes missed detections; ``correct_of_gt`` also
    counts those misses. Annotated empty meld regions contribute false positives.
    ``usable_correct_of_gt`` additionally requires an accepted reading and correct
    role, annotated orientation and pond slot; rejected/ignored evidence cannot
    count as reader success. ``retained`` separately applies each consumer's
    confidence floors and reassigns structure; reader success alone is not a
    downstream gate. ``evidence_policy`` explicitly selects a development policy
    without changing model files; otherwise the detector's metadata is used.
    An optional JSONL preserves every evaluated reading for paired model audits.
    Its ``raw_reading`` retains extra boxes before the human standing-hand scope
    filter; use it when recomputing policy-dependent structure from saved evidence.
    """
    from .perception.classifier import DEFAULT_DIR, Classifier  # noqa: PLC0415

    cal = Calibration.load("pml", video_path)
    det = detector.Detector(
        weights, settings=detector.InferenceOptions(confidence=conf)
    )
    clf = Classifier(classifier_dir or DEFAULT_DIR)
    policy = resolve_policy(
        evidence_policy if evidence_policy is not None else det.evidence_policy
    )
    hands = data.hand_table(video_path, work)
    frames_dir = work / Path(video_path).stem / "frames"
    metrics = PerceptionMetrics(clf.classes, policy)
    labels = [
        d
        for d in data.load_labels(video_path)
        if data.split_of(d["t"], hands) == "val"
        and (d["boxes"] or (d["kind"] == "meld" and d.get("melds") == []))
    ]
    for d in labels:
        kind, corner = d["kind"], d["corner"]
        frame = cv2.imread(str(frames_dir / f"{float(d['t']):.3f}.png"))
        if frame is None:
            raise ValueError(
                f"Missing cached evaluation frame at {d['t']}: {frames_dir}"
            )
        img, transform = region_upright(frame, cal, kind, corner)
        rd = reader.read_region(img, f"{kind}:{corner}", det, clf, t=d["t"])
        ground_truth = [
            dict(xyxy=list(quad_to_box(transform, box["quad"])), **box)
            for box in d["boxes"]
        ]
        metrics.consume(d, rd, ground_truth, data.hand_of(d["t"], hands))
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("x", encoding="utf-8") as stream:
            for row in metrics.audit:
                stream.write(json.dumps(row) + "\n")
    return metrics.report()


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Run a selected evaluation and print its metrics as JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detector", "perception", "observations"])
    ap.add_argument("video", nargs="?")
    ap.add_argument(
        "--weights", type=Path, default=ROOT / "models" / "detector" / "weights.pt"
    )
    ap.add_argument(
        "--dataset", type=Path, default=ROOT / "work" / "datasets" / "detector"
    )
    ap.add_argument(
        "--confidence",
        type=float,
        help="Override the detection threshold selected by checkpoint metadata",
    )
    ap.add_argument("--work", type=Path, default=ROOT / "work")
    ap.add_argument(
        "--predictions",
        type=Path,
        help="Save detector predictions for a paired crop audit",
    )
    ap.add_argument(
        "--classifier",
        type=Path,
        help="Candidate classifier directory for perception evaluation",
    )
    a = ap.parse_args(argv)
    if a.what == "detector":
        det = detector.Detector(
            a.weights, settings=detector.InferenceOptions(confidence=a.confidence)
        )
        RESULT.info(
            "%s",
            json.dumps(
                evaluate_detector_images(
                    det, a.dataset, predictions_path=a.predictions
                ),
                indent=1,
            ),
        )
    elif a.what == "perception":
        RESULT.info(
            "%s",
            json.dumps(
                eval_perception(
                    a.video,
                    a.work,
                    weights=a.weights,
                    conf=a.confidence,
                    predictions_path=a.predictions,
                    classifier_dir=a.classifier,
                ),
                indent=1,
            ),
        )
    else:
        RESULT.info("%s", json.dumps(eval_observations(a.video, a.work), indent=1))


# ---------------------------------------------------------------------------
# stage 4 metrics: observations against the pond / hand labels at their times
# ---------------------------------------------------------------------------


def _obs_at(obs_list: list[dict], t: float) -> dict | None:
    for o in obs_list:
        if o["t0"] - 0.75 <= t <= o["t1"] + 0.75:
            return o
    return None


def _score_pond_observation(d: dict, o: dict, st: dict, misses: list[str]) -> None:
    """Compare keyed pond positions, identities and sideways flags."""
    gt = {}
    for b in d["boxes"]:
        if "row" in b:
            gt[(b["row"], b["col"])] = (b["tile"], b.get("sideways", False))
    pred = {
        tuple(s["key"]): (
            s["tile"],
            s["sideways"] >= CONFIRMED_SIDEWAYS_FRACTION,
        )
        for s in o["slots"]
    }
    st["pond_gt_slots"] += len(gt)
    for key, (tile, side) in gt.items():
        if key not in pred:
            st["pond_slot_missing"] += 1
            continue
        st["pond_slot_found"] += 1
        if tile != "?":
            st["pond_ident_total"] += 1
            st["pond_ident_ok"] += pred[key][0] == tile
            if pred[key][0] != tile:
                misses.append(
                    f"pond {d['corner']} {d['t']} {key} {tile}->{pred[key][0]}"
                )
        st["pond_side_ok"] += pred[key][1] == bool(side)
    st["pond_slot_extra"] += len(set(pred) - set(gt))
    st["pond_exact"] += set(pred) == set(gt) and all(
        pred[k2][0] == v[0] for k2, v in gt.items() if v[0] != "?"
    )


def _score_hand_observation(d: dict, o: dict, st: dict, misses: list[str]) -> None:
    """Compare a resting hand multiset without treating unknown labels as errors."""
    gt = Counter(b["tile"] for b in d["boxes"] if b["tile"] != "?")
    unknown = sum(1 for b in d["boxes"] if b["tile"] == "?")
    pred = Counter(s["tile"] for s in o["slots"])
    st["hand_tiles"] += sum(gt.values())
    common = sum((gt & pred).values())
    st["hand_tiles_ok"] += common
    st["hand_count_ok"] += o["count"] == len(d["boxes"])
    st["hand_exact"] += (
        (gt == pred)
        if not unknown
        else (
            sum((gt & pred).values()) == sum(gt.values())
            and o["count"] == len(d["boxes"])
        )
    )
    if gt != pred:
        misses.append(
            f"hand {d['corner']} {d['t']} gt {sorted(gt.elements())} "
            f"pred {sorted(pred.elements())}"
        )


def eval_observations(video_path: str, work: Path = ROOT / "work") -> dict:
    """Compare pond and hand observations with human tile labels.

    Ponds compare row strings, gaps and sideways tiles; hands compare multisets. Every
    labelled time that falls inside a calm interval is compared.
    """
    wdir = work / Path(video_path).stem
    hands = data.hand_table(video_path, work)
    cache: dict[int, dict] = {}
    st = defaultdict(int)
    misses = []
    for d in data.load_labels(video_path):
        if (
            d["kind"] not in ("pond", "hand")
            or not d["boxes"]
            or all(b["tile"] == "?" for b in d["boxes"])
        ):
            continue
        hi = data.hand_of(d["t"], hands)
        if hi is None:
            continue
        if hi not in cache:
            p = wdir / "obs" / f"{hi:02d}.json"
            if not p.exists():
                continue
            cache[hi] = json.loads(p.read_text(encoding="utf-8"))
        region = f"{d['kind']}:{d['corner']}"
        o = _obs_at(cache[hi].get(region, []), float(d["t"]))
        k = d["kind"]
        st[f"{k}_labels"] += 1
        if o is None:
            st[f"{k}_no_calm_obs"] += 1
            continue
        st[f"{k}_covered"] += 1
        if k == "pond":
            _score_pond_observation(d, o, st, misses)
        else:
            _score_hand_observation(d, o, st, misses)
    return {
        "pond": {
            "labels": st["pond_labels"],
            "covered": st["pond_covered"],
            "slot_recall": round(
                st["pond_slot_found"] / max(1, st["pond_gt_slots"]), 4
            ),
            "slot_extra": st["pond_slot_extra"],
            "identity": round(st["pond_ident_ok"] / max(1, st["pond_ident_total"]), 4),
            "sideways_ok": round(st["pond_side_ok"] / max(1, st["pond_slot_found"]), 4),
            "exact_pond": round(st["pond_exact"] / max(1, st["pond_covered"]), 4),
        },
        "hand": {
            "labels": st["hand_labels"],
            "covered": st["hand_covered"],
            "tile_recall": round(st["hand_tiles_ok"] / max(1, st["hand_tiles"]), 4),
            "count_ok": round(st["hand_count_ok"] / max(1, st["hand_covered"]), 4),
            "exact_multiset": round(st["hand_exact"] / max(1, st["hand_covered"]), 4),
        },
        "misses": misses[:40],
    }


if __name__ == "__main__":
    main()
