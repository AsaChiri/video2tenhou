# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-recording evidence, saved answers, calibration and hand freshness.

Files on disk are the source of truth: hands, calibration, decodes and exports are
re-read when their modification time or size changes. Slow work (frame seeks,
model loading, clip encoding) never runs while `ReviewState.lock` is held; the lock
only guards caches and the answer journal. Reads never change evidence or
answers; the only file a read creates is a cached evidence clip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING

import cv2
import numpy as np

from video2tenhou import calibfit, video
from video2tenhou.cache import source_identity
from video2tenhou.commands import executable
from video2tenhou.engine.decode import DECODER_VERSION
from video2tenhou.engine.review import decode_context, load_facts
from video2tenhou.engine.validation import review_artifact
from video2tenhou.export import decode_identity, export_inputs
from video2tenhou.files import (
    atomic_write_json,
    atomic_write_text,
    read_published_text,
)
from video2tenhou.layout import CORNERS, Calibration, box_to_quad, fit_path
from video2tenhou.observe import load_obs
from video2tenhou.paths import DATA_DIR as ROOT
from video2tenhou.perception import detector
from video2tenhou.perception.crops import region_upright
from video2tenhou.perception.reader import read_region
from video2tenhou.record import from_dict

if TYPE_CHECKING:
    from pathlib import Path

    from video2tenhou.perception.classifier import Classifier

    from .processes import ProcessOwner

FRAME_CACHE_LIMIT = 40
FACT_TIMESTAMP_TOLERANCE = 1e-6
MIN_PREVIEW_SCALE = 0.05
MAX_PREVIEW_SCALE = 4.0
SCALE_CHANGE_TOLERANCE = 1e-6
MIN_ROLL_CORRECTION = 0.5
CONTEXT_TURN_WINDOW = 1.5
BORDER_CHECKS = "border-checks.json"
FACT_KINDS = frozenset(
    {
        "haipai",
        "final_hand",
        "draw",
        "discard",
        "missing_discard",
        "meld",
        "meld_remove",
        "riichi",
        "riichi_turn",
        "kan_time",
        "ura",
        "dora",
        "lost",
        "site_wrong",
        "dismiss",
    }
)
# Reconstruction internals and developer reasoning never reach the browser.
PRIVATE_DECODE_KEYS = (
    "diagnostics",
    "decode_context",
    "decode_inputs",
    "decoder_version",
)
LOGGER = logging.getLogger(__name__)


def item_id(item: dict, kind: str | None = None) -> str:
    """Identify a review question across rebuilds within its hand.

    The kind, seat and turn (or the rounded second when there is no turn) are
    stable when the same question is asked again.
    """
    j, t = item.get("j"), item.get("t")
    position = j if j is not None else (round(t) if t is not None else "")
    return f"{kind or item.get('kind')}:{item.get('seat') or ''}:{position}"


def _round(row: dict) -> tuple:
    return row.get("game"), row.get("kyoku"), row.get("honba")


class ReviewState:
    """One recording's review data, evidence rendering and saved answers."""

    def __init__(
        self, video: Path, work: Path, layout: str, out: Path, processes: ProcessOwner
    ) -> None:
        """Bind a recording's work, label and output folders without touching them."""
        self.video = video
        self.name = video.stem
        self.layout = layout
        self.work = work / self.name
        self.out = out / self.name
        self.labels = ROOT / "labels" / self.name
        self.processes = processes
        self.lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._models: tuple[detector.Detector, Classifier] | None = None
        self._frames: dict[float, np.ndarray] = {}
        self._files: dict[Path, tuple[tuple[int, int], object]] = {}
        self._views: dict[Path, tuple[tuple[int, int], dict]] = {}
        self._cal: tuple[tuple[int, int] | None, Calibration] | None = None
        self._reported_notes: set[str] = set()

    # -- cached inputs ---------------------------------------------------------

    def _load(self, path: Path) -> object | None:
        """Parse a JSON file, reusing it until the file changes; None when absent."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            cached = self._files.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        value = json.loads(read_published_text(path))
        with self.lock:
            self._files[path] = (key, value)
        return value

    @property
    def hands(self) -> list[dict]:
        """Hands of the current analysis; none while its inputs await analysis."""
        if (self.work / "inputs.changed").exists():
            return []
        hands = self._load(self.work / "hands.json")
        if hands is not None and not isinstance(hands, list):
            raise TypeError(f"Expected a JSON list in {self.work / 'hands.json'}")
        return hands or []

    def entry(self, hand: int) -> dict:
        """Return a hand's metadata from the current analysis."""
        found = next((h for h in self.hands if h["hand"] == hand), None)
        if found is None:
            raise KeyError("Unknown hand.")
        return found

    @property
    def cal(self) -> Calibration:
        """The layout with this recording's current fit applied."""
        path = fit_path(self.video)
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            key = None
        with self.lock:
            cached = self._cal
        if cached is not None and cached[0] == key:
            return cached[1]
        cal = Calibration.load(self.layout, self.video)
        with self.lock:
            self._cal = (key, cal)
        return cal

    def decode_path(self, i: int) -> Path:
        """Locate a hand's reconstruction within this recording's work directory."""
        return self.work / "decode" / f"{i:02d}.json"

    def decoded(self, hand: int) -> dict | None:
        """Return a hand's reconstruction by the current decoder, else None.

        A decode saved by another decoder version is stale: it is never shown or
        assembled, and its hand needs a rebuild.
        """
        value = self._load(self.decode_path(hand))
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {self.decode_path(hand)}")
        if value is None or value.get("decoder_version") != DECODER_VERSION:
            return None
        return value

    def review_decode(self, hand: int) -> dict | None:
        """Return a hand's reconstruction with its current export checks applied."""
        path = self.decode_path(hand)
        decoded = self.decoded(hand)
        if not decoded:
            return None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            cached = self._views.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        view = review_artifact(decoded, self.entry(hand))
        with self.lock:
            self._views[path] = (key, view)
        return view

    def all_facts(self) -> list[dict]:
        """Read the append-only human evidence file for this recording."""
        return load_facts(self.labels)

    def facts(self, hand: int | None = None) -> list[dict]:
        """Read human facts, optionally restricted to one hand's game and round."""
        rows = self.all_facts()
        if hand is None:
            return rows
        e = self.entry(hand)
        return [r for r in rows if _round(r) == _round(e)]

    # -- freshness -------------------------------------------------------------

    def pending(self) -> list[int]:
        """Return hands whose reconstruction or export lags the current inputs.

        A hand is pending when it has no decode by the current decoder, when its
        decode was made from other facts (dismissals excluded), metadata or site
        result, or when the exports were not built from its current decode file.
        """
        hands = self.hands
        record = self._load(self.work / "record.json")
        if not isinstance(record, list):
            return [h["hand"] for h in hands]
        games = [from_dict(row) for row in record]
        facts = self.all_facts()
        built = export_inputs(self.out)
        stale = []
        for h in hands:
            decoded = self.decoded(h["hand"])
            current = decoded is not None and decoded.get(
                "decode_context"
            ) == decode_context(h, games[h["game"]].hands[h["site_index"]], facts)
            if not current or built.get(str(h["hand"])) != decode_identity(
                self.decode_path(h["hand"])
            ):
                stale.append(h["hand"])
        return stale

    def revision(self) -> str:
        """Return a token that changes when review inputs or results change on disk."""
        paths = [
            self.work / "hands.json",
            self.labels / "facts.jsonl",
            fit_path(self.video),
            self.work / "inputs.changed",
            self.out / "export-inputs.json",
            *sorted((self.work / "decode").glob("*.json")),
        ]
        stats = []
        for p in paths:
            with suppress(FileNotFoundError):
                stat = p.stat()
                stats.append((p.name, stat.st_mtime_ns, stat.st_size))
        return hashlib.sha1(repr(stats).encode(), usedforsecurity=False).hexdigest()

    # -- questions and answers -------------------------------------------------

    def hand_summary(self) -> list[dict]:
        """List hands with their review status and whether an update is pending."""
        pending = set(self.pending())
        rows = []
        for h in self.hands:
            d = self.review_decode(h["hand"])
            status = None
            if d:
                status = (
                    "conflict"
                    if any(i["kind"] == "conflict" for i in d["items"])
                    else "unresolvable"
                    if d.get("solver", {}).get("status") == "unsolved"
                    else ("review" if d["items"] else "complete")
                )
            rows.append(
                {
                    **{
                        key: h.get(key)
                        for key in (
                            "hand",
                            "game",
                            "kyoku",
                            "honba",
                            "t_start",
                            "t_end",
                        )
                    },
                    "status": status,
                    "pending": h["hand"] in pending,
                    "turns": d["stats"]["turns"] if d else None,
                    "score": (d["score"] or {}).get("match") if d else None,
                }
            )
        return rows

    def _dismissed(self, entry: dict, items: list[dict]) -> set[str]:
        """Return the ids of a hand's questions the reviewer dismissed.

        Saved `note` answers from earlier versions dismiss the single question with
        the same text; ambiguous or unmatched notes are ignored and logged once.
        The journal itself is never rewritten.
        """
        ids: set[str] = set()
        for fact in self.all_facts():
            if _round(fact) != _round(entry):
                continue
            if fact.get("kind") == "dismiss" and isinstance(fact.get("item"), str):
                ids.add(fact["item"])
            elif fact.get("kind") == "note":
                same = [it for it in items if it.get("text") == fact.get("text")]
                identity = json.dumps(fact, sort_keys=True)
                if len(same) == 1:
                    ids.add(item_id(same[0]))
                elif identity not in self._reported_notes:
                    self._reported_notes.add(identity)
                    LOGGER.warning(
                        "Ignoring a saved note for hand %s that matches %s questions.",
                        entry["hand"] + 1,
                        len(same),
                    )
        return ids

    def all_items(self) -> list[dict]:
        """List open questions with stable ids and certified confidence.

        Dismissed questions are omitted. Review items can omit margin/coverage
        fields present in their confidence rows; those are filled for navigation
        without rewriting decoded evidence or using a feasible alternative's gap as
        a confidence certificate.
        """
        out = []
        for h in self.hands:
            d = self.review_decode(h["hand"])
            if d is None:
                continue
            dismissed = self._dismissed(h, d["items"])
            for it in d["items"]:
                row = dict(it)
                row["id"] = item_id(it)
                if row["id"] in dismissed:
                    continue
                field = row.get("kind")
                turn = row.get("j")
                if field in ("draw", "discard") and turn is not None:
                    confidence = next(
                        (
                            c
                            for c in d.get("confidence", [])
                            if c.get("field") == field
                            and c.get("seat") == row.get("seat")
                            and c.get("turn") == turn
                        ),
                        None,
                    )
                    if confidence is not None:
                        row.setdefault("lost", bool(confidence.get("lost")))
                        if row.get("margin") is None:
                            row["margin"] = confidence.get("margin")
                if "choices" in row:
                    row["choices"] = [
                        {**c, "id": item_id(c, c.get("field") or c.get("kind"))}
                        for c in row["choices"]
                    ]
                row.update(
                    hand=h["hand"], kyoku=h["kyoku"], honba=h["honba"], game=h["game"]
                )
                out.append(row)
        return out

    def hand_view(self, hand: int) -> dict:
        """Return a hand's metadata, its reconstruction and answers it ignored."""
        entry = self.entry(hand)
        d = self.review_decode(hand)
        if d is None:
            return {"entry": entry, "decode": None, "ignored": []}
        return {
            "entry": entry,
            "decode": {k: v for k, v in d.items() if k not in PRIVATE_DECODE_KEYS},
            "ignored": self._ignored(entry, d.get("ignored_facts", [])),
        }

    def _ignored(self, entry: dict, ignored: list[dict]) -> list[dict]:
        """Attach each answer the reconstruction could not apply to its saved fact."""
        facts = [f for f in self.all_facts() if _round(f) == _round(entry)]

        def matches(fact: dict, row: dict) -> bool:
            seat = entry["corner_wind"].get(fact.get("corner") or "", fact.get("seat"))
            times = [fact.get(key) for key in ("t", "t_discard")]
            return (
                fact.get("kind") == row.get("kind")
                and (row.get("seat") is None or seat == row["seat"])
                and (
                    row.get("t") is None
                    or any(
                        t is not None and abs(t - row["t"]) <= FACT_TIMESTAMP_TOLERANCE
                        for t in times
                    )
                )
            )

        return [
            {
                "ts": next((f.get("ts") for f in facts if matches(f, row)), None),
                "kind": row.get("kind"),
                "reason": row.get("reason"),
            }
            for row in ignored
        ]

    def add_fact(self, f: dict) -> dict:
        """Append a review answer with game identity, mapped corner and timestamp."""
        if f.get("kind") not in FACT_KINDS:
            raise ValueError("Unsupported answer.")
        if f["kind"] == "dismiss" and not isinstance(f.get("item"), str):
            raise ValueError("Choose the question to dismiss.")
        e = self.entry(int(f["hand"]))
        row = {
            **f,
            "game": e["game"],
            "kyoku": e["kyoku"],
            "honba": e["honba"],
            "hand": e["hand"],
            "author": "tool",
        }
        if row.get("seat") and "corner" not in row:
            corner = next(
                (c for c, s in e["corner_wind"].items() if s == row["seat"]), None
            )
            if corner is None:
                raise ValueError(f"seat {row['seat']} is not at this table")
            row["corner"] = corner
        path = self.labels / "facts.jsonl"
        with self.lock:
            row["ts"] = time.time()
            previous = path.read_text(encoding="utf-8") if path.exists() else ""
            if previous and not previous.endswith("\n"):
                previous += "\n"
            # The decoder reads in another process. Publish a complete journal
            # so an answer arriving mid-read cannot expose a partial JSON line.
            atomic_write_text(
                path,
                previous + json.dumps(row, ensure_ascii=False) + "\n",
                retry_windows=True,
            )
        return row

    def delete_fact(self, ts: float) -> int:
        """Remove an answer by its timestamp; the next rebuild reopens its question."""
        p = self.labels / "facts.jsonl"
        with self.lock:
            rows = (
                [
                    line
                    for line in p.read_text(encoding="utf-8").splitlines(keepends=True)
                    if line.strip()
                ]
                if p.exists()
                else []
            )
            keep = [
                line
                for line in rows
                if abs(float(json.loads(line).get("ts") or -1) - ts)
                > FACT_TIMESTAMP_TOLERANCE
            ]
            if len(keep) != len(rows):
                atomic_write_text(
                    p,
                    "".join(keep)
                    + ("\n" if keep and not keep[-1].endswith("\n") else ""),
                    retry_windows=True,
                )
        return len(rows) - len(keep)

    def context(self, hand: int, seat: str, t: float) -> dict:
        """Compare nearby camera readings with the reconstruction around time t."""
        e = self.entry(hand)
        corner = next(c for c, s in e["corner_wind"].items() if s == seat)
        obs = load_obs(self.work, hand).get(f"hand:{corner}", [])

        def brief(o: dict) -> dict:
            return {
                "t0": o["t0"],
                "t1": o["t1"],
                "count": o["count"],
                "n": o["n_used"],
                "partial": bool(o.get("partial")),
                "tiles": [
                    {"tile": s["tile"], "conf": s["conf"], "disagree": s["disagree"]}
                    for s in o["slots"]
                ],
            }

        d = self.decoded(hand)
        turn = next(
            (
                x
                for x in (d or {}).get("turns", [])
                if x["seat"] == seat and abs(x["t"] - t) < CONTEXT_TURN_WINDOW
            ),
            None,
        )
        return {
            "corner": corner,
            "read_before": [brief(o) for o in obs if o["t1"] <= t + 0.5][-2:],
            "read_after": [brief(o) for o in obs if o["t0"] >= t - 0.5][:2],
            "turn": turn,
        }

    # -- calibration (stage 0) -------------------------------------------------

    def plate(self) -> np.ndarray | None:
        """Return the prepared median table image, or None when missing or stale.

        Only reads: the plate is (re)built by preparation, never by a request.
        """
        try:
            current = calibfit.plate_current(self.video, self.work)
        except FileNotFoundError:  # the recording is gone
            return None
        return cv2.imread(str(calibfit.plate_path(self.work))) if current else None

    def calib(self) -> dict:
        """Return region quadrilaterals in frame coordinates for the calibration editor.

        Border checks are included when they were measured on this geometry.
        """
        cal = self.cal
        out = {
            "video": self.name,
            "layout": cal.name,
            "fit": cal.fit,
            "overhead": {
                "center": list(cal.center),
                "angle": cal.angle,
                "scale": cal.scale,
            },
            "frame": list(cal.frame),
            "regions": {},
        }
        for c in CORNERS:
            r, _k, _s = cal.pond[c]
            out["regions"][f"pond:{c}"] = {
                "kind": "pond",
                "corner": c,
                "quad": box_to_quad(cal.derotation(), r.xyxy),
                "rect": [r.x, r.y, r.w, r.h],
                "movable": False,
            }
            for kind, rect in (("hand", cal.hand[c][0]), ("meld", cal.meld[c][0])):
                out["regions"][f"{kind}:{c}"] = {
                    "kind": kind,
                    "corner": c,
                    "rect": [rect.x, rect.y, rect.w, rect.h],
                    "quad": [
                        [rect.x, rect.y],
                        [rect.x + rect.w, rect.y],
                        [rect.x + rect.w, rect.y + rect.h],
                        [rect.x, rect.y + rect.h],
                    ],
                    "roll": cal.roll(c) if kind == "hand" else None,
                    "movable": True,
                }
        u = cal.unit
        out["regions"]["unit"] = {
            "kind": "unit",
            "corner": None,
            "movable": False,
            "quad": box_to_quad(cal.derotation(), u.xyxy),
            "rect": [u.x, u.y, u.w, u.h],
        }
        saved = self._load(self.work / BORDER_CHECKS)
        out["checks"] = (
            saved["checks"]
            if isinstance(saved, dict) and saved.get("geometry") == cal.data
            else {}
        )
        return out

    def save_checks(self, checks: dict) -> None:
        """Remember border checks together with the geometry they measured."""
        atomic_write_json(
            self.work / BORDER_CHECKS, {"geometry": self.cal.data, "checks": checks}
        )

    def save_calib(self, body: dict) -> dict:
        """Save changed table regions, preserving the rest of the fit.

        Changing the geometry of an analyzed recording blocks review rebuilds until
        analysis refreshes the readings.
        """
        previous = self.cal.data
        p = fit_path(self.video)
        fit: dict = (
            json.loads(p.read_text(encoding="utf-8"))
            if p.exists()
            else {"video": self.name, "layout": self.layout}
        )
        parts = [part for part in ("overhead", "hand", "meld", "cam") if part in body]
        for part in parts:
            if part == "overhead":
                fit["overhead"] = {
                    **(fit.get("overhead") or {}),
                    **body["overhead"],
                    "source": "human",
                }
            else:
                cur = fit.get(part) or {}
                for c, v in body[part].items():
                    cur[c] = {**(cur.get(c) or {}), **v, "source": "human"}
                fit[part] = cur
        if parts:
            fit["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            atomic_write_json(p, fit, indent=1)
        self.geometry_saved(previous)
        return {"fit": self.cal.fit}

    def geometry_saved(self, previous: dict) -> None:
        """Block review rebuilds of an analyzed recording whose geometry changed."""
        if self.cal.data != previous and (self.work / "hands.json").exists():
            (self.work / "calibration.changed").write_text(
                "Run full analysis to refresh observations after changing geometry.\n",
                encoding="utf-8",
            )

    # -- evidence --------------------------------------------------------------

    def frame(self, t: float) -> np.ndarray:
        """Seek a normalized BGR frame, reusing recent evidence requests."""
        t = round(float(t), 2)
        with self.lock:
            cached = self._frames.get(t)
        if cached is not None:
            return cached
        frame = video.frame_at(self.video, t)
        with self.lock:
            if len(self._frames) >= FRAME_CACHE_LIMIT:
                self._frames.clear()
            self._frames[t] = frame
        return frame

    @property
    def models_loaded(self) -> bool:
        """Report whether this review state owns inference models."""
        return self._models is not None

    def models(self) -> tuple[detector.Detector, Classifier]:
        """Load detector/classifier lazily for label prefill, once per state.

        Prefill boxes are suggestions, never cached evidence, so the detector runs
        eagerly: CUDA graphs would keep one model per crop shape until the project
        closes.
        """
        with self._model_lock:
            if self._models is None:
                from video2tenhou.perception.classifier import (  # noqa: PLC0415
                    Classifier,
                )

                eager = detector.InferenceOptions(cuda_graph=False)
                self._models = (detector.Detector(settings=eager), Classifier())
            return self._models

    def render(self, t: float, region: str, scale: float = 1.0) -> bytes:
        """Render a frame or calibrated upright region as JPEG evidence."""
        cal = self.cal
        if (
            region not in ["frame", *cal.regions()]
            or not MIN_PREVIEW_SCALE <= scale <= MAX_PREVIEW_SCALE
        ):
            raise ValueError("Unknown region or unsupported image scale.")
        img = self.frame(t)
        if region != "frame":
            kind, _, corner = region.partition(":")
            img, _ = region_upright(img, cal, kind, corner)
        if scale != 1.0:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError("Could not encode the evidence image")
        return buf.tobytes()

    def read(self, t: float, region: str) -> dict:
        """Prefill editable boxes with model predictions and region coordinates."""
        det, clf = self.models()
        kind, _, corner = region.partition(":")
        img, _ = region_upright(self.frame(t), self.cal, kind, corner)
        rd = read_region(img, region, det, clf, t=t)
        return {
            "size": list(rd.size),
            "boxes": [
                {
                    "xyxy": [round(v, 1) for v in b.xyxy],
                    "conf": round(b.conf, 3),
                    "sideways": b.sideways,
                    "role": b.role,
                    "tile": clf.classes[int(np.argmax(b.p))],
                    "p": round(float(np.max(b.p)), 3),
                    "row": b.row,
                    "col": b.col,
                    "group": b.group,
                }
                for b in rd.boxes
            ],
        }

    def save_label(self, lab: dict) -> Path:
        """Store region-space tile boxes as frame-coordinate training labels."""
        kind, corner, t = lab["kind"], lab["corner"], float(lab["t"])
        _, transform = region_upright(self.frame(t), self.cal, kind, corner)
        boxes = [
            {
                "quad": box_to_quad(transform, b["xyxy"]),
                "tile": b.get("tile", "?"),
                "sideways": bool(b.get("sideways")),
                "role": b.get("role", "tile"),
            }
            for b in lab["boxes"]
        ]
        out = {
            "video": self.name,
            "t": t,
            "kind": kind,
            "corner": corner,
            "boxes": boxes,
            "source": "tool",
        }
        for k in ("tiles", "rows", "melds"):
            if k in lab:
                out[k] = lab[k]
        p = self.labels / "boxes" / f"{kind}_{corner}_{round(t)}.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            bak = p.with_suffix(".orig.json")
            # keep the human label once
            if old.get("source") != "tool" and not bak.exists():
                bak.write_text(
                    json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8"
                )
        atomic_write_json(p, out, indent=1)
        return p

    def clip(self, t0: float, t1: float, region: str) -> Path:
        """Return an MP4 of a camera region between t0 and t1, encoding it once.

        Clips are cached under work/<video>/clips with the geometry and source
        identity in their names; an encode publishes only a complete file.
        """
        cal = self.cal
        if region not in ["frame", *cal.regions()]:
            raise ValueError("Unknown video region.")
        t0, t1 = max(0.0, t0), max(t0 + 1.0, t1)
        geometry = hashlib.sha256(
            json.dumps([cal.data, source_identity(self.video)], sort_keys=True).encode()
        ).hexdigest()[:12]
        path = (
            self.work
            / "clips"
            / f"{geometry}_{region.replace(':', '_')}_{t0:.1f}_{t1:.1f}.mp4"
        )
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.mp4")
        cmd = [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{t0:.2f}",
            "-i",
            str(self.video),
            "-t",
            f"{t1 - t0:.2f}",
            "-vf",
            _clip_filter(cal, region) + ",scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            "10",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(pending),
        ]
        try:
            with self.processes.spawn(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            ) as process:
                _, errors = process.communicate()
            if process.returncode:
                LOGGER.error("Evidence clip encoding failed: %s", errors.strip())
                raise OSError("Could not encode this evidence clip.")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return path

    def close(self) -> None:
        """Release this state's inference models without waiting for a load."""
        self._models = None


def _clip_filter(cal: Calibration, region: str) -> str:
    """Build the ffmpeg filter matching layout.Calibration.transform for a region.

    De-rotate the table, cut the region, turn it upright and scale. ffmpeg's
    rotate turns clockwise, OpenCV's counter-clockwise.
    """
    kind, _, corner = region.partition(":")
    if kind in ("hand", "meld", "cam"):
        rect, scale = cal.panel(region)
        vf = f"crop={rect.w}:{rect.h}:{rect.x}:{rect.y}"
        if scale and abs(scale - 1.0) > SCALE_CHANGE_TOLERANCE:
            vf += f",scale=iw*{scale:g}:ih*{scale:g}"
        roll = cal.roll(corner) if kind == "hand" else 0.0
        if abs(roll) > MIN_ROLL_CORRECTION:
            angle = -roll * math.pi / 180
            vf += f",rotate={angle:.5f}:ow=rotw({angle:.5f}):oh=roth({angle:.5f})"
            vf += ":c=black"
        return vf
    if kind in ("pond", "overhead"):
        cx, cy, side = cal.center[0], cal.center[1], cal.side
        vf = (
            f"rotate={-cal.angle * math.pi / 180:.5f}:ow=iw:oh=ih:c=black,"
            f"crop={side}:{side}:{cx - side / 2:.0f}:{cy - side / 2:.0f}"
        )
        if kind == "pond":
            rect, k, scale = cal.pond[corner]
            vf += f",crop={rect.w}:{rect.h}:{rect.x}:{rect.y}" + ",transpose=1" * (
                k % 4
            )
            if scale and abs(scale - 1.0) > SCALE_CHANGE_TOLERANCE:
                vf += f",scale=iw*{scale:g}:ih*{scale:g}"
        return vf
    return "scale=960:-2"
