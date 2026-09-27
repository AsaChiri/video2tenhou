"""Review and label tool: a local HTTP server (stdlib) and one static page.

    uv run video2tenhou review samples/full_1080p.mp4            # http://localhost:8765

API
  GET  /api/hands                         hands.json entries with decode status and open item counts
  GET  /api/hand/<i>                      decode/<i>.json (turns, haipai, items, ...) + the entry
  GET  /api/frame?t=&region=[&scale=]     JPEG: a region render (upright) or the whole frame (region=frame)
  GET  /api/read?t=&region=               detector + classifier prefill on that region (boxes + top classes)
  GET  /api/facts?hand=<i>                facts of that hand
  POST /api/decode_all, GET /api/decode_all   re-decode every hand in the background; its status and progress
  POST /api/decode_pending, GET /api/decode_pending   rebuild only unapplied corrections; status and pending IDs
  POST /api/facts                         append one fact {kind, hand, corner|seat, t, tiles|tile|..., note}
  POST /api/label                         save a box label {t, kind, corner, boxes:[{xyxy (region px), tile, sideways}]}
  POST /api/decode/<i>                    start the re-decode of one hand (facts applied) in the background
  GET  /api/decode/<i>                    status of that re-decode {running, started, done, error}
  GET  /api/plate                         JPEG of the table plate (the median frame stage 0 fits on)
  GET  /api/calib                         this video's geometry: every region as a frame-coordinate quad, plus the fit
  POST /api/calib                         save the fit {overhead?, hand?, meld?, cam?} to labels/<video>/calib.json
  POST /api/calib/fit                     re-run the automatic fit (background job "calib")
  POST /api/calib/check                   run the border check (background job "check")
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..paths import DATA_DIR as ROOT
from .processes import ProcessOwner
STATIC = Path(__file__).resolve().parent / "static"


class State:
    """Per-recording review data, frame cache and serialized background jobs."""

    def __init__(self, video: Path, work: Path, calib: str, out: Path | None = None):
        self.calib_name = calib
        self.jobs: dict = {}
        self.processes = ProcessOwner()
        self._threads: list[threading.Thread] = []
        from ..layout import Calibration
        from ..record import from_dict
        self.video = video
        self.name = video.stem
        self.work = work / self.name
        self.out = (out or ROOT / "out") / self.name
        self.labels = ROOT / "labels" / self.name
        (self.labels / "boxes").mkdir(parents=True, exist_ok=True)
        self.cal = Calibration.load(calib, video)
        # a video whose geometry has not been measured has not been converted either, so there is no
        # header and no record yet: the tool still opens, on the Calibrate page, which needs neither
        self.hands = self._read_json(self.work / "hands.json") or []
        rec = self._read_json(self.work / "record.json") or []
        self.games = [from_dict(d) for d in rec]
        self.lock = threading.Lock()
        self._models = None
        self._checks: dict[str, dict] = {}
        self._frame_cache: dict[float, np.ndarray] = {}

    def frame(self, t: float) -> np.ndarray:
        """Seek a normalized BGR frame, reusing recent evidence requests."""
        from .. import video
        t = round(float(t), 2)
        with self.lock:
            if t not in self._frame_cache:
                if len(self._frame_cache) > 40:
                    self._frame_cache.clear()
                self._frame_cache[t] = video.frame_at(self.video, t)
            return self._frame_cache[t]

    def models(self):
        """Load detector/classifier lazily for calibration and label prefill."""
        with self.lock:
            if self._models is None:
                from ..perception.classifier import Classifier
                from ..perception.detector import Detector
                self._models = (Detector(), Classifier())
            return self._models

    # -- calibration (stage 0) ------------------------------------------------------------

    def reload_calib(self):
        """Invalidate rendered evidence after geometry changes."""
        from ..layout import Calibration
        previous = self.cal.data
        self.cal = Calibration.load(self.calib_name, self.video)
        self._frame_cache.clear()
        if previous != self.cal.data:
            self._checks.clear()
            if self.hands:
                self.work.mkdir(parents=True, exist_ok=True)
                (self.work / "calibration.changed").write_text("Run full analysis to refresh observations after changing geometry.\n", encoding="utf-8")
        return self.cal

    def revision(self) -> list:
        """Cheap on-disk change token for review refreshes from another tab or process."""
        paths = [self.work / "hands.json", self.labels / "facts.jsonl", self.labels / "calib.json",
                 self.work / "inputs.changed",
                 *sorted((self.work / "decode").glob("*.json"))]
        return [(str(p), str(p.stat().st_mtime_ns), p.stat().st_size) for p in paths if p.exists()]

    def plate(self):
        """Return the cached median table image used to adjust calibration."""
        from .. import calibfit
        return calibfit.table_plate(self.video, self.work)

    def calib(self) -> dict:
        """Every region as a quad in frame coordinates, so the page can draw and drag them on the plate."""
        from ..layout import CORNERS, box_to_quad
        cal = self.cal
        out = {"video": self.name, "layout": cal.name, "fit": cal.fit, "frame": list(cal.frame), "regions": {}}
        for c in CORNERS:
            r, _k, _s = cal.pond[c]
            out["regions"][f"pond:{c}"] = {"kind": "pond", "corner": c, "quad": box_to_quad(cal.derotation(), r.xyxy),
                                           "rect": [r.x, r.y, r.w, r.h], "movable": False}
            for kind, rect in (("hand", cal.hand[c][0]), ("meld", cal.meld[c][0])):
                out["regions"][f"{kind}:{c}"] = {"kind": kind, "corner": c, "rect": [rect.x, rect.y, rect.w, rect.h],
                                                 "quad": [[rect.x, rect.y], [rect.x + rect.w, rect.y],
                                                          [rect.x + rect.w, rect.y + rect.h], [rect.x, rect.y + rect.h]],
                                                 "roll": cal.roll(c) if kind == "hand" else None, "movable": True}
        u = cal.unit
        out["regions"]["unit"] = {"kind": "unit", "corner": None, "movable": False,
                                  "quad": box_to_quad(cal.derotation(), u.xyxy), "rect": [u.x, u.y, u.w, u.h]}
        out["checks"] = self._checks
        return out

    def save_calib(self, body: dict) -> dict:
        """Write the parts the page changed into labels/<video>/calib.json; the rest of the fit stays."""
        from ..layout import fit_path
        p = fit_path(self.video)
        fit = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"video": self.name, "layout": self.calib_name}
        for part in ("overhead", "hand", "meld", "cam"):
            if part not in body:
                continue
            if part == "overhead":
                fit["overhead"] = {**(fit.get("overhead") or {}), **body["overhead"], "source": "human"}
            else:
                cur = fit.get(part) or {}
                for c, v in body[part].items():
                    cur[c] = {**(cur.get(c) or {}), **v, "source": "human"}
                fit[part] = cur
        fit["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fit, indent=1, ensure_ascii=False), encoding="utf-8")
        self.reload_calib()
        return {"saved": str(p), "fit": fit}

    def start_job(self, key: str, fn, *, hands: list[int] | None = None) -> dict:
        """Run one review operation in the background and expose its failure."""
        with self.lock:
            if self.processes.closing:
                raise ValueError("The app is closing. Restart it to continue.")
            st = self.jobs.get(key)
            if st and st.get("running"):
                return st
            if any(j.get("running") for j in self.jobs.values()):
                raise ValueError("Another review job is running. Wait for it to finish.")
            st = {"key": key, "running": True, "started": time.time(), "done": None, "error": None, "result": None}
            if hands is not None:
                st["hands"] = list(hands)
            self.jobs[key] = st

        def run():
            try:
                st["result"] = fn()
            except Exception as e:  # noqa: BLE001
                st["error"] = str(e)
            st["running"] = False
            st["done"] = time.time()
        thread = threading.Thread(target=run, daemon=True)
        self._threads.append(thread)
        thread.start()
        return st

    def run_calib_fit(self) -> dict:
        """Re-measure this video's geometry. Rectangles a person drew here are marked `human` and kept."""
        result = self.processes.run([sys.executable, "-u", "-m", "video2tenhou.cli", "calib", "fit", str(self.video),
                                     "--calib", self.calib_name, "--work", str(self.work.parent), "--out", str(self.work / "calib")],
                                    env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.reload_calib()
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-2000:])
        return self.cal.fit

    def run_calib_check(self) -> dict[str, dict]:
        """Check that region borders do not cut visible tiles; keep results for the UI."""
        code = ("import json,sys; from pathlib import Path; from video2tenhou.layout import Calibration; "
                "from video2tenhou.calibfit import check_all; from video2tenhou.perception.detector import Detector; "
                "checks=check_all(Path(sys.argv[1]), Calibration.load(sys.argv[2],sys.argv[1]), Detector(), Path(sys.argv[3])); "
                "print(json.dumps({c.region:{'level':c.level,'held':c.held,'cut':c.cut,'note':c.note} for c in checks}))")
        result = self.processes.run([sys.executable, "-c", code, str(self.video), self.calib_name, str(self.work)],
                                    env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-2000:])
        self._checks = json.loads(result.stdout.strip().splitlines()[-1])
        return self._checks

    def decode_path(self, i: int) -> Path:
        """Locate a hand's reconstruction within this recording's work directory."""
        return self.work / "decode" / f"{i:02d}.json"

    def hand_summary(self) -> list[dict]:
        """Combine hand metadata, open questions and facts awaiting reconstruction."""
        out = []
        all_facts = self.all_facts()
        pending = set(self.pending_rebuilds())
        for h in self.hands:
            p = self.decode_path(h["hand"])
            d = self._read_json(p)
            status = None
            if d:
                status = "conflict" if any(i["kind"] == "conflict" for i in d["items"]) else ("review" if d["items"] else "complete")
            decoded_at = p.stat().st_mtime if p.exists() else None
            # facts newer than what the last decode used: a job loads the facts when it starts, not when it ends
            job = self.jobs.get(h["hand"])
            cutoff = decoded_at
            if decoded_at is not None and job and job.get("started") and job.get("done") and job["done"] >= decoded_at - 1:
                cutoff = min(decoded_at, job["started"])
            newer = sum(1 for f in all_facts if f.get("hand") == h["hand"] and cutoff is not None and (f.get("ts") or 0) > cutoff)
            decoded_at = cutoff
            out.append({**h, "status": status, "items": len(d["items"]) if d else None, "t_last": d.get("t_last") if d else None,
                        "decoded_at": decoded_at, "facts_newer": newer, "pending_rebuild": h["hand"] in pending,
                        "job": self.redecode_status(h["hand"]),
                        "play_window": d.get("play_window") if d else None,
                        "score": (d["score"] or {}).get("match") if d else None,
                        "turns": d["stats"]["turns"] if d else None})
        return out

    def all_items(self) -> list[dict]:
        """Project review questions with stable indices and certified confidence.

        Review items can omit margin/coverage fields present in their confidence
        rows. Fill those for navigation without rewriting decoded evidence or
        using a feasible alternative's gap as a confidence certificate.
        """
        out = []
        for h in self.hands:
            p = self.decode_path(h["hand"])
            if not p.exists():
                continue
            d = json.load(open(p, encoding="utf-8"))
            for k, it in enumerate(d["items"]):
                row = dict(it)
                field = "draw" if row.get("kind") == "lost" else row.get("kind")
                if field in ("draw", "discard", "haipai"):
                    turn = -1 if field == "haipai" else row.get("j")
                    confidence = next((c for c in d.get("confidence", [])
                                       if turn is not None and c.get("field") == field and c.get("seat") == row.get("seat")
                                       and c.get("turn") == turn), None)
                    if confidence is not None:
                        row.setdefault("lost", bool(confidence.get("lost")))
                        if row.get("margin") is None:
                            row["margin"] = confidence.get("margin")
                if isinstance(row.get("hand"), list):
                    row["tiles"] = row.pop("hand")          # a reconstructed hand, not the hand number
                row.update({"hand": h["hand"], "idx": k, "kyoku": h["kyoku"], "honba": h["honba"], "game": h["game"]})
                out.append(row)
        return out

    def facts(self, hand: int | None = None) -> list[dict]:
        """Read human facts, optionally restricted to one game's kyoku and honba."""
        p = self.labels / "facts.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
        if hand is None:
            return rows
        e = self.hands[hand]
        return [r for r in rows if r.get("game") == e["game"] and r.get("kyoku") == e["kyoku"] and r.get("honba") == e["honba"]]

    @staticmethod
    def _read_json(p: Path, tries: int = 3):
        """A decode file may be mid-write by a re-decode job: retry briefly."""
        for k in range(tries):
            if not p.exists():
                return None
            try:
                return json.load(open(p, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                if k == tries - 1:
                    return None
                time.sleep(0.2)

    def all_facts(self) -> list[dict]:
        """Read the append-only human evidence file for this recording."""
        p = self.labels / "facts.jsonl"
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []

    def _review_changes(self) -> dict:
        return self._read_json(self.work / "review-changes.json") or {"changes": {}, "rebuilds": {}}

    def _save_review_changes(self, data: dict) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        path = self.work / "review-changes.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        temp.replace(path)

    def _fact_hand(self, fact: dict) -> int | None:
        if all(key in fact for key in ("game", "kyoku", "honba")):
            return next((h["hand"] for h in self.hands if all(
                h[key] == fact[key] for key in ("game", "kyoku", "honba"))), None)
        return next((h["hand"] for h in self.hands if h["hand"] == fact.get("hand")), None)

    def _mark_review_changes(self, hands, changed_at: float) -> None:
        # Caller holds self.lock: additions and deletions share one atomic ledger.
        data = self._review_changes()
        for hand in hands:
            key = str(hand)
            data["changes"][key] = max(changed_at, data["changes"].get(key, 0))
        self._save_review_changes(data)

    def _output_times(self, hand: dict) -> list[int]:
        paths = (self.decode_path(hand["hand"]), self.out / f"g{hand['game']}.json")
        return [p.stat().st_mtime_ns if p.exists() else 0 for p in paths]

    def pending_rebuilds(self) -> list[int]:
        """Return hands with saved corrections absent from reconstructed exports.

        Both the hand decode and its game's JSON must postdate the correction.
        Deletions are retained in a local ledger because the removed fact can
        no longer carry its timestamp. Review rebuild receipts cap freshness at
        job start, including after restart: an edit made during a rebuild must
        survive its later output writes. A later external rebuild supersedes a
        receipt when it replaces that hand's decode and game export.
        """
        with self.lock:
            data = self._review_changes()
            changes = dict(data["changes"])
            for fact in self.all_facts():
                hand = self._fact_hand(fact)
                if hand is not None:
                    key = str(hand)
                    changes[key] = max(changes.get(key, 0), fact.get("ts") or 0)
            pending = []
            for hand in self.hands:
                i = hand["hand"]
                changed = changes.get(str(i), 0)
                if not changed:
                    continue
                times = self._output_times(hand)
                cutoff = min(times) / 1e9
                receipt = data["rebuilds"].get(str(i))
                if receipt and (times[0] == receipt["outputs"][0] or times[1] == receipt["outputs"][1]):
                    cutoff = min(cutoff, receipt["started"] if receipt["success"] else 0)
                for key in (i, "decode_all", "decode_pending"):
                    job = self.jobs.get(key)
                    if job and job.get("running") and ("hands" not in job or i in job["hands"]):
                        cutoff = min(cutoff, job["started"])
                if changed > cutoff:
                    pending.append(i)
            return sorted(pending)

    def add_fact(self, f: dict) -> dict:
        """Append a review answer with game identity, mapped corner and timestamp."""
        e = self.hands[int(f["hand"])]
        row = {**f, "game": e["game"], "kyoku": e["kyoku"], "honba": e["honba"], "hand": e["hand"], "author": "tool", "ts": time.time()}
        if row.get("seat") and "corner" not in row:
            corner = next((c for c, s in e["corner_wind"].items() if s == row["seat"]), None)
            if corner is None:
                raise ValueError(f"seat {row['seat']} is not at this table")
            row["corner"] = corner
        with self.lock:
            row["ts"] = time.time()
            self._mark_review_changes([e["hand"]], row["ts"])
            with open(self.labels / "facts.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def delete_fact(self, ts: float) -> int:
        """Remove an answer by its timestamp; the next rebuild reopens its question."""
        p = self.labels / "facts.jsonl"
        with self.lock:
            rows = [l for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
            keep = [l for l in rows if abs(float(json.loads(l).get("ts") or -1) - ts) > 1e-6]
            removed = [self._fact_hand(json.loads(line)) for line in rows if line not in keep]
            if removed:
                self._mark_review_changes([i for i in removed if i is not None], time.time())
            p.write_text("".join(keep) + ("\n" if keep and not keep[-1].endswith("\n") else ""), encoding="utf-8")
        return len(rows) - len(keep)

    def save_label(self, lab: dict) -> Path:
        """Store region-space tile boxes as frame-coordinate training labels."""
        from ..layout import box_to_quad
        from ..train.data import region_upright
        kind, corner, t = lab["kind"], lab["corner"], float(lab["t"])
        _, M = region_upright(self.frame(t), self.cal, kind, corner)
        boxes = [{"quad": box_to_quad(M, b["xyxy"]), "tile": b.get("tile", "?"), "sideways": bool(b.get("sideways")), "role": b.get("role", "tile")}
                 for b in lab["boxes"]]
        out = {"video": self.name, "t": t, "kind": kind, "corner": corner, "boxes": boxes, "source": "tool"}
        for k in ("tiles", "rows", "melds"):
            if k in lab:
                out[k] = lab[k]
        p = self.labels / "boxes" / f"{kind}_{corner}_{int(round(t))}.json"
        if p.exists():
            old = json.load(open(p, encoding="utf-8"))
            bak = p.with_suffix(".orig.json")
            if old.get("source") != "tool" and not bak.exists():
                bak.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")   # keep the human label once
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return p

    def render(self, t: float, region: str, scale: float = 1.0) -> bytes:
        """Render a frame or calibrated upright region as JPEG evidence."""
        if region not in ["frame", *self.cal.regions()] or not 0.05 <= scale <= 4.0:
            raise ValueError("Unknown region or unsupported image scale.")
        from ..train.data import region_upright
        frame = self.frame(t)
        if region == "frame":
            img = frame
        else:
            kind, _, corner = region.partition(":")
            img, _ = region_upright(frame, self.cal, kind, corner)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes()

    def read(self, t: float, region: str) -> dict:
        """Prefill editable boxes with model predictions and region coordinates."""
        from ..perception.reader import read_region
        from ..train.data import region_upright
        det, clf = self.models()
        kind, _, corner = region.partition(":")
        frame = self.frame(t)
        img, _ = region_upright(frame, self.cal, kind, corner)
        rd = read_region(frame, self.cal, region, det, clf, t=t, img=img)
        return {"size": list(rd.size), "boxes": [{"xyxy": [round(v, 1) for v in b.xyxy], "conf": round(b.conf, 3), "sideways": b.sideways,
                                                   "role": b.role, "tile": clf.classes[int(np.argmax(b.p))], "p": round(float(np.max(b.p)), 3),
                                                   "row": b.row, "col": b.col, "group": b.group} for b in rd.boxes]}

    def clip(self, t0: float, t1: float, region: str) -> Path:
        """An mp4 of one camera (or the whole frame) between t0 and t1, cached under work/<video>/clips."""
        if region not in ["frame", *self.cal.regions()]:
            raise ValueError("Unknown video region.")
        import subprocess
        import hashlib
        d = self.work / "clips"
        d.mkdir(parents=True, exist_ok=True)
        t0, t1 = max(0.0, t0), max(t0 + 1.0, t1)
        from ..cache import source_identity
        geometry = hashlib.sha256(json.dumps([self.cal.data, source_identity(self.video)], sort_keys=True).encode()).hexdigest()[:12]
        p = d / f"{geometry}_{region.replace(':', '_')}_{t0:.1f}_{t1:.1f}.mp4"
        if p.exists():
            return p
        kind, _, corner = region.partition(":")
        # the same geometry as the still images (layout.Calibration.transform): de-rotate the table, cut the
        # region, turn it upright, scale. ffmpeg's rotate turns clockwise, OpenCV's counter-clockwise.
        import math
        if kind in ("hand", "meld", "cam"):
            rect, scale = (self.cal.hand[corner] if kind == "hand" else
                           self.cal.meld[corner] if kind == "meld" else (self.cal.cam[corner], 1.0))
            vf = f"crop={rect.w}:{rect.h}:{rect.x}:{rect.y}"
            if scale and abs(scale - 1.0) > 1e-6:
                vf += f",scale=iw*{scale:g}:ih*{scale:g}"
            roll = self.cal.roll(corner) if kind == "hand" else 0.0
            if abs(roll) > 0.5:
                vf += f",rotate={-roll * math.pi / 180:.5f}:ow=rotw({-roll * math.pi / 180:.5f}):oh=roth({-roll * math.pi / 180:.5f}):c=black"
        elif kind in ("pond", "overhead"):
            cx, cy, side = self.cal.center[0], self.cal.center[1], self.cal.side
            vf = (f"rotate={-self.cal.angle * math.pi / 180:.5f}:ow=iw:oh=ih:c=black,"
                  f"crop={side}:{side}:{cx - side / 2:.0f}:{cy - side / 2:.0f}")
            if kind == "pond":
                rect, k, scale = self.cal.pond[corner]
                vf += f",crop={rect.w}:{rect.h}:{rect.x}:{rect.y}" + ",transpose=1" * (k % 4)
                if scale and abs(scale - 1.0) > 1e-6:
                    vf += f",scale=iw*{scale:g}:ih*{scale:g}"
        else:
            vf = "scale=960:-2"
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t0:.2f}", "-i", str(self.video), "-t", f"{t1 - t0:.2f}",
               "-vf", vf + ",scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", "10", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(p)]
        self.processes.run(cmd, check=True, capture_output=True)
        return p

    def context(self, hand: int, seat: str, t: float) -> dict:
        """What the camera read around time t (the seat's hand observations before and after) and what the
        reconstruction assumes (hand before / after the turn at t)."""
        from ..observe import load_obs
        e = self.hands[hand]
        corner = next(c for c, s in e["corner_wind"].items() if s == seat)
        obs = load_obs(self.work, hand).get(f"hand:{corner}", [])
        def brief(o):
            return {"t0": o["t0"], "t1": o["t1"], "count": o["count"], "n": o["n_used"], "partial": bool(o.get("partial")),
                    "tiles": [{"tile": s["tile"], "conf": s["conf"], "disagree": s["disagree"]} for s in o["slots"]]}
        before = [brief(o) for o in obs if o["t1"] <= t + 0.5][-2:]
        after = [brief(o) for o in obs if o["t0"] >= t - 0.5][:2]
        p = self.decode_path(hand)
        turn = None
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            turn = next((x for x in d["turns"] if x["seat"] == seat and abs(x["t"] - t) < 1.5), None)
        return {"corner": corner, "read_before": before, "read_after": after, "turn": turn}

    def start_redecode(self, i: int) -> dict:
        """Start the re-decode of hand i in a background thread (fresh process); returns its status."""
        with self.lock:
            if self.processes.closing:
                raise ValueError("The app is closing. Restart it to continue.")
            st = self.jobs.get(i)
            if st and st.get("running"):
                return st
            if any(j.get("running") for j in self.jobs.values()):
                raise ValueError("Another review job is running. Wait for it to finish.")
            st = {"hand": i, "running": True, "started": time.time(), "done": None, "error": None}
            self.jobs[i] = st

        def work():
            try:
                self.redecode(i)
            except Exception as e:  # noqa: BLE001
                st["error"] = str(e)
            st["running"] = False
            st["done"] = time.time()
        thread = threading.Thread(target=work, daemon=True)
        self._threads.append(thread)
        thread.start()
        return st

    def redecode_status(self, i: int) -> dict:
        """Return a hand job's current state, including its last error."""
        return self.jobs.get(i) or {"hand": i, "running": False, "started": None, "done": None, "error": None}

    def redecode(self, i: int) -> dict:
        """Reconstruct one hand and refresh the complete project's exports."""
        self._run_decode(str(i))
        p = self.decode_path(i)
        return json.load(open(p, encoding="utf-8")) if p.exists() else {}

    def start_redecode_all(self) -> dict:
        """Re-decode every hand with the facts saved so far, in the background (one fresh process)."""
        return self.start_job("decode_all", lambda: self._run_decode("all"))

    def start_redecode_pending(self) -> dict:
        """Rebuild the server's pending corrections in one child, without client-selected IDs.

        The job retains its selected hands for progress and freshness receipts.
        A no-change request is a no-op; completed hands are never forced merely
        because another hand has a new or deleted answer.
        """
        selected = self.pending_rebuilds()
        if not selected:
            return self.redecode_pending_status()
        self.start_job("decode_pending", lambda: self._run_decode(",".join(map(str, selected)),
                       job_key="decode_pending"), hands=selected)
        return self.redecode_pending_status()

    def redecode_pending_status(self) -> dict:
        """Report the pending-only job and corrections still absent from exports."""
        st = dict(self.jobs.get("decode_pending") or {"key": "decode_pending", "running": False,
                  "started": None, "done": None, "error": None, "hands": []})
        selected = st.get("hands", [])
        done = sum(1 for i in selected if st.get("started") and self.decode_path(i).exists()
                   and self.decode_path(i).stat().st_mtime >= st["started"])
        return {**st, "hands_done": done, "hands_total": len(selected), "pending": self.pending_rebuilds()}

    def redecode_all_status(self) -> dict:
        """The job's status, with how many hands were decoded since it started (the decode files it rewrote)."""
        st = dict(self.jobs.get("decode_all") or {"key": "decode_all", "running": False, "started": None, "done": None, "error": None})
        done = 0
        if st.get("started"):
            done = sum(1 for h in self.hands if self.decode_path(h["hand"]).exists()
                       and self.decode_path(h["hand"]).stat().st_mtime >= st["started"])
        return {**st, "hands_done": done, "hands_total": len(self.hands)}

    def _run_decode(self, which: str, *, job_key=None) -> None:
        """Decode selected comma-separated hand IDs or "all" in a fresh process, so engine code on disk runs
        (the server may have been up for hours)."""
        import subprocess
        selected = None if which == "all" else {int(value) for value in which.split(",")}
        if selected is not None and not selected <= {h["hand"] for h in self.hands}:
            raise ValueError("Unknown hand selected for rebuilding.")
        if any((self.work / marker).exists() for marker in ("calibration.changed", "inputs.changed")):
            raise ValueError("The table geometry or project settings changed. Close this view and choose Analyze recording to refresh readings and alignment before rebuilding logs.")
        code = ("import json, sys; from pathlib import Path; from video2tenhou.record import from_dict; "
                "from video2tenhou.layout import Calibration; from video2tenhou.engine.decode import run_decode; "
                "work = Path(sys.argv[1]); hands = json.load(open(work / 'hands.json', encoding='utf-8')); "
                "games = [from_dict(d) for d in json.load(open(work / 'record.json', encoding='utf-8'))]; "
                "only = None if sys.argv[2] == 'all' else {int(value) for value in sys.argv[2].split(',')}; "
                "run_decode(work, hands, games, force=True, only=only, log=lambda *a: None, "
                "video_path=Path(sys.argv[3]), cal=Calibration.load(sys.argv[4], sys.argv[3])); "
                "from video2tenhou.cli import write_outputs; "
                "decodes = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((work / 'decode').glob('*.json'))]; "
                "write_outputs(Path(sys.argv[5]), games, decodes, hands, Path(sys.argv[3]).stem)")
        with self.lock:
            started = time.time()
            key = job_key if job_key is not None else ("decode_all" if which == "all" else int(which))
            job = self.jobs.get(key)
            if job and job.get("running"):
                started = min(started, job["started"])
        success = False
        try:
            r = self.processes.run([sys.executable, "-c", code, str(self.work), which, str(self.video), self.calib_name, str(self.out)],
                               env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError("re-decode failed: " + (r.stderr or "")[-800:])
            success = True
        finally:
            with self.lock:
                data = self._review_changes()
                for hand in self.hands:
                    if selected is None or hand["hand"] in selected:
                        data["rebuilds"][str(hand["hand"])] = {
                            "started": started, "success": success, "outputs": self._output_times(hand)}
                self._save_review_changes(data)

    def close(self) -> None:
        """Stop review/calibration/video child processes before server shutdown."""
        self.processes.shutdown()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)


STATE: State


class Handler(SimpleHTTPRequestHandler):
    """Review HTTP routes shared by the standalone and project-scoped servers."""

    @property
    def state(self):
        """Resolve review state; the studio overrides this per request."""
        return STATE

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def end_headers(self):
        """Avoid stale job responses and evidence after review edits."""
        self.send_header("Cache-Control", "no-store")   # the page and images change between runs
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter: no line per API call
        """Keep periodic API polling out of the terminal access log."""
        if args and "/api/" in str(args[0]):
            return
        super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        if code >= 400 and self.command == "POST":
            self._drain_rejected_body()
        from ..engine.decode import sanitize
        data = json.dumps(sanitize(obj), ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_request_bytes(self, size: int) -> bytes:
        """Account for consumed bytes so an error never reads the body twice."""
        data = self.rfile.read(size)
        self._request_bytes_read = getattr(self, "_request_bytes_read", 0) + len(data)
        return data

    def _drain_rejected_body(self) -> None:
        """Finish small rejected POST bodies before closing the TCP connection.

        Closing with unread payload can turn a useful 4xx response into a
        connection reset on Windows. Never wait indefinitely for a dishonest
        Content-Length or drain an entire rejected video upload: both bytes
        and waiting time are bounded, independently of endpoint body limits.
        """
        self.close_connection = True
        try:
            remaining = int(self.headers.get("Content-Length", "0")) - getattr(self, "_request_bytes_read", 0)
        except ValueError:
            return
        if not 0 < remaining <= 1024 * 1024:
            return
        previous_timeout = self.connection.gettimeout()
        try:
            deadline = time.monotonic() + 0.5
            while remaining > 0:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    break
                self.connection.settimeout(min(previous_timeout or wait, wait))
                chunk = self.rfile.read1(min(65536, remaining))
                self._request_bytes_read = getattr(self, "_request_bytes_read", 0) + len(chunk)
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, TimeoutError):
            pass  # The response is still attempted; this connection is closed.
        finally:
            self.connection.settimeout(previous_timeout)

    def do_GET(self):
        """Serve read-only review data and generated evidence for the bound recording."""
        if not self._host_ok():
            return self._json({"error": "Use the local app address."}, 403)
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/revision":
                return self._json(self.state.revision())
            if u.path == "/api/hands":
                return self._json(self.state.hand_summary())
            if u.path == "/api/decode_all":
                return self._json(self.state.redecode_all_status())
            if u.path == "/api/decode_pending":
                return self._json(self.state.redecode_pending_status())
            if u.path.startswith("/api/decode/"):
                return self._json(self.state.redecode_status(int(u.path.rsplit("/", 1)[1])))
            if u.path == "/api/items":
                return self._json(self.state.all_items())
            if u.path.startswith("/api/hand/"):
                i = int(u.path.rsplit("/", 1)[1])
                d = self.state._read_json(self.state.decode_path(i))
                return self._json({"entry": self.state.hands[i], "decode": d, "facts": self.state.facts(i)})
            if u.path == "/api/frame":
                data = self.state.render(float(q["t"]), q.get("region", "frame"), float(q.get("scale", "1")))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path == "/api/read":
                return self._json(self.state.read(float(q["t"]), q["region"]))
            if u.path == "/api/context":
                return self._json(self.state.context(int(q["hand"]), q["seat"], float(q["t"])))
            if u.path == "/api/clip":
                p = self.state.clip(float(q["t0"]), float(q["t1"]), q.get("region", "frame"))
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path == "/api/facts":
                return self._json(self.state.facts(int(q["hand"]) if "hand" in q else None))
            if u.path == "/api/calib":
                return self._json(self.state.calib())
            if u.path == "/api/calib/job":
                return self._json(self.state.jobs.get(q.get("key", "calib")) or {"running": False})
            if u.path == "/api/plate":
                img = self.state.plate()
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                data = buf.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)
        if u.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        """Accept same-origin JSON review edits and background actions."""
        u = urlparse(self.path)
        try:
            if not self._local_request():
                return self._json({"error": "Only requests from this local app are accepted."}, 403)
            n = int(self.headers.get("Content-Length", "0"))
            if n > 4 * 1024 * 1024 or n < 0:
                return self._json({"error": "Request body is too large."}, 413)
            body = json.loads(self._read_request_bytes(n) or b"{}")
            if u.path == "/api/facts":
                return self._json(self.state.add_fact(body))
            if u.path == "/api/facts/delete":
                return self._json({"deleted": self.state.delete_fact(float(body["ts"]))})
            if u.path == "/api/label":
                return self._json({"saved": str(self.state.save_label(body))})
            if u.path == "/api/decode_all":
                return self._json(self.state.start_redecode_all())
            if u.path == "/api/decode_pending":
                return self._json(self.state.start_redecode_pending())
            if u.path.startswith("/api/decode/"):
                return self._json(self.state.start_redecode(int(u.path.rsplit("/", 1)[1])))
            if u.path == "/api/calib":
                return self._json(self.state.save_calib(body))
            if u.path == "/api/calib/fit":
                return self._json(self.state.start_job("calib", self.state.run_calib_fit))
            if u.path == "/api/calib/check":
                return self._json(self.state.start_job("check", self.state.run_calib_check))
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)
        return self._json({"error": "unknown endpoint"}, 404)

    def _local_request(self):
        """Reject browser cross-origin writes and DNS-rebinding hosts."""
        host = self.headers.get("Host", "")
        if not self._host_ok():
            return False
        origin = self.headers.get("Origin")
        return (origin is None or origin == f"http://{host}") and self.headers.get("X-Video2Tenhou") == "1"

    def _host_ok(self) -> bool:
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")


class _Server(ThreadingHTTPServer):
    allow_reuse_address = False      # a second instance on the same port would split requests; fail instead

    def server_close(self):
        """Shut down owned jobs before releasing the local listener."""
        if hasattr(self, "workspace"):
            self.workspace.close()
        if hasattr(self, "review_state"):
            self.review_state.close()
        super().server_close()


def serve(video: Path, work: Path, calib: str = "pml", port: int = 8765):
    """Serve the single-recording review page on loopback only."""
    global STATE
    STATE = State(video, work, calib)
    try:
        httpd = _Server(("127.0.0.1", port), Handler)
        httpd.review_state = STATE
    except OSError as e:
        raise SystemExit(f"port {port} is already in use (another review server is running): {e}")
    print(f"review tool: http://localhost:{port}  ({video})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
