"""Header stage: overlay scan -> hand windows -> alignment with the site record.

    scan()     one sequential pass over the video reading the overlay at `fps`
    segment()  debounced hand windows keyed by (kyoku, honba), split into hanchan
    align()    hands matched one-to-one with the site record; corner -> seat map;
               every disagreement is a hard error (wrong game id or wrong video)

Hand entries use current-hand winds E/S/W/N in ``corner_wind`` and ``scores``.
Corners identify fixed chairs throughout the broadcast. The site's player list
and score trajectory use starting winds (E is the first hand's dealer), mapped
by ``corner_site``: starting seat = (current wind index + kyoku) mod 4.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


from . import video
from .cache import source_identity
from .layout import CORNERS, Calibration
from .overlay import read_overlay
from .record import Game, SEAT_LETTER

WINDS = "ESWN"
SEATS = tuple(WINDS)
TOTAL_POINTS = 100_000


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def reading_from_state(t: float, st) -> dict:
    """Create the header cache row at seconds t, retaining unknown overlay fields as None."""
    return {
        "t": round(float(t), 3),
        "kyoku": st.kyoku_index,
        "honba": st.honba,
        "sticks": st.riichi_sticks,
        "scores": {c: st.corners[c].score for c in CORNERS},
        "winds": {c: st.corners[c].wind for c in CORNERS},
    }


def valid(r: dict) -> bool:
    """Require all winds and a 100,000-point total including riichi sticks before segmentation."""
    if r["kyoku"] is None or r["honba"] is None or r["sticks"] is None:
        return False
    sc, wd = r["scores"], r["winds"]
    if any(sc[c] is None for c in CORNERS) or any(wd[c] is None for c in CORNERS):
        return False
    if len(set(wd.values())) != 4:
        return False
    return sum(sc.values()) + 1000 * r["sticks"] == TOTAL_POINTS


def scan(path: str | Path, cal: Calibration, fps: float = 1.0, start: float = 0.0,
         end: Optional[float] = None) -> Iterator[dict]:
    """Yield overlay rows at the requested rate, with a checksum validity flag.

    Incomplete rows stop recognition early and retain unknown fields. They never
    participate in segmentation; valid rows preserve every observed score, wind
    and stick change rather than filling missing pixels from the site record.
    """
    for t, frame in video.sample(path, fps=fps, start=start, end=end):
        r = reading_from_state(t, read_overlay(frame, names=False, cal=cal, complete_only=True))
        r["ok"] = valid(r)
        yield r


def read_nicks(path: str | Path, t: float, cal: Calibration | None = None) -> dict[str, str]:
    """Nick per corner ('Zem M. (zemzem)' -> 'zemzem'), read with tesseract at one frame."""
    st = read_overlay(video.frame_at(path, t), names=True, cal=cal)
    out = {}
    for c in CORNERS:
        m = re.search(r"\(([^)]+)\)", st.corners[c].name or "")
        out[c] = (m.group(1) if m else st.corners[c].name or "").strip()
    return out


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

@dataclass
class Hand:
    """One stable overlay segment; t_read may extend earlier to cover delayed overlay updates."""
    game: int                      # hanchan index within the video
    kyoku: int
    honba: int
    t_start: float
    t_end: float
    sticks: int                    # riichi sticks on the table at the start
    scores: dict                   # corner -> score at the start
    winds: dict                    # corner -> seat wind shown during the hand
    stick_events: list = field(default_factory=list)   # [{t, sticks}] changes during the hand
    n: int = 0                     # valid readings
    t_read: Optional[tuple] = None                     # (t0, t1) window to read: starts before the overlay switch

    @property
    def key(self):
        """Round/honba identity within a hanchan, excluding the video game index."""
        return self.kyoku, self.honba


def _mode(values):
    return Counter(values).most_common(1)[0][0]



OVERLAY_LAG = 60.0        # seconds: how long before its overlay segment a hand's play may have started

def segment(readings: list[dict], *, min_run: int = 5, min_duration: float = 30.0) -> list[Hand]:
    """Debounced hand windows: a new (kyoku, honba) key is accepted after `min_run`
    consecutive valid readings; runs shorter than `min_duration` are dropped."""
    runs: list[list[dict]] = []
    cur: list[dict] = []
    pending: list[dict] = []
    for r in readings:
        if not r.get("ok", valid(r)):
            continue
        k = (r["kyoku"], r["honba"])
        if cur and k == (cur[-1]["kyoku"], cur[-1]["honba"]):
            cur.append(r)
            pending = []
            continue
        if pending and k == (pending[-1]["kyoku"], pending[-1]["honba"]):
            pending.append(r)
        else:
            pending = [r]
        if len(pending) >= min_run or not cur:
            if cur:
                runs.append(cur)
            cur, pending = pending, []
    if cur:
        runs.append(cur)

    hands: list[Hand] = []
    game = 0
    for run in runs:
        t0, t1 = run[0]["t"], run[-1]["t"]
        if t1 - t0 < min_duration:
            continue
        if hands and (run[0]["kyoku"], run[0]["honba"]) <= hands[-1].key:
            game += 1
        head = run[: max(3, min(10, len(run) // 4))]
        scores = {c: _mode([r["scores"][c] for r in head]) for c in CORNERS}
        winds = {c: _mode([r["winds"][c] for r in run]) for c in CORNERS}
        sticks0 = _mode([r["sticks"] for r in head])
        events, last = [], sticks0
        stable: list[dict] = []
        for r in run:
            if stable and r["sticks"] == stable[-1]["sticks"]:
                stable.append(r)
            else:
                stable = [r]
            if len(stable) >= 3 and stable[0]["sticks"] != last:
                last = stable[0]["sticks"]
                events.append({"t": stable[0]["t"], "sticks": last})
        hands.append(Hand(game, run[0]["kyoku"], run[0]["honba"], t0, t1, sticks0, scores, winds, events, len(run)))
    return hands


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------

def corner_wind(winds: dict) -> dict[str, str]:
    """corner -> the seat wind of this hand (what the overlay shows at that corner; E deals).

    A player is identified by the corner: the chairs and the cameras do not move. The wind is the role in
    this hand, so it is what the player is called. Nothing is ever named by the wind a player started in."""
    return dict(winds)


def corner_site(winds: dict, kyoku: int) -> dict[str, str]:
    """corner -> the seat name the site record uses (EAST..NORTH), which is fixed for the hanchan: the wind
    that corner held in its first hand. The site keys its player list and its per-hand deltas by it, and the
    tenhou log indexes players the same way; nothing else in the program uses it."""
    return {c: SEAT_NAME[WINDS[(WINDS.index(w) + kyoku) % 4]] for c, w in winds.items()}


def trajectory(game: Game) -> list[dict]:
    """Starting-seat scores before each hand, accumulated from the authoritative result deltas."""
    s = {k: 25000 for k in SEATS}
    out = [dict(s)]
    for h in game.hands:
        s = {k: s[k] + h.deltas[SEAT_NAME[k]] for k in SEATS}
        out.append(dict(s))
    return out


SEAT_NAME = {v: k for k, v in SEAT_LETTER.items()}   # 'E' -> 'EAST'


def align(hands: list[Hand], games: list[Game], nicks: Optional[dict[int, dict[str, str]]] = None) -> tuple[list[dict], list[str]]:
    """Match the segmented hands to the site games (one per hanchan, in order).

    Returns (hand entries for hands.json, problems). Any problem means the
    video and the record do not describe the same games.
    """
    problems: list[str] = []
    entries: list[dict] = []
    by_game: dict[int, list[Hand]] = {}
    for h in hands:
        by_game.setdefault(h.game, []).append(h)
    if len(by_game) != len(games):
        problems.append(f"video has {len(by_game)} hanchan, {len(games)} game ids given")
    for gi, game in enumerate(games):
        hs = by_game.get(gi, [])
        traj = trajectory(game)
        if len(hs) != len(game.hands):
            problems.append(f"hanchan {gi}: {len(hs)} hands in the video, {len(game.hands)} on the site")
        seat_map = None
        for j, h in enumerate(hs):
            site = game.hands[j] if j < len(game.hands) else None
            cw, cs = corner_wind(h.winds), corner_site(h.winds, h.kyoku)
            if seat_map is None:
                seat_map = cs
            elif cs != seat_map:
                problems.append(f"hanchan {gi} hand {j}: corner->site seat {cs} differs from {seat_map}")
            scores = {cw[c]: v for c, v in h.scores.items()}
            entry = {
                "hand": len(entries), "game": gi, "game_id": game.id, "kyoku": h.kyoku, "honba": h.honba,
                "sticks": h.sticks, "t_start": (h.t_read or (h.t_start, h.t_end))[0], "t_end": (h.t_read or (h.t_start, h.t_end))[1],
                "t_overlay": [h.t_start, h.t_end], "corner_wind": cw, "corner_site": cs,
                "scores": scores, "stick_events": h.stick_events, "site_index": j if site else None,
            }
            if site is not None:
                if (site.kyoku, site.honba) != h.key:
                    problems.append(f"hanchan {gi} hand {j}: video {h.kyoku}/{h.honba} vs site {site.kyoku}/{site.honba}")
                if site.sticks != h.sticks:
                    problems.append(f"hanchan {gi} hand {j}: sticks {h.sticks} vs site {site.sticks}")
                by_site = {cs[c]: v for c, v in h.scores.items()}
                want = {SEAT_NAME[k]: v for k, v in traj[j].items()}
                if by_site != want:
                    problems.append(f"hanchan {gi} hand {j}: scores {by_site} vs site {want}")
            entries.append(entry)
        if nicks and gi in nicks and seat_map:
            for c, nick in nicks[gi].items():
                want = game.players.get(seat_map[c])
                if nick.lower() != (want or "").lower():
                    problems.append(f"hanchan {gi}: corner {c} shows nick {nick!r}, the site's {seat_map[c]} is {want!r}")
            for e in entries:
                if e["game"] == gi:
                    e["nicks"] = nicks[gi]
    return entries, problems


# ---------------------------------------------------------------------------
# stage driver
# ---------------------------------------------------------------------------

def _overlay_signature(path: str | Path, cal: Calibration, fps: float) -> dict:
    """Inputs affecting scanned header pixels; table-camera fits are independent."""
    source = Path(path).resolve()
    stat = source.stat()
    return {"version": 3, "fps": float(fps), "frame": list(cal.frame),
            "source": {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "sha256": source_identity(source, refresh=True)},
            "strips": {corner: list(rect.xyxy) for corner, rect in cal.strip.items()},
            "round": {name: list(getattr(cal, name).xyxy)
                      for name in ("round_wind", "round_num", "honba", "sticks")}}


def _cached_overlay(ov: Path, manifest: Path, signature: dict) -> Optional[list[dict]]:
    """Accept only a completed, matching scan, including its exact file digest."""
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return None
        content = ov.read_bytes()
        if hashlib.sha256(content).hexdigest() != metadata.get("sha256"):
            return None
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        return rows if len(rows) == metadata.get("readings") else None
    except (OSError, ValueError, AttributeError):
        return None


def _write_overlay(path: str | Path, cal: Calibration, fps: float, ov: Path,
                   manifest: Path, signature: dict, log) -> list[dict]:
    """Publish a completed scan; an interrupted refresh leaves the old pair intact."""
    pending: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=ov.parent, prefix=".overlay-", suffix=".tmp", delete=False) as output:
            temp_overlay = Path(output.name)
            pending.append(temp_overlay)
            digest = hashlib.sha256()
            readings = []
            for row in scan(path, cal, fps=fps):
                line = (json.dumps(row) + "\n").encode("utf-8")
                output.write(line)
                digest.update(line)
                readings.append(row)
                if len(readings) % 600 == 0:
                    log(f"  overlay: {row['t']:.0f} s")
            output.flush()
            os.fsync(output.fileno())
        metadata = {"signature": signature, "readings": len(readings), "sha256": digest.hexdigest()}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ov.parent,
                                         prefix=".overlay-meta-", suffix=".tmp", delete=False) as output:
            temp_manifest = Path(output.name)
            pending.append(temp_manifest)
            json.dump(metadata, output, indent=1)
            output.flush()
            os.fsync(output.fileno())
        temp_overlay.replace(ov)
        # A crash between replacements cannot authenticate different data with
        # the old manifest. The next attempt will repeat that incomplete commit.
        temp_manifest.replace(manifest)
        return readings
    finally:
        for temporary in pending:
            temporary.unlink(missing_ok=True)


def run_scan(path: str | Path, cal: Calibration, work: Path, *, fps: float = 1.0,
             force: bool = False, log=print) -> list[dict]:
    """Reuse or build hash-verified numeric overlay rows, without site alignment.

    Calibration and conversion share this scan so first-use border checks sample
    the same play windows as later checks. Failed scans never publish a cache.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Header sampling fps must be finite and positive")
    work.mkdir(parents=True, exist_ok=True)
    ov = work / "overlay.jsonl"
    manifest = work / "overlay.meta.json"
    signature = _overlay_signature(path, cal, fps)
    readings = None if force else _cached_overlay(ov, manifest, signature)
    if readings is None:
        if ov.exists() and not force:
            log("  overlay: source, layout or sampling cache changed; scanning again")
        readings = _write_overlay(path, cal, fps, ov, manifest, signature, log)
    return readings


def play_hands(readings: list[dict]) -> list[Hand]:
    """Segment valid overlay rows and extend windows for broadcast overlay delay."""
    hands = segment(readings)
    # the overlay switches 15-30 s after the table moved on: a hand's first discards fall inside the previous
    # overlay segment. Each read window starts OVERLAY_LAG before the switch; the physical window is found in
    # the ponds (engine.ponds.play_window, anchored on the overlay segment).
    for i, h in enumerate(hands):
        lo = hands[i - 1].t_start + 30.0 if i > 0 and hands[i - 1].game == h.game else 0.0
        h.t_read = (max(lo, h.t_start - OVERLAY_LAG), h.t_end)
    return hands


def run_header(path: str | Path, cal: Calibration, games: list[Game], work: Path, *, fps: float = 1.0,
               force: bool = False, log=print) -> tuple[list[dict], list[str]]:
    """Build play windows from a verified scan and align them with the site record.

    Returns ``(entries, alignment problems)``. Player names and site alignment
    are evaluated on every call, independently of cached numeric overlay rows.
    """
    hands = play_hands(run_scan(path, cal, work, fps=fps, force=force, log=log))
    nicks: dict[int, dict[str, str]] = {}
    for gi in sorted({h.game for h in hands}):
        first = next(h for h in hands if h.game == gi)
        nicks[gi] = read_nicks(path, (first.t_start + first.t_end) / 2, cal=cal)
    entries, problems = align(hands, games, nicks)
    json.dump(entries, open(work / "hands.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return entries, problems
