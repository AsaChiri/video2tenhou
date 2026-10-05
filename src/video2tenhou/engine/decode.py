# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 5: reconstruct one hand and write its decode artifact (DESIGN.md 4.8).

The stages run in order, each a function of the hand and the results before it:
events.py (discards, calls, the dead wall, the turn sequence, riichi), reconstruct.py
(haipai, draws and call tiles as one constraint program, every kan's indicator),
score_reconcile.py (the site's result as a constraint, then the score check), and
the questions that remain with the confidence of every decision. What the reviewer
sees beyond the reconstruction is collected in one report (questions.py).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING

from video2tenhou.files import atomic_write_json, json_digest, sanitize
from video2tenhou.layout import Calibration
from video2tenhou.observe import load_obs, provenance, validate_observation_cache
from video2tenhou.paths import DATA_DIR
from video2tenhou.perception import evidence_policy as retention
from video2tenhou.read import ReadContext, validate_read_cache

from . import pond_evidence, questions
from .confidence import low_margin
from .dense import DenseContext
from .events import (
    Hand,
    Ponds,
    Riichi,
    TurnSequence,
    acquire_replacements,
    anchored_calls,
    apply_meld_facts,
    dead_wall,
    hand_of,
    place_riichi,
    pond_logs,
    turn_sequence,
)
from .hand import site_seat
from .questions import Report, ranked, uncertain_discards, uncertain_tiles
from .reconstruct import (
    Search,
    build_model,
    fit_evidence,
    kan_indicators,
    unnamed_kans,
)
from .review import (
    ReviewContext,
    confidence_rows,
    decode_context,
    facts_for_hand,
    load_facts,
    turn_key,
    unseen_draw,
)
from .score_reconcile import Win, check_score, check_tenpai_and_ura, reconcile_score
from .validation import review_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from video2tenhou.perception.evidence_policy import EvidencePolicy
    from video2tenhou.read import ReadModels
    from video2tenhou.record import Game, HandResult

    from .solver import HandModel, Solution

# bump when reconstruction semantics or the artifact change; old results are rebuilt
DECODER_VERSION = 20
LOST_FACT_WINDOW = 3.0  # s between a Can't-tell answer and the discard of its draw
LOGGER = logging.getLogger("video2tenhou.engine.decode")


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeOptions:
    """Solver budgets and optional targeted-reading inputs for one hand.

    ``time_limit`` caps each main solve and ``confidence_timeout`` each
    certification pass; ``workers`` is the CP-SAT parallelism of both.
    """

    time_limit: float = 60.0
    confidence_timeout: float = 60.0
    workers: int = 8
    models: ReadModels | None = None
    work_dir: Path | None = None


def decode_hand(
    entry: dict,
    obs: dict[str, list[dict]],
    result: HandResult,
    facts: dict | None = None,
    *,
    options: DecodeOptions | None = None,
) -> dict:
    """Reconstruct one hand from observations and its mandatory site result.

    Facts are the normalized output of ``facts_for_hand``. Supplying ``options.models``
    as (detector, classifier, video path, calibration) enables targeted rereads;
    otherwise the returned artifact exposes uncertainty from existing evidence.
    This function does not write artifacts or alter human facts.
    ``options.confidence_timeout`` bounds each certification pass: one of the first
    fit when targeted rereads are enabled, and one of the final reconstruction.
    """
    decoded = _decode(entry, obs, result, facts or {}, options or DecodeOptions())
    return review_artifact(decoded, entry)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Reconstruction:
    """A hand's events and final solution: what its decode artifact is written from."""

    hand: Hand
    ponds: Ponds
    seq: TurnSequence
    riichi: Riichi
    search: Search
    sol: Solution
    repaired: dict[tuple[str, int], str]  # (seat, j) -> the kind a repair read
    dora: list[str]
    indicators: list[dict]
    score: dict | None
    lost: set[tuple[str, int]]  # the draws the reviewer could not tell


def _decode(
    entry: dict,
    obs: dict[str, list[dict]],
    result: HandResult,
    facts: dict,
    options: DecodeOptions,
) -> dict:
    """Run every stage once, in dependency order, and build the artifact.

    The first fit is certified only when rereads are possible; the final solution is
    certified once, after the kans' indicators and the score reconciliation, and is
    then checked against the site record.
    """
    report = Report()
    hand = hand_of(entry, obs, result, facts)
    if facts.get("site_score"):
        used = (hand.result.han, hand.result.fu)
        report.notes.append(questions.site_corrected(hand.site_han_fu, used))
    context = DenseContext(
        entry=entry,
        models=options.models,
        work_dir=options.work_dir,
        t0=hand.t0,
        t1=hand.t1,
        diagnostics=report.diagnostics,
    )
    ponds = pond_logs(hand, report)
    anchored = anchored_calls(hand, ponds, context)
    wall = dead_wall(hand, ponds, anchored.calls, report)
    calls = apply_meld_facts(hand, wall.calls, report)
    seq = turn_sequence(hand, ponds, calls, anchored.unanchored, context, report)
    riichi = place_riichi(hand, seq.turns, context, report)
    acquire_replacements(hand, seq.turns, options.models, options.work_dir, report)
    model, melds_before = build_model(hand, seq, wall.dora, report)
    search = Search(model, seq.turns, unnamed_kans(seq.calls), options)
    sol, repaired = fit_evidence(
        search,
        hand,
        seq,
        riichi,
        melds_before=melds_before,
        context=context,
        report=report,
    )
    dora, indicators = kan_indicators(
        hand, seq, wall, sol, ponds.last_discard or hand.t1, report
    )
    if sol.ok and dora != wall.dora:
        model.indicators = list(dora)
        sol = search.solve(prior=sol)
    win = Win(
        hand=hand,
        turns=seq.turns,
        live_calls=seq.live_calls,
        logs=ponds.logs,
        riichi=riichi,
        dora=dora,
        wall_tiles=sum(seq.wall_use(hand.tsumo_winner)),
    )
    # nothing changes tiles after the reconciliation: only the final solution is
    # certified, and only it is scored
    sol = reconcile_score(win, search, sol, report)
    search.certify(sol)
    score = check_score(win, search, sol, report)
    check_tenpai_and_ura(hand, model, sol, seq.turns, riichi, report)
    rec = Reconstruction(
        hand=hand,
        ponds=ponds,
        seq=seq,
        riichi=riichi,
        search=search,
        sol=sol,
        repaired=repaired,
        dora=dora,
        indicators=indicators,
        score=score,
        lost=_lost_draws(hand.facts, model, report),
    )
    unseen_tiles(rec, report)
    return artifact(rec, report)


def _lost_draws(facts: dict, model: HandModel, report: Report) -> set[tuple[str, int]]:
    """Find the draws the reviewer answered Can't tell for."""
    keys = set()
    for f in facts.get("lost", []):
        key = turn_key(model, f, LOST_FACT_WINDOW)
        if key is None:
            report.ignore("lost", f, "no draw of this player near this time")
        else:
            keys.add(key)
    return keys


def unseen_tiles(rec: Reconstruction, report: Report) -> None:
    """Ask for every tile the rules chose and nothing showed.

    A draw no frame covers, the caller's tiles of a call the turn order implies and
    no camera read, the tile of an ankan no camera named (section 6, `lost`); the
    rules' choice is the guess. A Can't-tell answer closes a draw's question.
    """
    sol, model, turns = rec.sol, rec.search.model, rec.seq.turns
    if not sol.ok:
        return  # the conflict question comes first
    for (s, j), tile in sorted(sol.draws.items()):
        if (
            tile is None
            or (s, j) in rec.lost
            or (s, j) in model.facts.draws
            or not unseen_draw(model, sol, s, j)
        ):
            continue
        st = model.turns[s][j] if j < len(model.turns[s]) else None
        t = st.t_discard if st else (turns[-1].t if turns else rec.hand.t1)
        certificate = sol.certificates.get(("draw", s, j))
        report.items.append(questions.unseen_draw(s, j, t, tile, certificate))
    for c in rec.seq.live_calls:
        if c.human:
            continue
        if c.anchor == "hidden":
            report.items.append(questions.hidden_call(c))
        elif id(c) in rec.search.unknown_kans:
            report.items.append(questions.unnamed_kan(c))


def artifact(rec: Reconstruction, report: Report) -> dict:
    """Build the decode artifact, preserving evidence, alternatives and questions."""
    hand, sol, model = rec.hand, rec.sol, rec.search.model
    items = list(report.items)
    turns = _turn_rows(rec, items)
    confidence = confidence_rows(
        model,
        sol,
        context=ReviewContext(
            turns=rec.seq.turns,
            calls=rec.seq.live_calls,
            inds=rec.indicators,
            entry=hand.entry,
            facts=hand.facts,
            lost_keys=rec.lost,
            t0=hand.t0,
        ),
    )
    items.extend(uncertain_discards(confidence, items))
    pending = uncertain_tiles(confidence, items)
    diagnostics = list(report.diagnostics)
    if sol.status == "unsolved":
        diagnostics.append("no legal hand was found within the processing time limit")
    unresolvable = sum(1 for row in confidence if row["state"] == "unresolvable")
    if unresolvable:
        diagnostics.append(f"{unresolvable} confidence checks reached their time limit")
    entry, result = hand.entry, hand.result
    return {
        "decoder_version": DECODER_VERSION,
        "hand": entry["hand"],
        "game": entry["game"],
        "kyoku": entry["kyoku"],
        "honba": entry["honba"],
        "play_window": [hand.t0, hand.t1],
        "t_last": rec.seq.turns[-1].t if rec.seq.turns else hand.t1,
        "dealer": model.dealer,
        "turns": turns,
        "haipai": sol.haipai,
        "haipai_margin": {
            seat: certificate.margin
            for (field, seat, _), certificate in sol.certificates.items()
            if field == "haipai"
        },
        "draws": {f"{s}:{j}": v for (s, j), v in sol.draws.items()},
        "dora": rec.dora,
        "ura": hand.ura,
        "indicators": rec.indicators,
        "riichi": sorted(rec.riichi.seats),
        "result": {
            "outcome": result.outcome,
            "winner": hand.winner,
            "loser": hand.loser,
            "han": result.han,
            "fu": result.fu,
            "site": list(hand.site_han_fu),
            "site_wrong": bool(hand.facts.get("site_score")),
            "deltas": result.deltas,
            "riichi": result.riichi,
            "tenpai": [site_seat(s, entry) for s in result.tenpai],
        },
        "score": rec.score,
        "solver": {
            "status": sol.status,
            "objective": round(sol.objective, 3),
            "optimal": sol.optimal,
            "low_margin": sum(
                1
                for (field, _, _), certificate in sol.certificates.items()
                if field == "draw" and low_margin(certificate.margin)
            ),
        },
        "calls": [c.to_dict() for c in rec.seq.calls],
        "notes": report.notes,
        "ignored_facts": report.ignored_facts,
        "diagnostics": diagnostics,
        "items": ranked(items + ([pending] if pending else [])),
        "confidence": confidence,
        "stats": {
            "discards": sum(len(v) for v in rec.ponds.logs.values()),
            "turns": len(rec.seq.turns),
            "calls": len(rec.seq.calls),
            "hand_evidence": len(model.hand_ev),
            "draw_evidence": len(model.draw_ev),
        },
    }


def _turn_rows(rec: Reconstruction, items: list[dict]) -> list[dict]:
    """Write each turn of the log, asking about discards the solution re-read.

    A repaired discard is a question unless a reviewer confirmed it; an unresolved
    pond correspondence is one unless verified acquisition certified the identity.
    """
    hand, sol, model = rec.hand, rec.sol, rec.search.model
    replacements = {
        (r["seat"], r["slot_id"]): r
        for r in pond_evidence.replacement_requests(
            rec.seq.turns, hand.entry, hand.facts, hand.t0, hand.t1
        )
    }

    def asked(seat: str, j: int | None) -> bool:
        return any(
            i.get("kind") == "discard" and i.get("seat") == seat and i.get("j") == j
            for i in items
        )

    rows: list[dict] = []
    for t in rec.seq.turns:
        st = next((x for x in model.turns[t.seat] if x.t_discard == t.t), None)
        j = st.j if st else None
        key = (t.seat, j)
        draw = sol.draws.get(key) if j is not None else None
        chosen = (
            (sol.discards.get(key) or rec.repaired.get(key)) if j is not None else None
        )
        draw2 = sol.draws2.get(key) if j is not None else None
        replacement = (
            replacements.get((t.seat, t.slot.id)) if t.slot is not None else None
        )
        receipt = t.slot.replacement_acquisition if t.slot is not None else None
        discard_certificate = (
            sol.certificates.get(("discard", t.seat, j)) if j is not None else None
        )
        replacement_certified = bool(
            replacement
            and replacement["acquired"]
            and receipt
            and t.slot is not None
            and (chosen or t.slot.tile) in receipt["supported_tiles"]
            and sol.ok
            and discard_certificate is not None
            and discard_certificate.state == "resolved"
        )
        if chosen is not None and t.slot is not None and j is not None:
            question = questions.changed_discard(
                (t.seat, j), t.t, t.slot.tile, chosen, hand.facts.get("discard", [])
            )
            if question and not replacement_certified and not asked(t.seat, j):
                items.append(question)
        if replacement and not replacement_certified and not asked(t.seat, j):
            question = pond_evidence.replacement_question(replacement, chosen=chosen)
            question["j"] = j
            items.append(question)
        certificate = (
            sol.certificates.get(("draw", t.seat, j)) if j is not None else None
        )
        rows.append(
            {
                "i": t.i,
                "seat": t.seat,
                "j": j,
                "kind": t.kind,
                "t": t.t,
                "t_prev": rows[-1]["t"] if rows else hand.t0,
                "riichi": t.riichi,
                "hand_before": sol.hands.get((t.seat, j - 1))
                if j is not None
                else None,
                "hand_after": sol.hands.get(key) if j is not None else None,
                "draw2": draw2,
                "discard_box": [round(float(v), 1) for v in t.slot.xyxy]
                if (t.slot and t.slot.xyxy)
                else None,
                "discard_pos": [t.slot.row, t.slot.index] if t.slot else None,
                "discard": chosen or (t.slot.tile if t.slot else None),
                "discard_conf": round(t.slot.conf, 3) if t.slot else None,
                "discard_id": t.slot.id if t.slot else None,
                "virtual": t.virtual,
                "draw": draw,
                "pond_tracking": (
                    {
                        "pending_replacement": t.slot.to_dict().get(
                            "pending_replacement"
                        ),
                        "acquisition": receipt,
                    }
                    if replacement and t.slot is not None
                    else None
                ),
                "margin": certificate.margin if certificate else None,
                "tsumogiri": (draw2 or draw) is not None
                and t.slot is not None
                and (draw2 or draw) == (chosen or t.slot.tile),
                "call": t.call.to_dict() if t.call else None,
                "own_call": t.own_call.to_dict() if t.own_call else None,
            }
        )
    return rows


# ---- the workspace -----------------------------------------------------------------


def _decode_input_binding(work: Path, hand: int, policy: EvidencePolicy) -> dict | None:
    """Bind a hand to its observation record and the dense retention policy.

    The record names the reading manifest and the observation digest it was
    written with, so new readings or votes change the binding.
    """
    record = provenance(work, hand)
    if record is None:
        return None
    return {
        "observations": json_digest(record),
        "dense_policy": policy.fingerprint_for("dense"),
    }


def _cached_decodes(
    ddir: Path,
    selected: list[dict],
    games: list[Game],
    all_facts: list[dict],
    bindings: dict,
) -> dict[int, dict]:
    """Reuse only complete results bound to current observations, results and facts."""
    cached = {}
    for h in selected:
        p = ddir / f"{h['hand']:02d}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (
                isinstance(d, dict)
                and d.get("decoder_version") == DECODER_VERSION
                and bindings[h["hand"]] is not None
                and d.get("decode_inputs") == bindings[h["hand"]]
                and d.get("decode_context")
                == decode_context(h, games[h["game"]].hands[h["site_index"]], all_facts)
            ):
                cached[h["hand"]] = d
    return cached


def run_decode(
    work: Path,
    hands: list[dict],
    games: list[Game],
    *,
    force: bool = False,
    only: set[int] | None = None,
    video_path: Path | None = None,
    cal: Calibration | None = None,
    evidence_policy: EvidencePolicy | dict | None = None,
) -> list[dict]:
    """Decode selected observed hands and persist each result under ``work/decode``.

    Current-version results bound to the hand's current observation record (see
    ``observe.provenance``), dense policy, hand metadata, site result and facts
    are reused unless ``force`` is true; ``only`` selects hands. Results are
    published atomically. Model
    identities are resolved only when a hand needs decoding and a video is
    available; weights load only when a hand rereads the video. Human labels and
    the default video resolve under VIDEO2TENHOU_HOME (the working directory by
    default), never inside an installed package.
    When dense rereads are enabled, sparse-read manifests must match the current
    source, models/runtime and calibration before any decoded hand is replaced.
    `evidence_policy` is an explicit policy or is read from detector metadata
    without loading weights. Sparse-policy provenance must match; the dense policy
    also binds decoded caches so dense-only changes cannot reuse an old result.
    An explicitly supplied video must exist. Model, reading and filesystem
    failures propagate; only reconstruction without a video skips dense reads.
    """
    if video_path is not None and not video_path.is_file():
        raise FileNotFoundError(video_path)
    policy = (
        retention.load_policy()
        if evidence_policy is None
        else retention.resolve_policy(evidence_policy)
    )
    ddir = work / "decode"
    ddir.mkdir(parents=True, exist_ok=True)
    all_facts = load_facts(DATA_DIR / "labels" / work.name)
    models: ReadModels | None = None
    selected = [h for h in hands if only is None or h["hand"] in only]
    bindings = {
        h["hand"]: _decode_input_binding(work, h["hand"], policy) for h in selected
    }
    cached = (
        {} if force else _cached_decodes(ddir, selected, games, all_facts, bindings)
    )
    pending = [
        h
        for h in selected
        if h["hand"] not in cached and (work / "obs" / f"{h['hand']:02d}.json").exists()
    ]
    if video_path is None:
        cand = DATA_DIR / "samples" / f"{work.name}.mp4"
        video_path = cand if cand.exists() else None
    if pending and video_path is not None:
        if retention.load_policy() != policy:
            raise ValueError(
                "Recognition evidence policy changed before rebuilding. "
                "Choose Analyze recording to refresh evidence."
            )
        # Identities are resolved now; weights load only if a hand rereads video.
        from video2tenhou.perception.classifier import LazyClassifier  # noqa: PLC0415
        from video2tenhou.perception.detector import LazyDetector  # noqa: PLC0415

        det, clf = LazyDetector(), LazyClassifier()
        if det.evidence_policy != policy:
            raise ValueError(
                "Recognition evidence policy changed before rebuilding. Choose "
                "Analyze recording to refresh evidence."
            )
        calibration = cal or Calibration.load("pml", video_path)
        models = (det, clf, video_path, calibration)
        # Every selected hand must pass before any existing result is replaced.
        validate_read_cache(
            ReadContext(video_path, calibration, work, det, clf), pending
        )
        validate_observation_cache(work, pending, policy=policy)
    out = []
    for h in selected:
        p = ddir / f"{h['hand']:02d}.json"
        if h["hand"] in cached:
            out.append(cached[h["hand"]])
            continue
        if not (work / "obs" / f"{h['hand']:02d}.json").exists():
            continue
        obs = load_obs(work, h["hand"])
        result = games[h["game"]].hands[h["site_index"]]
        d = decode_hand(
            h,
            obs,
            result,
            facts_for_hand(all_facts, h),
            options=DecodeOptions(models=models, work_dir=work),
        )
        d["decode_inputs"] = bindings[h["hand"]]
        d["decode_context"] = decode_context(h, result, all_facts)
        atomic_write_json(p, sanitize(d), indent=1)
        out.append(d)
        sc = d["score"]
        LOGGER.info(
            "  hand %2d: %s turns, %s calls, dora %s, solver %s obj %s, "
            "low-margin draws %s, score %s/%s vs site %s/%s",
            h["hand"],
            d["stats"]["turns"],
            d["stats"]["calls"],
            d["dora"],
            d["solver"]["status"],
            d["solver"]["objective"],
            d["solver"]["low_margin"],
            "ok" if sc and sc["match"] else sc["han"] if sc else "-",
            sc["fu"] if sc else "-",
            d["result"]["han"],
            d["result"]["fu"],
        )
    return out
