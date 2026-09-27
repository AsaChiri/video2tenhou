import numpy as np
import pytest

from video2tenhou.engine import rules
from video2tenhou.engine.solver import (HandEvidence, HandModel, SeatTurn, TILES, TI, hand_evidence, state_of)
from video2tenhou.train.data import CLASS_INDEX, CLASSES


@pytest.mark.parametrize("optimal,bound,expected", [(False, 12.0, 2.0), (False, 8.0, 0.0), (True, 12.0, 5.0)])
def test_timeout_margin_uses_certified_bound_not_candidate_cost(monkeypatch, optimal, bound, expected):
    """A poor candidate at timeout is no proof that the chosen draw is certain."""
    from types import SimpleNamespace
    from video2tenhou.engine import solver

    class TimedSolver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, model):
            return solver.cp_model.OPTIMAL if optimal else solver.cp_model.FEASIBLE

        def ObjectiveValue(self):
            return 15.0 * solver.SCALE ** 2

        def BestObjectiveBound(self):
            return bound * solver.SCALE ** 2

    monkeypatch.setattr(solver.cp_model, "CpSolver", TimedSolver)
    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    hint = ({s: [0] * len(TILES) for s in rules.SEATS}, {})
    margin, alternative, gap = model._resolve(hint, 10.0, workers=1)
    assert margin == expected
    assert alternative is None
    assert gap == 5.0


@pytest.mark.parametrize("status,expected", [("UNKNOWN", 0.0), ("INFEASIBLE", float("inf"))])
def test_counterfactual_without_candidate_has_conservative_acquisition_gap(monkeypatch, status, expected):
    from types import SimpleNamespace
    from video2tenhou.engine import solver

    class NoCandidate:
        def __init__(self):
            self.parameters=SimpleNamespace()
        def Solve(self, model):
            return getattr(solver.cp_model,status)
        def BestObjectiveBound(self):
            return 0.0

    monkeypatch.setattr(solver.cp_model,'CpSolver',NoCandidate)
    model=HandModel('E',{s:[] for s in rules.SEATS},[])
    hint=({s:[0]*len(TILES) for s in rules.SEATS},{})
    assert model._resolve(hint,10,workers=1)==(expected,None,expected)


def test_confidence_search_stops_only_on_certified_bound_above_threshold(monkeypatch):
    from types import SimpleNamespace
    from video2tenhou.engine import solver

    instances = []

    class ProofSolver:
        def __init__(self):
            self.parameters = SimpleNamespace()
            self.stopped = False
            instances.append(self)
        def StopSearch(self):
            self.stopped = True
        def Solve(self, model):
            # A poor incumbent would not justify stopping. A bound on the
            # inclusive review boundary is insufficient as well.
            self.best_bound_callback(10.5 * solver.SCALE ** 2)
            assert not self.stopped
            self.best_bound_callback(10.5001 * solver.SCALE ** 2)
            assert self.stopped
            return solver.cp_model.UNKNOWN
        def BestObjectiveBound(self):
            return 10.5001 * solver.SCALE ** 2

    monkeypatch.setattr(solver.cp_model, 'CpSolver', ProofSolver)
    model = HandModel('E', {s: [] for s in rules.SEATS}, [])
    hint = ({s: [0] * len(TILES) for s in rules.SEATS}, {})
    margin, alternative, gap = model._resolve(hint, 10, workers=1)
    assert margin == pytest.approx(.5001)
    assert alternative is None and gap == 0


def test_variable_discard_is_certified_even_when_raw_reading_is_kept():
    from video2tenhou.engine.solver import Facts
    turns = {s: [] for s in rules.SEATS}
    p = np.zeros(len(TILES))
    p[TI['1m']], p[TI['2m']] = .51, .49
    turns['S'] = [SeatTurn(0, 'draw', '1m', 0, 10, discard_p=p)]
    model = HandModel('E', turns, [])
    model.facts = Facts(draws={('S', 0): '3m'})
    tiles = ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', '1z', '1z', '3z', '6z']
    model.facts.haipai['S'] = tiles
    after_second = tiles + ['3m']
    after_second.remove('2m')
    # The hand view almost offsets the penalty for choosing the second pond
    # reading: raw1m remains best, but alternative2m is only .01 more costly.
    model.hand_ev = [HandEvidence('S', 0, False, counts(after_second), .76, 11, 12)]
    first = model.solve(margins=False, workers=1)
    assert first.ok
    model.facts.haipai.update(first.haipai)
    sol = model.solve(workers=1)
    assert ('S', 0) not in sol.discards
    assert sol.discard_margins[('S', 0)] == pytest.approx(.01)
    assert sol.discard_runner_up[('S', 0)] == '2m'


@pytest.mark.parametrize("status,bound,expected", [("UNKNOWN", 12.0, 2.0),
                                                  ("UNKNOWN", 8.0, 0.0),
                                                  ("UNKNOWN", float("nan"), 0.0),
                                                  ("MODEL_INVALID", 12.0, 0.0)])
def test_candidate_free_search_retains_only_certified_bounds(monkeypatch, status, bound, expected):
    from types import SimpleNamespace
    from video2tenhou.engine import solver
    from video2tenhou.engine.review import draws_to_reread

    class NoCandidate:
        parameters = SimpleNamespace()
        def Solve(self, model):
            return getattr(solver.cp_model, status)
        def BestObjectiveBound(self):
            return bound * solver.SCALE ** 2

    monkeypatch.setattr(solver.cp_model, "CpSolver", NoCandidate)
    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    hint = ({s: [0] * len(TILES) for s in rules.SEATS}, {})
    margin, alternative, gap = model._resolve(hint, 10.0, workers=1)
    assert (margin, alternative, gap) == (expected, None, 0.0)
    result = solver.Solution("optimal", 10.0, {}, {("S", 0): "2p"}, {},
                             margins={("S", 0): margin}, alternative_gaps={("S", 0): gap})
    assert draws_to_reread(result) == ([] if expected > 0.5 else [("S", 0)])


@pytest.mark.parametrize("status,optimal,needs_review", [("feasible", False, True), ("repaired", False, True),
                                                       ("optimal", True, False), ("repaired", True, False)])
def test_incomplete_legal_reconstruction_stays_provisional(status, optimal, needs_review):
    """Repair must not hide a timed-out search or mislabel a legal log as a conflict."""
    from types import SimpleNamespace
    from video2tenhou.engine.decode import HandDecoder
    from video2tenhou.engine.solver import Solution

    decoder = SimpleNamespace(sol=Solution(status, 1.0, {}, {}, {}, optimal=optimal),
                              items=[], turns=[], t1=10)
    for stage in ("window", "discards", "calls", "indicators", "meld_facts", "turn_sequence", "end_of_hand",
                  "riichi", "pond_replacements", "solve", "kan_indicators", "check_score", "check_draw", "unseen_tiles"):
        setattr(decoder, stage, lambda: None)
    decoder.output = lambda: decoder.items
    items = HandDecoder.run(decoder)
    assert [item['kind'] for item in items] == (["solver_incomplete"] if needs_review else [])
    assert decoder.sol.ok


def counts(tiles):
    e = np.zeros(len(TILES))
    for t in tiles:
        e[TI[t]] += 1
    return e


def slot_dict(tile):
    p = np.full(len(CLASSES), 0.001)
    p[CLASS_INDEX[tile]] = 0.95
    p /= p.sum()
    return {"key": [0], "tile": tile, "conf": 0.95, "seen": 3, "sideways": 0.0, "disagree": False, "xyxy": [0, 0, 1, 1], "p": p.tolist()}


def obs(t0, t1, tiles):
    return {"region": "hand:TL", "t0": t0, "t1": t1, "n_readings": 3, "n_used": 3, "count": len(tiles), "quality": 0.9,
            "slots": [slot_dict(t) for t in tiles], "indicators": []}


def test_solver_recovers_draws_from_perfect_states():
    # one non-dealer seat, three turns; the other seats have no turns and no evidence
    haipai = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "6z"]
    draws = ["9m", "2z", "3z"]
    discards = ["6z", "9m", "1z"]        # turn 2 is a tsumogiri
    turns = {s: [] for s in rules.SEATS}
    hand = list(haipai)
    states = []
    for j, (dr, di) in enumerate(zip(draws, discards)):
        turns["S"].append(SeatTurn(j, "draw", di, 100 * j + 10, 100 * j + 20))
        hand.append(dr); hand.remove(di)
        states.append(list(hand))
    model = HandModel("E", turns, ["5z"])
    model.hand_ev.append(HandEvidence("S", -1, False, counts(haipai), 1.0, 0, 5))
    for j, st in enumerate(states):
        model.hand_ev.append(HandEvidence("S", j, False, counts(st), 1.0, 100 * j + 30, 100 * j + 90))
    sol = model.solve(time_limit=10, margins=True)
    assert sol.status in ("optimal", "feasible")
    assert sorted(sol.haipai["S"]) == sorted(haipai)
    assert [sol.draws[("S", j)] for j in range(3)] == draws
    assert all(m > 0 for m in sol.margins.values())


def test_state_mapping_and_direct_draw_evidence():
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 110, 120)]
    assert state_of(st, 0, 5) == -1
    assert state_of(st, 25, 60) == 0
    assert state_of(st, 15, 60) is None       # straddles the discard of turn 0
    assert state_of(st, 25, 115) == 0        # ends before the discard of turn 1 (its disturbed window started at 110)
    assert state_of(st, 130, 200) == 1
    o13 = obs(25, 60, ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"])
    o14 = obs(100, 108, ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m", "2z"])
    hev, dev, habit = hand_evidence("S", st, [o13, o14], {}, dealer=False)
    assert [(e.j, e.after_draw) for e in hev] == [(0, False), (0, True)]
    assert len(dev) == 1 and dev[0].j == 1 and TILES[int(np.argmax(dev[0].p))] == "2z" and dev[0].weight > 1.0
    assert habit == (0, 1)


def test_oversized_row_cannot_draw_before_the_preceding_players_discard():
    """A detector's extra tile before the next legal draw remains resting-state evidence."""
    tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    turns = [SeatTurn(0, "draw", "6z", 10, 20),
             SeatTurn(1, "draw", "9m", 110, 120, t_draw_min=100)]
    early = obs(80, 85, tiles + ["2z"])
    oversized = obs(90, 95, tiles + ["2z", "3z"])
    timely = obs(105, 108, tiles + ["2z"])
    hev, dev, _ = hand_evidence("S", turns, [early, oversized, timely], {}, dealer=False)
    assert len(hev) == 2  # do not admit impossible rows the existing mapper excluded
    assert hev[0].j == 0 and not hev[0].after_draw and hev[0].subset
    assert hev[1].j == 0 and hev[1].after_draw and not hev[1].subset
    assert dev and all(d.j == 1 for d in dev)
    assert sum(d.weight for d in dev) < 2.0  # only the timely row names an end


def test_drawn_end_needs_the_rest_to_match_and_learns_the_habit():
    from video2tenhou.engine.solver import drawn_end
    thirteen = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    ref = [obs(0, 1, thirteen)["slots"][i]["p"] for i in range(13)]
    # the player did not sort: the rest matches as a multiset, not by position
    row = obs(0, 1, ["2z"] + thirteen[::-1])["slots"]
    assert drawn_end(row, ref, None) == [("L", 2.0)]
    # a reference row of another state (four tiles differ) names nothing: the habit weighs the ends
    other = obs(0, 1, ["5z", "6z", "7z", "2p"] + thirteen[4:] + ["2z"])["slots"]
    weights = dict(drawn_end(other, ref, (0, 9)))
    assert weights["R"] > 5 * weights["L"]


def test_repair_re_reads_a_discard_but_not_one_a_call_took():
    """A discard the hand cannot hold is re-read by repair, at the cost of its posterior; a discard a call took
    was read by the meld camera too, so repair leaves it and the hand stays illegal (a contradiction elsewhere)."""
    haipai = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "6z"]
    for taken in (False, True):
        p = np.full(len(TILES), 0.001)
        p[TI["5z"]], p[TI["6z"]] = 0.8, 0.15
        turns = {s: [] for s in rules.SEATS}
        turns["S"].append(SeatTurn(0, "draw", "5z", 10, 20, discard_p=p / p.sum(), taken=taken))
        model = HandModel("E", turns, ["2z"])
        model.facts.haipai["S"] = list(haipai)
        model.facts.draws[("S", 0)] = "9m"
        model.repair = True
        sol = model.solve(time_limit=10, margins=False)
        if taken:
            assert not sol.ok
        else:
            assert sol.ok and sol.discards[("S", 0)] in haipai + ["9m"]


def test_a_forbidden_hand_gives_the_next_best_reconstruction():
    """The second VOD's hand 2: the views split one tile between 5s and 8s; the cheaper reading scored below the
    site, so the winner's hand is forbidden and the next best, as the reveal read it, is found."""
    haipai = ["1m", "2m", "3m", "6p", "7p", "4s", "5s", "6s", "6s", "7s", "8s", "8s", "9m"]
    turns = {s: [] for s in rules.SEATS}
    turns["S"].append(SeatTurn(0, "draw", "9m", 10, 20))
    model = HandModel("E", turns, ["2z"])
    model.facts.haipai["S"] = list(haipai)
    ambiguous = np.zeros(len(TILES))
    ambiguous[TI["8s"]], ambiguous[TI["5s"]] = 0.6, 0.4           # the draw: 8s read a little better than 5s
    from video2tenhou.engine.solver import DrawEvidence
    model.draw_ev.append(DrawEvidence("S", 0, ambiguous, 1.0))
    best = model.solve(time_limit=10, margins=False)
    assert best.draws[("S", 0)] == "8s"
    model.forbidden_hands.append(("S", 0, sorted(best.hands[("S", 0)])))
    nxt = model.solve(time_limit=10, margins=False)
    assert nxt.ok and nxt.draws[("S", 0)] == "5s" and nxt.objective >= best.objective


def test_a_bound_hand_holds_through_the_solve():
    """Week 11 hand 3: the hand the site's score required must survive the solve that follows it (a time-limited
    solve returned another of equal cost). A bound state fixes the draws that bring it."""
    haipai = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "5s", "6s", "6s", "8s", "9m"]
    turns = {s: [] for s in rules.SEATS}
    turns["S"].append(SeatTurn(0, "draw", "9m", 10, 20))
    model = HandModel("E", turns, ["2z"])
    model.facts.haipai["S"] = list(haipai)
    ambiguous = np.zeros(len(TILES))
    ambiguous[TI["8s"]], ambiguous[TI["5s"]] = 0.6, 0.4
    from video2tenhou.engine.solver import DrawEvidence
    model.draw_ev.append(DrawEvidence("S", 0, ambiguous, 1.0))
    want = sorted([t for t in haipai if t != "9m"] + ["5s"])
    model.bound_hands = [("S", 0, want)]
    sol = model.solve(time_limit=10, margins=False)
    assert sol.ok and sol.draws[("S", 0)] == "5s" and sorted(sol.hands[("S", 0)]) == want
