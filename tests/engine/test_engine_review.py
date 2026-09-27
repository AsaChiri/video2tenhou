"""Engine fixes from the code review: turns, melds, ponds, solver, assemble, decode facts."""
import numpy as np
import pytest

from video2tenhou.engine import rules
from video2tenhou.engine.assemble import call_string, kyoku_from_decode
from video2tenhou.engine.decode import apply_choices, scoring_melds, seat_turns_of
from video2tenhou.engine.indicators import reconcile_kans
from video2tenhou.engine.review import facts_for_hand
from video2tenhou.engine.melds import Call, track_melds
from video2tenhou.engine.ponds import PondSlot
from video2tenhou.engine.solver import HandModel, SeatTurn, hand_evidence, open_turn
from video2tenhou.engine.turns import Turn, assign_calls, merge
from video2tenhou.record import HandResult
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def test_confidence_boundary_requests_more_evidence():
    from video2tenhou.engine.review import low_margin
    assert low_margin(0.5)
    assert low_margin(0.50000000000001)
    assert low_margin(0.0)
    assert not low_margin(0.5001)
    assert not low_margin(None)
    assert not low_margin(float("inf"))


def test_covered_ambiguous_tiles_stay_in_review_without_duplicate_or_answered_questions():
    from video2tenhou.engine.review import uncertain_tiles

    def draw(j, **changes):
        return {"field": "draw", "seat": "W", "turn": j, "value": "4m", "margin": 0.0,
                "human": False, "lost": False, "evidence": [{"region": "hand:TL", "t": 100 + j}], **changes}

    rows = [draw(4), draw(5, value="7m"), draw(6, human=True), draw(7, lost=True), draw(8), draw(9, margin=2.0),
            {"field": "haipai", "seat": "E", "turn": -1, "value": ["1m"] * 14, "margin": 0.5,
             "human": False, "lost": False, "evidence": []}]
    item = uncertain_tiles(rows, [{"kind": "draw", "seat": "W", "j": 8}])
    assert item["kind"] == "uncertain_tiles" and item["count"] == 3
    assert [(c["field"], c["seat"], c["j"]) for c in item["choices"]] == [
        ("draw", "W", 4), ("draw", "W", 5), ("haipai", "E", -1)]
    assert uncertain_tiles([draw(6, human=True), draw(7, lost=True)], []) is None


def test_cant_tell_starting_hand_does_not_answer_a_draw():
    from video2tenhou.engine.review import confidence_rows, uncertain_tiles
    from video2tenhou.engine.solver import Solution

    entry={"game": 0, "kyoku": 0, "honba": 0, "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}}
    facts=facts_for_hand([{"game": 0, "kyoku": 0, "honba": 0, "corner": "TR", "kind": "lost", "field": "haipai"}],entry)
    assert facts['lost_haipai']==['S'] and facts['lost']==[]
    model=HandModel('E',{s: [] for s in rules.SEATS},[])
    model.turns['S']=[SeatTurn(0,'draw','1m',10,20)]
    sol=Solution('optimal',0,{'S':['1m']*13},{('S',0):'2m'},{},margins={('S',0):0},haipai_margins={'S':0})
    rows=confidence_rows(model,sol,[],[],[],entry,facts,set(),0)
    haipai=next(r for r in rows if r['field']=='haipai')
    assert haipai['lost'] and not haipai['human']
    assert uncertain_tiles([haipai],[]) is None
    # The first draw is still unreviewed and receives its own unseen question;
    # the starting-hand answer never enters the set of answered draw keys.
    assert facts['lost']==[] and not next(r for r in rows if r['field']=='draw')['human']


def test_solver_changed_discard_needs_specific_review_unless_human_fixed():
    from video2tenhou.engine.review import changed_discard

    item=changed_discard('S',17,3097.5,'1m','2m',[])
    assert item['kind']=='discard' and item['observed']=='1m' and item['tile']=='2m'
    assert changed_discard('S',17,3097.5,'2m','2m',[]) is None
    assert changed_discard('S',17,3097.5,'1m','2m',[{'seat':'S','t':3097.5,'tile':'2m'}]) is None
    assert changed_discard('S',17,3097.5,'1m','2m',[{'seat':'S','t':3097.5,'tile':'1m'}]) is not None


def test_search_uncertainty_does_not_alone_request_more_video():
    from video2tenhou.engine.review import draws_to_reread,uncertain_tiles
    from video2tenhou.engine.solver import Solution

    sol=Solution('optimal',0,{}, {('S',j):'2m' for j in range(4)}, {},
                 margins={('S',j):0.0 for j in range(4)},
                 alternative_gaps={('S',0):5.0,('S',1):0.5,('S',2):0.0,('S',3):float('inf')})
    assert draws_to_reread(sol)==[('S',1),('S',2)]  # close candidate and UNKNOWN
    rows=[{'field':'draw','seat':'S','turn':0,'value':'2m','margin':sol.margins[('S',0)],
           'alternative_gap':5.0,'human':False,'lost':False,'evidence':[]}]
    assert uncertain_tiles(rows,[])['count']==1  # high candidate gap never certifies the tile


def test_serialized_confidence_keeps_threshold_precision():
    from video2tenhou.engine.review import confidence_rows,uncertain_tiles
    from video2tenhou.engine.solver import Solution

    entry={'corner_wind':{'TL':'E','TR':'S','BR':'W','BL':'N'}}
    model=HandModel('E',{s:[] for s in rules.SEATS},[])
    model.turns['S']=[SeatTurn(0,'draw','1m',10,20),SeatTurn(1,'draw','2m',30,40)]
    sol=Solution('optimal',0,{}, {('S',0):'3m',('S',1):'4m'}, {},
                 margins={('S',0):0.5004,('S',1):0.5},alternative_gaps={('S',0):0.5004,('S',1):0.5})
    rows=confidence_rows(model,sol,[],[],[],entry,{},set(),0)
    assert rows[0]['margin']==rows[0]['alternative_gap']==0.5004
    # Supply image coverage: these choices belong to the grouped review policy.
    for row in rows:
        row['lost']=False
    assert [c['j'] for c in uncertain_tiles(rows,[])['choices']]==[1]


def pslot(i, tile, t, removed=None, side=False, row=0, index=None):
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.9
    s = PondSlot(i, row, i if index is None else index, p, t, (t - 3, t), t + 4, 3, 3.0 if side else 0.0)
    s.t_removed = removed
    return s


def mslot(tile, group, i, sideways=0.0):
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.8
    p /= p.sum()
    return {"key": [group, i], "tile": tile, "conf": 0.9, "seen": 3, "sideways": sideways, "disagree": False,
            "xyxy": [i * 40, group * 70, i * 40 + 38, group * 70 + 58], "p": p.tolist()}


def mobs(t0, t1, slots, n_used=3, partial=False):
    return {"region": "meld:TL", "t0": t0, "t1": t1, "n_readings": n_used, "n_used": n_used, "count": len(slots), "quality": 0.9,
            "slots": slots, "indicators": [], "partial": partial}


# ---------------------------------------------------------------- turns

def test_unknown_source_prefers_the_pond_that_shows_the_removal():
    # W pons 3p; its turned tile was not read. S (kamicha of W) has no removed 3p, N (shimocha) does: N is the source
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
            "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
            "W": [pslot(4, "5s", 30), pslot(5, "6s", 49)],
            "N": [pslot(6, "3p", 40, removed=47), pslot(7, "9m", 52)]}
    pon = Call("W", 48, (44, 48), "pon", ["3p", "3p", "3p"], None, None, None, [], 0.9)
    turns, problems = merge(logs, [pon], "E")
    assert pon.source == "shimocha" and pon.called_pos == 2
    assert problems == [] and not any(t.virtual for t in turns)
    # after W's call and discard the turn passes to N, who discards; a seat never passes for free
    assert [t.seat for t in turns] == ["E", "S", "W", "N", "W", "N", "E", "S"]


def test_daiminkan_from_shimocha_has_the_turned_tile_last():
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
            "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
            "W": [pslot(4, "5s", 30), pslot(5, "6s", 70)],
            "N": [pslot(6, "7z", 40, removed=47)]}
    kan = Call("W", 48, (44, 48), "kan", ["7z", "7z", "7z", "7z"], None, None, None, [], 0.9)
    merge(logs, [kan], "E")
    assert kan.source == "shimocha" and kan.called_pos == 3


def test_one_call_takes_one_removed_slot():
    # S's pond lost two 3p; one pon by W: only the removal nearest the call gets it
    seq = {"E": [], "S": [pslot(0, "3p", 20, removed=27), pslot(1, "3p", 40, removed=60)], "W": [], "N": []}
    pon = Call("W", 62, (58, 62), "pon", ["3p", "3p", "3p"], 0, "kamicha", "3p", [], 0.9)
    callers = assign_calls([pon], seq)
    assert callers["S"] == [None, pon]


# ---------------------------------------------------------------- melds

def test_two_identical_chis_are_two_calls():
    g = lambda k: [mslot("2m", k, 0, sideways=1.0), mslot("3m", k, 1), mslot("4m", k, 2)]
    seq = [mobs(0, 5, []), mobs(10, 14, g(0)), mobs(20, 24, g(0)), mobs(30, 34, g(0) + g(1)), mobs(40, 44, g(0) + g(1))]
    calls = track_melds("E", seq)
    assert [c.type for c in calls] == ["chi", "chi"] and [c.t_first for c in calls] == [10, 30]


def test_a_meld_seen_once_and_then_absent_is_not_real():
    pon = [mslot("7z", 0, 0), mslot("7z", 0, 1), mslot("7z", 0, 2, sideways=1.0)]
    # seen once at 10, then two full views without it: a misread; the chi seen once in the last view stays
    chi = [mslot("2m", 1, 0, sideways=1.0), mslot("3m", 1, 1), mslot("4m", 1, 2)]
    seq = [mobs(0, 5, []), mobs(10, 14, pon), mobs(20, 24, []), mobs(30, 34, []), mobs(40, 44, chi)]
    calls = track_melds("E", seq)
    assert [c.type for c in calls] == ["chi"]


def test_observation_without_readings_keeps_the_call_window():
    pon = [mslot("7z", 0, 0), mslot("7z", 0, 1), mslot("7z", 0, 2, sideways=1.0)]
    seq = [mobs(0, 5, []), mobs(8, 9, [], n_used=0), mobs(10, 14, pon), mobs(20, 24, pon)]
    calls = track_melds("E", seq)
    assert calls[0].t_window == (5, 10)


# ---------------------------------------------------------------- solver

def hobs(t0, t1, tiles):
    def sd(tile):
        p = np.full(len(CLASSES), 0.001)
        p[CLASS_INDEX[tile]] = 0.95
        p /= p.sum()
        return {"key": [0], "tile": tile, "conf": 0.95, "seen": 3, "sideways": 0.0, "disagree": False, "xyxy": [0, 0, 1, 1], "p": p.tolist()}
    return {"region": "hand:TL", "t0": t0, "t1": t1, "n_readings": 3, "n_used": 3, "count": len(tiles), "quality": 0.9,
            "slots": [sd(t) for t in tiles], "indicators": []}


def test_hand_row_inside_a_turn_window_counts_only_after_a_draw():
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 100, 120)]
    # a row ending within 8 s of the discard's first sighting is ambiguous; one ending long before it is not
    assert open_turn(st, 25, 60) is None and open_turn(st, 108, 115) is st[1] and open_turn(st, 102, 110) is None
    thirteen = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    o13 = hobs(108, 115, thirteen)                      # the discard may already have happened: ambiguous
    o14 = hobs(108, 115, thirteen + ["2z"])             # one more tile: after the draw, before the discard
    # the 13 tiles are in the hand at some moment of the turn: all of them are in the hand after its draw
    hev, dev, _ = hand_evidence("S", st, [o13], {}, dealer=False)
    assert [(e.j, e.after_draw, e.subset) for e in hev] == [(0, True, True)] and dev == []
    hev, dev, _ = hand_evidence("S", st, [o14], {}, dealer=False)
    assert [(e.j, e.after_draw, e.subset) for e in hev] == [(0, True, False)] and len(dev) == 2


def test_ura_indicators_count_against_the_four():
    turns = {s: [] for s in rules.SEATS}
    turns["S"].append(SeatTurn(0, "draw", "9p", 10, 20))
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9p"]
    with_ura = HandModel("E", turns, ["1m", "1m", "1m"], ura=["1m"])
    with_ura.facts.haipai["S"] = hand
    assert with_ura.solve(time_limit=5, margins=False).status == "infeasible"
    without = HandModel("E", turns, ["1m", "1m", "1m"])
    without.facts.haipai["S"] = hand
    assert without.solve(time_limit=5, margins=False).status != "infeasible"


def test_kakan_of_fives_reports_which_five_was_added():
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [SeatTurn(0, "call", "1z", 10, 20, removed=["5p", "5p"]),
                  SeatTurn(1, "kan", "2z", 100, 120, kan="kakan", kan_tile="5p", two_draws=True, rinshan=True)]
    model = HandModel("E", turns, ["9s"])
    model.facts.haipai["S"] = ["1m", "2m", "3m", "5p", "5p", "0p", "7s", "8s", "9s", "1z", "2z", "3z", "6z"]
    sol = model.solve(time_limit=5, margins=False)
    assert sol.status != "infeasible" and sol.kan_added[("S", 1)] == "0p"


def test_ankan_after_riichi_is_of_the_drawn_tile_and_the_rinshan_draw_is_discarded():
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [SeatTurn(0, "draw", "1z", 10, 20, riichi=True),
                  SeatTurn(1, "kan", "9p", 100, 120, kan="ankan", kan_tile="3m", two_draws=True, rinshan=True)]
    model = HandModel("E", turns, ["9s"])
    model.facts.haipai["S"] = ["3m", "3m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "2z", "6z"]
    sol = model.solve(time_limit=5, margins=False)
    assert sol.status != "infeasible"
    assert sol.draws[("S", 1)] == "3m" and sol.draws2[("S", 1)] == "9p"


# ---------------------------------------------------------------- assemble

def test_kans_of_fives_hold_the_red_one():
    assert call_string({"type": "ankan", "tiles": ["5p", "5p", "X", "X"]}, "E") == "522525a25"
    assert call_string({"type": "kakan", "tiles": ["5s", "5s", "5s", "0s"], "called_pos": 0, "source": "kamicha"}, "E") == "k53353535"
    assert call_string({"type": "kakan", "tiles": ["5s", "5s", "5s", "5s"], "called_pos": 0, "source": "kamicha"}, "E") == "k35533535"
    assert call_string({"type": "kan", "tiles": ["5m", "5m", "5m", "5m"], "called_pos": 0, "source": "kamicha"}, "E") == "m15511515"


def test_ura_and_the_dealer_split_reach_the_log():
    d = {"dealer": "E", "haipai": {"E": ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "6z", "9p"],
                                   "S": [], "W": [], "N": []},
         "turns": [{"i": 0, "seat": "E", "j": 0, "kind": "draw", "t": 10, "riichi": False, "draw": None, "draw2": None,
                    "discard": "9p", "tsumogiri": False, "margin": None, "call": None, "own_call": None}],
         "draws": {}, "dora": ["1s"], "ura": ["2s"], "result": {"winner": None, "loser": None}, "score": None}
    entry = {"kyoku": 0, "honba": 1, "sticks": 0, "scores": {"E": 25000, "S": 25000, "W": 25000, "N": 25000}}
    result = HandResult(0, 1, 0, {"EAST": 0, "SOUTH": 0, "WEST": 0, "NORTH": 0}, "draw")
    k, conf = kyoku_from_decode(d, entry, result)
    assert k.ura == [32] and k.draws[0] == [29] and k.discards[0] == [60]           # 9p drawn, discarded tsumogiri
    assert not any(c["lost"] for c in conf)


# ---------------------------------------------------------------- decode: kans, melds, facts

def test_scoring_melds_fill_kans_and_reds():
    calls = [Call("E", 10, (5, 10), "ankan", ["5p", "5p", "X", "X"], None, None, None, [], 0.9),
             Call("E", 20, (15, 20), "kakan", ["3s", "3s", "3s", "3s"], 0, "kamicha", "3s", [], 0.9),
             Call("E", 30, (25, 30), "pon", ["5m", "5m", "5m"], 0, "kamicha", "5m", [], 0.9),
             Call("S", 30, (25, 30), "pon", ["7z", "7z", "7z"], 0, "kamicha", "7z", [], 0.9)]
    assert scoring_melds(calls, "E") == [{"type": "ankan", "tiles": ["5p", "5p", "5p", "0p"]},
                                         {"type": "kakan", "tiles": ["3s", "3s", "3s", "3s"]},
                                         {"type": "pon", "tiles": ["5m", "5m", "5m"]}]
    assert rules.count_ok([t for m in scoring_melds(calls, "E") for t in m["tiles"]])


def test_solver_kan_choices_reach_the_calls():
    kan = Call("E", 50, (42, 50), "ankan", ["?", "?", "X", "X"], None, None, None, [], 0.3)
    kakan = Call("S", 80, (72, 80), "kakan", ["5p", "5p", "5p", "5p"], 0, "kamicha", "5p", [], 0.9)
    turns = [Turn(0, "E", "kan", pslot(0, "1z", 55), 55, own_call=kan), Turn(1, "S", "kan", pslot(1, "2z", 85), 85, own_call=kakan)]
    model = HandModel("E", {s: [] for s in rules.SEATS}, ["9s"])
    for s in rules.SEATS:
        model.turns[s], _ = seat_turns_of(turns, s, "E")

    class Sol:
        kans = {("E", 0): "3m"}
        kan_added = {("S", 0): "0p"}
        melds = {}
    apply_choices(Sol(), model, turns, {id(kan)})
    assert kan.tiles == ["3m"] * 4 and kakan.tiles == ["5p", "5p", "5p", "0p"]


def test_dealer_first_turn_ankan_has_no_normal_draw():
    kan = Call("E", 12, (5, 12), "ankan", ["3m", "3m", "X", "X"], None, None, None, [], 0.9)
    turns = [Turn(0, "E", "kan", pslot(0, "1z", 15), 15, own_call=kan), Turn(1, "S", "draw", pslot(1, "2z", 25), 25)]
    st, _ = seat_turns_of(turns, "E", "E")
    assert st[0].kind == "first" and st[0].kan == "ankan" and st[0].two_draws
    model = HandModel("E", {"E": st, "S": [], "W": [], "N": []}, ["9s"])
    assert ("E", 0) not in model.build()[2] and ("E", 0) in model._d2


def test_a_kan_after_the_winners_last_discard_is_its_last_turn():
    kan = Call("S", 100, (92, 100), "ankan", ["3m", "3m", "X", "X"], None, None, None, [], 0.9)
    turns = [Turn(0, "E", "draw", pslot(0, "1z", 15), 15), Turn(1, "S", "draw", pslot(1, "2z", 25), 25),
             Turn(2, "S", "kan", None, 100, own_call=kan)]
    st, melds_before = seat_turns_of(turns, "S", "E")
    assert [x.kind for x in st] == ["draw", "kan"] and not st[1].two_draws and st[1].discard is None
    model = HandModel("E", {"E": [], "S": st, "W": [], "N": []}, ["9s"], tsumo_winner="S")
    assert model._draw_turns("S") == [0, 1, 2]           # the normal draw of the kan turn, then the winning rinshan draw
    model.facts.haipai["S"] = ["3m", "3m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "2z", "6z"]
    sol = model.solve(time_limit=5, margins=False)
    assert sol.status != "infeasible" and len(sol.hands[("S", 2)]) == 11


def test_indicator_after_the_last_discard_is_the_tsumo_winners_kan():
    logs = {"E": [pslot(0, "1z", 15)], "S": [pslot(1, "2z", 25)], "W": [], "N": []}
    inds = [{"tile": "1s", "t_first": 0}, {"tile": "4p", "t_first": 40}]
    entry = {"corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}, "corner_site": {"TL": "EAST", "TR": "SOUTH", "BR": "WEST", "BL": "NORTH"}}
    problems = []
    out = reconcile_kans(inds, logs, [], {}, entry, 0, problems)
    assert out == [] and "no discard follows" in problems[-1]
    out = reconcile_kans(inds, logs, [], {}, entry, 0, [], tsumo_winner="S")
    assert len(out) == 1 and out[0].seat == "S" and out[0].type == "ankan" and out[0].t_window[0] > 25


def test_camera_kan_explains_the_indicator_inside_its_window():
    # the meld camera first shows the ankan 60 s after the indicator, but its window (last view without it) opens before
    kan = Call("S", 100, (30, 100), "ankan", ["3m", "3m", "X", "X"], None, None, None, [], 0.9)
    logs = {"E": [pslot(0, "1z", 15), pslot(2, "3z", 50)], "S": [pslot(1, "2z", 25), pslot(3, "4z", 45)], "W": [], "N": []}
    inds = [{"tile": "1s", "t_first": 0}, {"tile": "4p", "t_first": 40}]
    entry = {"corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}, "corner_site": {"TL": "EAST", "TR": "SOUTH", "BR": "WEST", "BL": "NORTH"}}
    out = reconcile_kans(inds, logs, [kan], {}, entry, 0, [])
    assert out == [kan] and kan.t_window == (32, 40)


def test_an_anchored_kan_stands_without_an_indicator():
    """The second VOD's hand 3: the dora lies at the crop's edge and the ankan's indicator outside every region.
    A kan the call anchor established stands; an indicator it does not explain adds the kan the cameras missed."""
    logs = {"E": [pslot(0, "1z", 15)], "S": [], "W": [], "N": []}
    entry = {"corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}, "corner_site": {"TL": "EAST", "TR": "SOUTH", "BR": "WEST", "BL": "NORTH"}}
    kan = Call("S", 30, (20, 30), "ankan", ["1s", "1s", "X", "X"], None, None, None, [], 0.9, 0, 5, anchor="kan")
    assert reconcile_kans([{"tile": "4s", "t_first": 0}], logs, [kan], {}, entry, 0, []) == [kan]


def test_facts_for_hand_passes_the_new_kinds():
    entry = {"game": 0, "kyoku": 1, "honba": 0, "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}, "corner_site": {"TL": "SOUTH", "TR": "WEST", "BR": "NORTH", "BL": "EAST"}}
    facts = [{"game": 0, "kyoku": 1, "honba": 0, "kind": "draw", "corner": "TR", "j": 7, "t": None, "tile": "3m"},
             {"game": 0, "kyoku": 1, "honba": 0, "kind": "lost", "corner": "TR", "j": 2, "t": 300.0},
             {"game": 0, "kyoku": 1, "honba": 0, "kind": "riichi_turn", "corner": "TL", "t": 250.0},
             {"game": 0, "kyoku": 1, "honba": 0, "kind": "kan_time", "t": 260.0},
             {"game": 0, "kyoku": 1, "honba": 0, "kind": "meld_remove", "corner": "BL", "t": 270.0, "type": "kakan"},
             {"game": 0, "kyoku": 2, "honba": 0, "kind": "draw", "corner": "TR", "j": 1, "tile": "9m"}]
    out = facts_for_hand(facts, entry)
    assert out["draw"] == [{"seat": "S", "t": None, "tile": "3m", "j": 7}]
    assert out["lost"] == [{"seat": "S", "t": 300.0, "j": 2}]
    assert out["riichi_turn"] == [{"seat": "E", "t": 250.0}]
    assert out["kan_time"] == [{"seat": None, "t": 260.0}]
    assert out["meld_remove"] == [{"seat": "N", "t": 270.0, "type": "kakan"}]


# ---------------------------------------------------------------- decode_hand end to end (synthetic ponds)

def pond_obs(region, t0, t1, tiles, sideways=(), indicators=()):
    def sd(tile, r, c, side):
        p = np.full(len(CLASSES), 0.002)
        p[CLASS_INDEX[tile]] = 0.9
        p /= p.sum()
        return {"key": [r, c], "tile": tile, "conf": 0.9, "seen": 3, "sideways": 1.0 if side else 0.0, "disagree": False,
                "xyxy": [c * 40, r * 60, c * 40 + 38, r * 60 + 58], "p": p.tolist()}
    slots = [sd(t, i // 6, i % 6, i in sideways) for i, t in enumerate(tiles)]
    ind = [{**sd(t, 0, 0, False), "xyxy": [400, 10, 438, 68]} for t in indicators]
    return {"region": region, "t0": t0, "t1": t1, "n_readings": 3, "n_used": 3, "count": len(slots), "quality": 0.9,
            "slots": slots, "indicators": ind}


def synthetic_hand(riichi_seat=None):
    """Four ponds, three discards each in rotation from 20 s on, one dora indicator; no hands or melds seen."""
    corners = {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}
    discards = {"E": ["1z", "2z", "3z"], "S": ["4z", "5z", "6z"], "W": ["1m", "2m", "3m"], "N": ["9p", "8p", "7p"]}
    obs = {}
    for corner, seat in corners.items():
        k = "ESWN".index(seat)
        seq = [pond_obs(f"pond:{corner}", 0, 5, [], indicators=["1s"] if corner == "TL" else [])]
        for n in range(1, 4):
            t = 20 + 40 * (n - 1) + 10 * k + 2
            seq.append(pond_obs(f"pond:{corner}", t, t + 6, discards[seat][:n], indicators=["1s"] if corner == "TL" else []))
        seq.append(pond_obs(f"pond:{corner}", 150, 155, discards[seat], indicators=["1s"] if corner == "TL" else []))
        seq.append(pond_obs(f"pond:{corner}", 170, 175, []))
        obs[f"pond:{corner}"] = seq
    entry = {"hand": 0, "game": 0, "kyoku": 0, "honba": 0, "sticks": 0, "t_start": 0, "t_end": 180, "t_overlay": [10, 160],
             "corner_wind": corners, "corner_site": {c: {"E":"EAST","S":"SOUTH","W":"WEST","N":"NORTH"}[w] for c, w in corners.items()}, "scores": {s: 25000 for s in "ESWN"}}
    result = HandResult(0, 0, 0, {"EAST": 0, "SOUTH": 0, "WEST": 0, "NORTH": 0}, "draw",
                        riichi=[{"E": "EAST", "S": "SOUTH", "W": "WEST", "N": "NORTH"}[riichi_seat]] if riichi_seat else [])
    return entry, obs, result


def test_decode_hand_returns_ura_and_places_a_riichi_turn_fact():
    from video2tenhou.engine.decode import decode_hand
    entry, obs, result = synthetic_hand(riichi_seat="S")
    d = decode_hand(entry, obs, result, {"ura": ["2s"]}, time_limit=5, log=lambda *a: None)
    assert d["ura"] == ["2s"] and d["dora"] == ["1s"]
    # no turned tile: the last discard of S is the guess and the reviewer is asked
    assert [t["seat"] for t in d["turns"]] == list("ESWN") * 3
    assert [t["i"] for t in d["turns"] if t["riichi"]] == [9] and any(it["kind"] == "riichi" for it in d["items"])
    # the reviewer names S's second discard (at 72 s): that turn alone is the riichi, nothing is asked
    t_second = next(t["t"] for t in d["turns"] if t["seat"] == "S" and t["j"] == 1)
    d2 = decode_hand(entry, obs, result, {"riichi_turn": [{"seat": "S", "t": t_second + 1.0}]}, time_limit=5, log=lambda *a: None)
    assert [t["i"] for t in d2["turns"] if t["riichi"]] == [5] and not any(it["kind"] == "riichi" for it in d2["items"])


def test_decode_hand_places_a_named_indicator_with_a_kan_time_fact():
    from video2tenhou.engine.decode import decode_hand
    entry, obs, result = synthetic_hand()
    # the reviewer names a second indicator no frame shows: its kan has no time and is asked for
    d = decode_hand(entry, obs, result, {"dora": ["1s", "4p"]}, time_limit=5, log=lambda *a: None)
    assert d["dora"] == ["1s", "4p"] and any(it["kind"] == "kan" and "Which discard" in it["text"] for it in d["items"])
    assert d["stats"]["calls"] == 0
    # the reviewer clicks W's second discard: a kan by W just before it, its tile chosen by the solver and written
    t_w = next(t["t"] for t in d["turns"] if t["seat"] == "W" and t["j"] == 1)
    d2 = decode_hand(entry, obs, result, {"dora": ["1s", "4p"], "kan_time": [{"seat": None, "t": t_w}]}, time_limit=5, log=lambda *a: None)
    assert not any(it["kind"] == "kan" and "Which discard" in it["text"] for it in d2["items"])
    kan = next(t for t in d2["turns"] if t["seat"] == "W" and t["j"] == 1)
    assert kan["kind"] == "kan" and kan["own_call"]["type"] == "ankan" and "?" not in kan["own_call"]["tiles"]
    assert d2["calls"][0]["tiles"] == kan["own_call"]["tiles"] and kan["draw2"] is not None


def test_concealed_size_counts_melds_the_way_the_hand_does():
    """13 tiles, three out per meld set; a kan's fourth tile is the one the kan adds, and a kakan is the
    pon it grew from, not a second set. The winning tile is never in the count."""
    from video2tenhou.engine.decode import concealed_size
    assert concealed_size([]) == 13
    assert concealed_size([{"type": "pon", "tiles": ["1m"] * 3}]) == 10
    assert concealed_size([{"type": "chi", "tiles": ["1m", "2m", "3m"]},
                           {"type": "pon", "tiles": ["5p"] * 3}]) == 7
    assert concealed_size([{"type": "kan", "tiles": ["9s"] * 4}]) == 10
    assert concealed_size([{"type": "ankan", "tiles": ["5z"] * 4}]) == 10
    # a kakan is the pon it grew from: the two entries are one set
    assert concealed_size([{"type": "pon", "tiles": ["2p"] * 3}, {"type": "kakan", "tiles": ["2p"] * 4}]) == 10
    # a tsumo whose winning tile was never named leaves the draw in the list
    assert concealed_size([], drawn_left_in=True) == 14
    assert concealed_size([{"type": "pon", "tiles": ["1m"] * 3}], drawn_left_in=True) == 11


def _stub_decoder(calls, dora, *, facts=None):
    """A decoder at the point after the solve, holding only what kan_indicators reads (no win: a draw)."""
    from types import SimpleNamespace
    from video2tenhou.engine.decode import HandDecoder
    d = HandDecoder.__new__(HandDecoder)
    d.live_calls, d.dora, d.ura, d.facts = calls, list(dora), [], facts or {}
    d.inds = [{"tile": x, "t_first": 30.0 + i, "region": "pond:TL"} for i, x in enumerate(dora)]
    d.sol = SimpleNamespace(ok=False, haipai={"E": ["1s"] * 3 + ["2s"]}, draws={}, draws2={})
    d.winner, d.result = None, HandResult("draw", None, None, None, None, [0, 0, 0, 0], [])
    d.items, d.problems, d.last_discard, d.t1 = [], [], 300.0, 320.0
    return d


def test_a_kan_whose_indicator_no_view_shows_is_asked_and_the_log_keeps_a_guess():
    ankan = Call("N", 200.0, (195.0, 200.0), "ankan", ["1s", "1s", "X", "X"], None, None, None, [], 0.5)
    d = _stub_decoder([ankan], ["4s"])
    d.kan_indicators()
    assert len(d.dora) == 2 and d.dora[0] == "4s"
    guess = d.dora[1]
    assert guess != "1s" and d.inds[-1]["lost"] and d.inds[-1]["tile"] == guess      # 1s: every copy is placed
    (item,) = d.items
    assert item["kind"] == "dora" and item["tiles"] == ["4s"] and item["guess"] == [guess] and item["t"] == 200.0
    # the reviewer cannot tell: the guess stays, nothing is asked
    d = _stub_decoder([ankan], ["4s"], facts={"lost_dora": True})
    d.kan_indicators()
    assert len(d.dora) == 2 and not d.items and d.inds[-1]["human"]


def test_a_lost_dora_fact_reaches_the_decoder():
    entry = {"game": 0, "kyoku": 2, "honba": 0, "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}}
    f = facts_for_hand([{"game": 0, "kyoku": 2, "honba": 0, "kind": "lost", "field": "dora", "t": 200.0}], entry)
    assert f["lost_dora"] and not f["lost"]


def test_a_draw_nothing_covers_is_unseen():
    from types import SimpleNamespace
    from video2tenhou.engine.review import unseen_draw
    sol = SimpleNamespace(margins={("E", 1): 0.1, ("E", 2): 3.0})
    model = SimpleNamespace(draw_ev=[], hand_ev=[])
    assert unseen_draw(model, sol, "E", 1) and not unseen_draw(model, sol, "E", 2)
    model.draw_ev = [SimpleNamespace(seat="E", j=1)]
    assert not unseen_draw(model, sol, "E", 1)


def test_a_read_floor_row_showing_the_whole_hand_between_turns_is_the_state():
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 100, 120)]
    thirteen = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    o = {**hobs(40, 60, thirteen), "partial": True}
    hev, _, _ = hand_evidence("S", st, [o], {}, dealer=False)
    assert [(e.j, e.after_draw, e.subset, e.hidden) for e in hev] == [(0, False, False, 0)]


def test_a_calm_row_one_tile_short_between_turns_holds_all_but_one():
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 100, 120)]
    twelve = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z"]
    hev, _, _ = hand_evidence("S", st, [hobs(40, 60, twelve)], {}, dealer=False)
    assert [(e.j, e.subset, e.hidden) for e in hev] == [(0, False, 1)]
    # a read-floor view that short is a sub-multiset only (an arm may hide more than the one)
    hev, _, _ = hand_evidence("S", st, [{**hobs(40, 60, twelve), "partial": True}], {}, dealer=False)
    assert [(e.subset, e.hidden) for e in hev] == [(True, 0)]
    # and it does not cover the draw by itself: the hidden tile may be the drawn one
    from types import SimpleNamespace
    from video2tenhou.engine.review import unseen_draw
    model = SimpleNamespace(draw_ev=[], hand_ev=hand_evidence("S", st, [hobs(40, 60, twelve)], {}, dealer=False)[0])
    assert unseen_draw(model, SimpleNamespace(margins={("S", 1): 0.1}), "S", 1)


def test_a_misread_run_of_three_is_a_fragment_the_discard_can_anchor():
    # 4p 3p 0p laid, the 4p read 2p (no legal meld as read): the surest two of the run make the fragment
    from video2tenhou.engine.melds import fragments
    g = lambda: [mslot("2p", 0, 0), mslot("3p", 0, 1), mslot("0p", 0, 2)]
    seq = [mobs(0, 5, []), mobs(10, 14, g()), mobs(20, 24, g())]
    (f,) = fragments("E", seq, track_melds("E", seq))
    assert f.t_first == 10 and len(f.ps) == 3 and sorted(f.tiles) in (["2p", "3p"], ["3p", "5p"])


# ---------------------------------------------------------------- dense still runs

def _dframe(t, tiles, flicker=None):
    def box(x, tile):
        p = np.full(len(CLASSES), 0.001)
        p[CLASS_INDEX[tile]] = 0.95
        return {"role": "tile", "xyxy": [x * 40, 0, x * 40 + 38, 58], "p": (p / p.sum()).tolist(), "conf": 0.95}
    tiles = list(tiles)
    if flicker is not None:
        tiles[flicker[0]] = flicker[1]
    return {"t": t, "boxes": [box(x, tl) for x, tl in enumerate(tiles)]}


def test_still_runs_are_the_frames_whose_reading_holds():
    from video2tenhou.engine.dense import still_runs
    row = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    frames = ([_dframe(10 + 0.2 * k, row, flicker=(4, "6p") if k == 2 else None) for k in range(6)]    # one flicker: still
              + [_dframe(11.2 + 0.2 * k, row[:k % 5 + 8]) for k in range(4)]                         # moving: no run
              + [_dframe(12.0 + 0.2 * k, row + ["2z"]) for k in range(5)])                           # the drawn tile
    runs = still_runs(frames, "hand:TL")
    assert [(r["count"], r["n_used"], r["t0"]) for r in runs] == [(13, 6, 10.0), (14, 5, 12.0)]


def test_dense_draws_read_the_hand_before_and_after_the_turn(monkeypatch):
    from types import SimpleNamespace
    from video2tenhou.engine import dense
    before = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    after = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "2z"]      # drew 2z, discarded 9m
    frames = ([_dframe(22 + 0.2 * k, before) for k in range(10)]           # at rest before the draw
              + [_dframe(55 + 0.2 * k, after) for k in range(10)])         # at rest after the discard
    monkeypatch.setattr(dense, "dense_reads", lambda *a, **k: {"hand:TL": [f for f in frames if a[5] <= f["t"] <= a[6]]})
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 40, 50)]
    model = SimpleNamespace(turns={"S": st}, hand_ev=[], draw_ev=[], dealer="E", tsumo_winner=None, end_prior={})
    turns = [Turn(0, "S", "draw", None, 20.0), Turn(1, "W", "draw", None, 30.0), Turn(2, "S", "draw", None, 50.0),
             Turn(3, "W", "draw", None, 60.0)]
    entry = {"corner_wind": {"TL": "S", "TR": "W", "BL": "E", "BR": "N"}}
    got = dense.draws([("S", 1)], model, turns, {"S": {}}, entry, {}, (None, None, None, None), None, 0.0, [])
    assert got == 1
    assert sorted((e.j, e.subset, e.hidden) for e in model.hand_ev) == [(0, False, 0), (1, False, 0)]


# ---------------------------------------------------------------- the site's han/fu can be wrong

def test_han_fu_that_pay_the_same_differ_only_on_paper():
    from video2tenhou.engine.scoring import payment
    # hand 19 of the second VOD: 10/40 against the site's 9/70, a baiman either way
    assert payment(10, 40, dealer=False, tsumo=False) == payment(9, 70, dealer=False, tsumo=False)
    assert payment(4, 40, dealer=True, tsumo=True) == payment(5, 30, dealer=True, tsumo=True)        # both mangan
    assert payment(3, 30, dealer=False, tsumo=False) != payment(3, 40, dealer=False, tsumo=False)


def test_a_site_wrong_fact_replaces_the_sites_han_fu_and_keeps_the_deltas():
    from video2tenhou.engine.decode import HandDecoder
    entry = {"game": 1, "kyoku": 3, "honba": 1, "hand": 19, "corner_wind": {"TL": "S", "TR": "E", "BL": "W", "BR": "N"},
             "corner_site": {"TL": "EAST", "TR": "NORTH", "BL": "SOUTH", "BR": "WEST"}}
    fact = {"game": 1, "kyoku": 3, "honba": 1, "kind": "site_wrong", "corner": "BL", "han": 10, "fu": 40, "site": [9, 70]}
    facts = facts_for_hand([fact], entry)
    assert facts["site_score"] == {"han": 10, "fu": 40}
    deltas = {"EAST": 0, "SOUTH": 17300, "WEST": 0, "NORTH": -16300}
    result = HandResult(3, 1, 1, deltas, "ron", "SOUTH", "NORTH", 9, 70)
    d = HandDecoder(entry, {}, result, facts, time_limit=1.0, models=None, work_dir=None)
    assert (d.result.han, d.result.fu, d.result.deltas) == (10, 40, deltas) and d.site_han_fu == (9, 70)
    assert any("the site's 9/70 is wrong" in p for p in d.problems)


def test_the_log_is_written_with_the_confirmed_han_fu_and_the_sites_deltas():
    d = {"dealer": "E", "haipai": {"E": ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "6z", "9p"],
                                   "S": [], "W": [], "N": []},
         "turns": [{"i": 0, "seat": "E", "j": 0, "kind": "draw", "t": 10, "riichi": False, "draw": None, "draw2": None,
                    "discard": "9p", "tsumogiri": False, "margin": None, "call": None, "own_call": None}],
         "draws": {}, "dora": ["1s"], "ura": [],
         "result": {"winner": "S", "loser": "E", "han": 10, "fu": 40, "site": [9, 70], "site_wrong": True},
         "score": {"match": True, "yaku": ["Chinitsu (6)", "Dora 4 (4)"]}}
    entry = {"kyoku": 0, "honba": 0, "sticks": 0, "scores": {"E": 25000, "S": 25000, "W": 25000, "N": 25000}}
    result = HandResult(0, 0, 0, {"EAST": -16000, "SOUTH": 16000, "WEST": 0, "NORTH": 0}, "ron", "SOUTH", "EAST", 9, 70)
    k, _ = kyoku_from_decode(d, entry, result)
    (win,) = k.result.wins
    assert win.score_text == "倍満16000点" and win.delta == [-16000, 16000, 0, 0] and win.yaku == ["清一色(6飜)", "ドラ(4飜)"]
def test_uncertain_raw_discard_gets_question_without_duplicate_or_fact_override():
    from video2tenhou.engine.review import uncertain_discards
    row = {"field": "discard", "seat": "S", "turn": 17, "value": "2m", "runner_up": "1m",
           "margin": .5, "human": False, "evidence": [{"region": "pond:BR", "t": 3097.5}]}
    items = uncertain_discards([row], [])
    assert len(items) == 1 and items[0]["kind"] == "discard"
    assert items[0]["tile"] == "2m" and items[0]["runner_up"] == "1m"
    assert items[0]["t"] == 3097.5
    assert not uncertain_discards([row], items)
    assert not uncertain_discards([{**row, "human": True}], [])
    assert not uncertain_discards([{**row, "margin": .5001}], [])
    assert not uncertain_discards([{**row, "margin": None}], [])


@pytest.mark.parametrize('kind', ['discard', 'missing_discard'])
def test_reviewed_discard_remains_fixed_during_repair(kind):
    from types import SimpleNamespace
    import numpy as np
    from video2tenhou.engine.decode import HandDecoder
    from video2tenhou.engine.solver import HandModel, SeatTurn, TI, TILES
    from video2tenhou.engine import rules
    turns = {s: [] for s in rules.SEATS}
    p = np.zeros(len(TILES))
    p[TI['1m']] = 1
    turns['S'] = [SeatTurn(0, 'draw', '1m', 0, 10, discard_p=p)]
    model = HandModel('E', turns, [])
    decoder = SimpleNamespace(facts={kind: [{'seat': 'S', 't': 10, 'tile': '1m'}]}, problems=[])
    HandDecoder._apply_hand_facts(decoder, model)
    model.repair = True
    model.build()
    assert turns['S'][0].discard_p is None
    assert model._x[('S', 0)] == {TI['1m']: 1}
    # If no starting tile or draw can supply the reviewed discard, repair
    # must expose the contradiction rather than paying to change the fact.
    model.facts.haipai['S'] = ['2m', '3m', '4m', '4p', '5p', '6p', '7s', '8s', '9s', '1z', '1z', '3z', '6z']
    model.facts.draws[('S', 0)] = '9m'
    assert not model.solve(margins=False, workers=1).ok


def test_confidence_reports_repaired_model_discard_when_final_solve_has_no_override():
    from types import SimpleNamespace
    from video2tenhou.engine.review import confidence_rows
    from video2tenhou.engine.solver import HandModel, SeatTurn, Solution
    from video2tenhou.engine import rules

    turns = {s: [] for s in rules.SEATS}
    turns['S'] = [SeatTurn(0, 'draw', '1m', 0, 10)]
    model = HandModel('E', turns, [])
    slot = SimpleNamespace(tile='2m', conf=.93)
    turn = SimpleNamespace(seat='S', t=10, slot=slot, virtual=False)
    # The repaired identity is now fixed, so the final solve records no
    # variable-discard override. The original pond reading remains evidence.
    sol = Solution('repaired', 0, {}, {}, {})
    rows = confidence_rows(model, sol, [turn], [], [], {'corner_wind': {'BR': 'S'}}, {}, set(), 0)
    assert len(rows) == 1
    assert rows[0]['field'] == 'discard' and rows[0]['value'] == '1m'
    assert not rows[0]['human']
    assert slot.tile == '2m'
    facts = {'missing_discard': [{'seat': 'S', 't': 10, 'tile': '1m'}]}
    reviewed = confidence_rows(model, sol, [turn], [], [], {'corner_wind': {'BR': 'S'}}, facts, set(), 0)
    assert reviewed[0]['human'] and reviewed[0]['value'] == '1m'
