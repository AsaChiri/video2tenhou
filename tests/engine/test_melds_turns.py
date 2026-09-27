import numpy as np

from video2tenhou.engine import rules
from video2tenhou.engine.melds import decode_group, track_melds
from video2tenhou.engine.ponds import PondSlot
from video2tenhou.engine.turns import merge
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def slot(tile, group, i, sideways=0.0, second=None):
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.8
    if second:
        p[CLASS_INDEX[second]] = 0.3
    p /= p.sum()
    return {"key": [group, i], "tile": tile, "conf": 0.9, "seen": 3, "sideways": sideways, "disagree": False,
            "xyxy": [i * 40, group * 70, i * 40 + 38, group * 70 + 58], "p": p.tolist()}


def obs(t0, t1, slots):
    return {"region": "meld:TL", "t0": t0, "t1": t1, "n_readings": 3, "n_used": 3, "count": len(slots), "quality": 0.9,
            "slots": slots, "indicators": []}


def test_rules():
    assert rules.is_chi(["3m", "4m", "5m"]) and rules.is_chi(["0m", "4m", "6m"])
    assert not rules.is_chi(["3m", "4m", "5p"]) and not rules.is_chi(["1z", "2z", "3z"])
    assert rules.is_pon(["5s", "0s", "5s"]) and not rules.is_pon(["5s", "5s", "6s"])
    assert rules.relative("E", "N") == "kamicha" and rules.relative("E", "S") == "shimocha" and rules.relative("E", "W") == "toimen"
    assert rules.count_ok(["5m"] * 3 + ["0m"]) and not rules.count_ok(["5m"] * 4)


def test_decode_group_pon_rule_repairs_one_tile():
    g = [slot("9p", 0, 0), slot("9p", 0, 1, sideways=1.0), slot("5s", 0, 2, second="9p")]
    m = decode_group(g)
    assert m.type == "pon" and m.tiles == ["9p", "9p", "9p"] and m.called_pos == 1
    m = decode_group([slot("3s", 0, 0, sideways=1.0), slot("4s", 0, 1), slot("5s", 0, 2)])
    assert m.type == "chi" and m.called_pos == 0
    assert decode_group([slot("1m", 0, 0), slot("5p", 0, 1), slot("9s", 0, 2)]) is None


def test_four_boxes_are_a_kan_only_when_all_four_read_so():
    """The second VOD's hand 22: the chi 2m-3m-4m with its 4m boxed twice was read as four 4m."""
    m = decode_group([slot("2m", 0, 0, sideways=1.0), slot("4m", 0, 1), slot("4m", 0, 2), slot("3m", 0, 3)])
    assert m.type == "chi" and sorted(m.tiles) == ["2m", "3m", "4m"] and m.tiles[m.called_pos] == "2m"
    # the reference VOD's hand 7: a pon of 9m beside the 7z of the next meld
    m = decode_group([slot("9m", 0, 0), slot("9m", 0, 1), slot("7z", 0, 2), slot("9m", 0, 3, sideways=1.0)])
    assert m.type == "pon" and m.tiles == ["9m", "9m", "9m"]
    m = decode_group([slot("6z", 0, 0, sideways=1.0), slot("6z", 0, 1), slot("6z", 0, 2), slot("6z", 0, 3)])
    assert m.type == "kan" and m.called_pos == 0


def test_a_long_run_of_boxes_is_several_melds():
    from video2tenhou.engine.melds import split_group
    run = [slot("8p", 0, 0, second="9p"), slot("8p", 0, 1), slot("7p", 0, 2), slot("6s", 0, 3, sideways=1.0), slot("8s", 0, 4), slot("7s", 0, 5)]
    melds = split_group(run)
    assert [m.type for m in melds] == ["chi", "chi"]
    assert [m.tiles[m.called_pos] for m in melds][1] == "6s"


def test_a_known_meld_with_a_stray_box_is_the_same_meld():
    chi = [slot("2m", 0, 0, sideways=1.0), slot("4m", 0, 1), slot("3m", 0, 2)]
    seq = [obs(0, 5, []), obs(10, 14, chi), obs(20, 24, chi),
           obs(30, 34, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0),
                        slot("2m", 1, 0, sideways=1.0), slot("4m", 1, 1), slot("4m", 1, 2), slot("3m", 1, 3)]),
           obs(40, 44, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0),
                        slot("2m", 1, 0, sideways=1.0), slot("4m", 1, 1), slot("4m", 1, 2), slot("3m", 1, 3)])]
    calls = track_melds("S", seq)
    assert sorted(c.type for c in calls) == ["chi", "pon"]


def test_track_melds_sources_and_kakan():
    seq = [
        obs(0, 5, []),
        obs(10, 14, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0)]),
        obs(20, 24, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0)]),
        obs(30, 34, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0), slot("7z", 0, 3),
                     slot("2m", 1, 0, sideways=1.0), slot("3m", 1, 1), slot("4m", 1, 2)]),
        obs(40, 44, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0), slot("7z", 0, 3),
                     slot("2m", 1, 0, sideways=1.0), slot("3m", 1, 1), slot("4m", 1, 2)]),
    ]
    calls = track_melds("E", seq)
    # the pon stays (its called tile fixes a turn); the kakan is a second event timed by its fourth tile
    assert sorted(c.type for c in calls) == ["chi", "kakan", "pon"]
    pon, chi, kakan = (next(c for c in calls if c.type == t) for t in ("pon", "chi", "kakan"))
    assert pon.source == "shimocha" and pon.t_window == (5, 10)
    assert chi.source == "kamicha" and chi.called_tile == "2m" and chi.t_first == 30
    assert kakan.t_first == 30 and kakan.t_window == (24, 30) and kakan.source == "shimocha" and len(kakan.tiles) == 4


def pslot(i, tile, t, removed=None, side=False):
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.9
    s = PondSlot(i, 0, i, p, t, (t - 3, t), t + 4, 3, 3.0 if side else 0.0)   # sideways accumulates one vote per observation
    s.t_removed = removed
    return s


def test_merge_follows_rotation_and_calls():
    from video2tenhou.engine.melds import Call
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
            "S": [pslot(2, "3p", 20, removed=27), pslot(3, "4p", 60)],
            "W": [pslot(4, "5s", 30, side=True), pslot(5, "6s", 70)],
            "N": [pslot(6, "7z", 40)]}
    # W pons S's 3p at ~27: W discards next (no draw), then N, E, S, W, ...
    pon = Call("W", 28, (24, 28), "pon", ["3p", "3p", "3p"], 0, "kamicha", "3p", [], 0.9)
    turns, problems = merge(logs, [pon], "E")
    assert problems == []
    assert [(t.seat, t.kind) for t in turns] == [("E", "draw"), ("S", "draw"), ("W", "call"), ("N", "draw"), ("E", "draw"),
                                                  ("S", "draw"), ("W", "draw")]
    assert turns[1].call is pon and turns[2].riichi


def test_merge_drops_a_phantom_instead_of_dragging_the_hand():
    """One slot at a time no discard could have (a tile of the next hand, a misread): it costs one drop, and
    every other turn keeps its place."""
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 50), pslot(8, "9p", 400)],
            "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
            "W": [pslot(4, "5s", 30), pslot(5, "6s", 70)],
            "N": [pslot(6, "7z", 40), pslot(7, "6z", 80)]}
    logs["E"].insert(1, pslot(9, "1z", 390))                # a tile laid far out of time, second in E's log
    turns, problems = merge(logs, [], "E")
    assert [t.seat for t in turns] == list("ESWNESWNE")
    assert any("1z seen at 390s is not a discard of this hand" in p for p in problems)


def test_a_seat_that_runs_out_of_discards_is_a_skip_not_a_free_pass():
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
            "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
            "W": [pslot(4, "5s", 30)],                       # W's second discard was never read
            "N": [pslot(6, "7z", 40), pslot(7, "6z", 80)]}
    turns, problems = merge(logs, [], "E")
    assert [t.seat for t in turns] == list("ESWNESN")
    assert any("no discard of W" in p for p in problems)


def test_the_result_fixes_where_the_sequence_ends():
    """After a tsumo the winner is next to play: a trailing tile that puts another seat there is dropped, and a
    sequence that cannot end at the winner says so."""
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
            "S": [pslot(2, "3p", 20), pslot(3, "4p", 60), pslot(8, "9m", 75)],   # 9m: after the win, the reveal
            "W": [pslot(4, "5s", 30)],
            "N": [pslot(6, "7z", 40)]}
    turns, problems = merge(logs, [], "E", end="W")
    assert [t.seat for t in turns] == list("ESWNES") and any("9m seen at 75s" in p for p in problems)
    logs["S"].pop()
    turns, problems = merge(logs, [], "E", end="N")
    assert any("ends with W to play, but the result says N" in p for p in problems)


def test_a_removal_no_call_took_is_a_hidden_call():
    """Hand 0 of the second VOD: S discards twice with nobody between; W's 2z had left W's pond. Every removal is
    a call: the turn order says S took it (a pon from its shimocha), which the camera missed."""
    logs = {"E": [pslot(0, "1m", 10), pslot(1, "2m", 90)],
            "S": [pslot(2, "3p", 20), pslot(3, "4p", 34), pslot(9, "5p", 100)],
            "W": [pslot(4, "2z", 28, removed=31), pslot(5, "5s", 60)],
            "N": [pslot(6, "7z", 70)]}
    turns, problems = merge(logs, [], "E")
    assert [t.seat for t in turns][:4] == ["E", "S", "W", "S"]
    hidden = turns[2].call
    assert hidden is not None and hidden.anchor == "hidden" and hidden.seat == "S" and hidden.source == "shimocha"
    assert turns[3].own_call is hidden and not any("skipped" in p for p in problems)


def test_meld_options_come_from_the_called_tile():
    """The second VOD's hand 23: a chi on the red five from the kamicha, read by the camera as 0p 4p 3p."""
    from video2tenhou.engine.melds import meld_options
    ps = [np.asarray(slot(t, 0, i)["p"]) for i, t in enumerate(["0p", "4p", "3p"])]
    best = meld_options("0p", "kamicha", ps, four=False)[0]
    assert best.type == "chi" and sorted(best.hand) == ["3p", "4p"] and best.cost == 0
    # from the toimen only a pon (or, with four boxes, a daiminkan) is possible
    assert {o.type for o in meld_options("4p", "toimen", ps, four=False)} == {"pon"}
    assert {o.type for o in meld_options("4p", "toimen", ps, four=True)} == {"kan"}


def test_fragments_are_pairs_of_a_meld_the_camera_does_not_complete():
    """Hand 0 of the second VOD: S's camera shows "2z 2z 3z" (no legal meld: a tile misread) and hand 3's shows
    "1s 1s" (the turned tile not boxed): both are pairs of a pon for the pond or the turn order to complete."""
    from video2tenhou.engine.melds import fragments
    views = [obs(t, t + 3, [slot("2z", 0, 0), slot("2z", 0, 1), slot("3z", 0, 2)]) for t in (10, 20, 30)]
    views += [obs(t, t + 3, [slot("2z", 0, 0), slot("2z", 0, 1), slot("3z", 0, 2), slot("1s", 1, 0), slot("1s", 1, 1)])
              for t in (40, 50)]
    frags = {tuple(f.tiles): (f.t_first, f.seen) for f in fragments("S", views, [])}
    assert frags == {("2z", "2z"): (10, 5), ("1s", "1s"): (40, 2)}
    # a pair that is part of a meld the seat is known to have is that meld, not a fragment
    from video2tenhou.engine.melds import Call
    known = Call("S", 5, (0, 5), "pon", ["1s"] * 3, 2, "shimocha", "1s", [], 0.9)
    assert [f.tiles for f in fragments("S", views, [known])] == [["2z", "2z"]]


def test_an_ankan_is_all_four_of_its_kind():
    """The second VOD's hand 3: N chi'd W's 5m, then N's camera showed a 5m 5m pair with nothing taken. An ankan
    of 5m would need a fifth five: the pair was the chi re-read, not a kan. A kan the camera saw whole whose
    kind a pond shows keeps its place and loses its kind."""
    from video2tenhou.engine.calls import CallAnchor
    from video2tenhou.engine.melds import Call
    logs = {"E": [pslot(0, "1m", 10)], "S": [pslot(1, "9p", 20)], "W": [pslot(2, "5m", 30, removed=33)], "N": [pslot(3, "7z", 40)]}
    ps = [np.asarray(slot(t, 0, i)["p"]) for i, t in enumerate(["5m", "4m", "6m"])]
    chi = Call("N", 35, (30, 35), "chi", ["5m", "4m", "6m"], 0, "kamicha", "5m", ps, 0.9, 0, 5)
    pair = Call("N", 70, (60, 70), "fragment", ["5m", "5m"], None, None, None, ps[:2], 0.9, 0, 3)
    whole = Call("E", 80, (75, 80), "ankan", ["9p", "9p", "X", "X"], None, None, None, [], 0.9, 0, 3)
    problems: list[str] = []
    calls = CallAnchor(logs, {}, None, None, 0, problems).anchor([chi, pair, whole], 10, 90, None)
    assert [(c.seat, c.type) for c in calls] == [("N", "chi"), ("E", "ankan")]
    assert calls[0].anchor == "discard" and calls[0].source == "kamicha" and calls[0].called_tile == "5m"
    assert calls[1].tiles[:2] == ["?", "?"] and any("misread" in p for p in problems)
    assert any("no kan" in p for p in problems)


def test_an_exhaustive_draw_ends_with_the_last_draw():
    """The second VOD's hand 7: the ponds hold one discard more than the live wall allows; the last, seen once
    during the reveal, was laid after the hand ended. With a wall of five draws, the fifth discard ends it."""
    logs = {"E": [pslot(0, "1m", 10), pslot(4, "5m", 50)], "S": [pslot(1, "2m", 20), pslot(5, "6m", 60)],
            "W": [pslot(2, "3m", 30)], "N": [pslot(3, "4m", 40)]}
    turns, problems = merge(logs, [], "E", None, wall=5, exhaustive=True)
    assert [t.seat for t in turns] == ["E", "S", "W", "N", "E"]
    assert any("6m" in p and "after the hand ended" in p for p in problems)
    # a call turn draws nothing: with S's pon of E's first discard the same five draws reach S's second discard
    from video2tenhou.engine.melds import Call
    logs["E"][0].t_removed = 12
    pon = Call("S", 14, (10, 14), "pon", ["1m", "1m", "1m"], 0, "kamicha", "1m", [], 0.9, 0, 5)
    logs["N"][0].t_first = 55
    turns, _ = merge(logs, [pon], "E", None, wall=4, exhaustive=True)
    assert [(t.seat, t.kind) for t in turns] == [("E", "draw"), ("S", "call"), ("W", "draw"), ("N", "draw"), ("E", "draw")]


def test_a_meld_takes_a_discard_of_its_own_suit():
    """The second VOD's hand 22: S's camera read 7z 7z 7z; E's 0m had just left its pond, N's 7z a little earlier.
    A pon is of one suit and the inset confuses numbers, not suits: the 7z is the called tile, not the 0m."""
    from video2tenhou.engine.calls import CallAnchor
    from video2tenhou.engine.melds import Call
    logs = {"E": [pslot(0, "0m", 30, removed=32)], "S": [pslot(1, "6s", 26)], "W": [], "N": [pslot(2, "7z", 18, removed=21)]}
    for sl in logs["N"] + logs["E"]:
        sl.t_last = sl.t_first + 1
    ps = [np.asarray(slot("7z", 0, i)["p"]) for i in range(3)]
    pon = Call("S", 41, (36, 41), "pon", ["7z", "7z", "7z"], 0, None, None, ps, 0.9, 0, 12)
    calls = CallAnchor(logs, {}, None, None, 0, []).anchor([pon], 10, 90, None)
    assert [(c.type, c.called_tile, c.source) for c in calls] == [("pon", "7z", "toimen")]


def test_a_meld_the_hand_does_not_confirm_is_a_reread():
    """The second VOD's hand 19: E's hand fell by three tiles at its chi and held that afterwards; 110 s later
    the inset regrouped the same chi and a 2p flickered in N's pond. The hand holds the tiles of the melds
    already known: the second chi is the camera re-reading the first, and it takes nothing."""
    from video2tenhou.engine.calls import CallAnchor
    from video2tenhou.engine.melds import Call
    entry = {"corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}}
    hand = [{"region": "hand:TL", "t0": t, "t1": t + 2, "n_used": 3, "count": c, "slots": [{}] * c}
            for t, c in ((50, 13), (60, 13), (75, 10), (90, 10), (190, 10), (200, 10))]
    logs = {"E": [], "S": [], "W": [], "N": [pslot(0, "2p", 66, removed=68), pslot(1, "2p", 176, removed=178)]}
    for sl in logs["N"]:
        sl.t_last = sl.t_first + 1
    ps = [np.asarray(slot(t, 0, i)["p"]) for i, t in enumerate(["2p", "3p", "4p"])]
    first = Call("E", 72, (65, 72), "chi", ["2p", "3p", "4p"], 0, None, None, ps, 0.9, 0, 12)
    again = Call("E", 183, (175, 183), "chi", ["2p", "3p", "4p"], 0, None, None, ps, 0.9, 1, 12)
    problems: list[str] = []
    calls = CallAnchor(logs, entry, None, None, 0, problems, {"hand:TL": hand}).anchor([first, again], 10, 300, None)
    assert [(c.seat, c.type, round(c.t_first)) for c in calls] == [("E", "chi", 72)]
    assert any("re-read" in p for p in problems)
