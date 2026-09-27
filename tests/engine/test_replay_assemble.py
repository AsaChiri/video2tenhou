"""The replayer (turn by turn, in the real interleaving) and the writer's result block."""
from video2tenhou import tenhou6 as t
from video2tenhou.engine.assemble import _yaku_text, score_text, tenhou_deltas
from video2tenhou.record import HandResult


def kyoku(haipai, draws, discards, result, head=(0, 0, 0), dora=(21,)):
    k = [list(head), [25000] * 4, list(dora), []]
    for i in range(4):
        k += [list(haipai[i]), list(draws[i]), list(discards[i])]
    return k + [result]


# seat 1 waits on 5p (tanki); seat 0 draws 5p and discards it at once; seat 1 rons
HAIPAI = [t.tiles("111222333444s1z"), t.tiles("123456789m123p5p"), t.tiles("555666777888s2z"), t.tiles("999s3334445556z")]
RON = ["和了", [-1000, 1000, 0, 0], [1, 0, 1, "30符1飜1000点"]]


def test_a_legal_ron_replays_clean():
    k = kyoku(HAIPAI, [[25], [], [], []], [[60], [], [], []], RON)
    assert t.replay_kyoku(k) == []


def test_the_replayer_catches_what_the_per_seat_check_missed():
    # a discard after the hand ended (the ron): never played
    k = kyoku(HAIPAI, [[25], [], [35], []], [[60], [], [60], []], RON)
    assert any("never came to be played" in p for p in t.replay_kyoku(k))
    # after riichi every discard is the drawn tile
    k = kyoku(HAIPAI, [[31, 25], [41], [42], [43]], [["r60", 11], [60], [60], [60]], ["流局", [0, 0, 0, 0]])
    assert any("after riichi" in p for p in t.replay_kyoku(k))
    # deltas that do not balance, and a winner who does not hold a winning hand
    k = kyoku(HAIPAI, [[25], [], [], []], [[60], [], [], []], ["和了", [-1000, 2000, 0, 0], [1, 0, 1, "30符1飜1000点"]])
    assert any("deltas sum" in p for p in t.replay_kyoku(k))
    k = kyoku(HAIPAI, [[26], [], [], []], [[60], [], [], []], RON)
    assert any("does not hold a winning hand" in p for p in t.replay_kyoku(k))
    # a fifth 1s among the hands, the draws and the indicators
    k = kyoku(HAIPAI, [[31], [], [], []], [[60], [], [], []], ["流局", [0, 0, 0, 0]], dora=(31,))
    assert any("1s appears 5 times" in p for p in t.replay_kyoku(k))


def test_a_chi_only_from_the_kamicha():
    # seat 1 chis seat 0's 3p (its kamicha): it discards next without drawing
    haipai = [t.tiles("111222333444s1z"), t.tiles("123456789m12p55p"), t.tiles("555666777888s2z"), t.tiles("999s3334445556z")]
    ok = kyoku(haipai, [[23], ["c231222"], [], []], [[60], [11], [], []], ["流局", [0, 0, 0, 0]])
    assert not any("never came to be played" in p or "out of turn" in p for p in t.replay_kyoku(ok))
    # the same chi string in seat 2's list: seat 0 is not its kamicha, so it is no call on that discard
    bad = kyoku(haipai, [[23], [], ["c231222"], []], [[60], [], [11], []], ["流局", [0, 0, 0, 0]])
    assert any("out of turn" in p or "never came to be played" in p for p in t.replay_kyoku(bad))


def test_value_text_in_tenhou_form():
    assert score_text(2, 30, dealer=False, tsumo=False) == "30符2飜2000点"
    assert score_text(2, 30, dealer=False, tsumo=True) == "30符2飜500-1000点"
    assert score_text(2, 30, dealer=True, tsumo=True) == "30符2飜1000点∀"
    assert score_text(4, 30, dealer=False, tsumo=False) == "満貫8000点"           # kiriage mangan
    assert score_text(6, 30, dealer=True, tsumo=True) == "跳満6000点∀"


def test_the_deposits_leave_the_deltas():
    """The site charges a declarer's 1000 in the hand's deltas; the viewer charges it at the r discard."""
    r = HandResult(0, 0, 0, {"EAST": -2000, "SOUTH": 5000, "WEST": -2000, "NORTH": -1000}, "tsumo", winner="SOUTH",
                   han=3, fu=30, riichi=["WEST"])
    assert tenhou_deltas(r) == [-2000, 5000, -1000, -1000]


def test_yaku_names_of_the_scoring_library():
    assert _yaku_text(["Yakuhai (seat wind east) (1)", "Riichi (1)", "Dora 2 (2)", "Double Riichi (2)", "Suu Ankou (13)"]) == \
        ["自風 東(1飜)", "立直(1飜)", "ドラ(2飜)", "両立直(2飜)", "四暗刻(役満)"]


def test_the_site_fu_counts_below_yakuman():
    """Reference hand 18: a 6/20 pinfu tsumo reconstructed as a 6/30 hand scores the same haneman, but the site
    records the fu, and a different fu means a different hand. Only a yakuman's fu means nothing."""
    from video2tenhou.engine.scoring import ScoreResult, matches_site
    assert matches_site(ScoreResult(True, han=6, fu=20), 6, 20)
    assert not matches_site(ScoreResult(True, han=6, fu=30), 6, 20)
    assert matches_site(ScoreResult(True, han=13, fu=50), 13, 40)
