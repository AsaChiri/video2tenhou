from video2tenhou import timeline as T
from video2tenhou.record import Game, HandResult

C = ("TL", "TR", "BL", "BR")


def reading(t, kyoku, honba, sticks, scores, winds):
    return {"t": t, "kyoku": kyoku, "honba": honba, "sticks": sticks,
            "scores": dict(zip(C, scores)), "winds": dict(zip(C, winds)), "ok": True}


def hand_readings(t0, n, kyoku, honba, scores, winds, sticks=0):
    return [reading(t0 + i, kyoku, honba, sticks, scores, winds) for i in range(n)]


def test_valid_requires_the_checksum_and_four_winds():
    r = reading(0, 0, 0, 1, (25000, 25000, 25000, 24000), "ENSW")
    assert T.valid(r)
    r["scores"]["TL"] = 26000
    assert not T.valid(r)
    r = reading(0, 0, 0, 0, (25000,) * 4, "ENSE")
    assert not T.valid(r)


def test_segment_debounces_and_splits_hanchan():
    rs = hand_readings(0, 100, 0, 0, (25000,) * 4, "ENSW")
    rs += [reading(100, 3, 0, 0, (25000,) * 4, "ENSW")]  # a one-frame glitch
    rs += hand_readings(101, 100, 0, 0, (25000,) * 4, "ENSW")
    rs += hand_readings(201, 100, 1, 0, (26500, 23500, 25500, 24500), "NWES")
    rs[250]["sticks"] = 1; rs[250]["scores"]["TL"] = 25500
    for r in rs[251:301]:
        r["sticks"] = 1; r["scores"]["TL"] = 25500
    rs += hand_readings(301, 100, 0, 0, (25000,) * 4, "SEWN")  # second hanchan
    rs += hand_readings(401, 10, 1, 0, (25000,) * 4, "SEWN")   # too short
    hands = T.segment(rs)
    assert [(h.game, h.kyoku, h.honba) for h in hands] == [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
    assert hands[0].t_start == 0 and hands[0].t_end == 200
    assert hands[1].scores == dict(zip(C, (26500, 23500, 25500, 24500)))
    assert hands[1].sticks == 0 and hands[1].stick_events == [{"t": 250, "sticks": 1}]


def _game(gid, hands):
    return Game(id=gid, players={"EAST": "a", "SOUTH": "b", "WEST": "c", "NORTH": "d"}, final={}, hands=hands)


def test_align_maps_corners_to_seats_and_checks_scores():
    hands = [
        T.Hand(0, 0, 0, 0, 100, 0, dict(zip(C, (25000,) * 4)), dict(zip(C, "ENSW"))),
        T.Hand(0, 1, 0, 101, 200, 0, dict(zip(C, (26500, 23500, 25500, 24500))), dict(zip(C, "NWES"))),
    ]
    game = _game(7, [
        HandResult(0, 0, 0, {"EAST": 1500, "SOUTH": 500, "WEST": -500, "NORTH": -1500}, "draw"),
        HandResult(1, 0, 0, {"EAST": 0, "SOUTH": 0, "WEST": 0, "NORTH": 0}, "draw"),
    ])
    entries, problems = T.align(hands, [game], {0: {"TL": "a", "TR": "d", "BL": "b", "BR": "c"}})
    assert problems == []
    # the winds are the overlay's, hand by hand; the site's fixed seat names follow the first hand
    assert entries[0]["corner_wind"] == {"TL": "E", "TR": "N", "BL": "S", "BR": "W"}
    assert entries[0]["corner_site"] == {"TL": "EAST", "TR": "NORTH", "BL": "SOUTH", "BR": "WEST"}
    assert entries[1]["corner_wind"] == {"TL": "N", "TR": "W", "BL": "E", "BR": "S"}
    assert entries[1]["corner_site"] == entries[0]["corner_site"]      # the players do not move
    assert entries[1]["scores"] == {"N": 26500, "W": 23500, "E": 25500, "S": 24500}
    assert entries[1]["nicks"]["TL"] == "a"
    game.hands[1].honba = 1
    _, problems = T.align(hands, [game])
    assert any("vs site" in p for p in problems)
