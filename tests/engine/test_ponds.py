import numpy as np

from tests.engine.helpers import obs, row, slot
from video2tenhou.engine.ponds import (
    PondSlot,
    clearings,
    insert_slot,
    play_window,
    tail_runs,
    track_pond,
)
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def test_tracker_appends_and_a_call_takes_the_last_tile():
    seq = [
        obs(0, 5, row(["1m"])),
        obs(8, 12, row(["1m", "2p"])),
        obs(15, 20, row(["1m", "2p", "7z"])),
        obs(23, 26, row(["1m", "2p"])),  # 7z called away: the last tile, gone twice
        obs(30, 33, row(["1m", "2p"])),
        obs(36, 40, row(["1m", "2p", "9s"])),  # the next discard refills its position
    ]
    log = track_pond(seq)
    assert [(s.row, s.index, s.tile) for s in log] == [
        (0, 0, "1m"),
        (0, 1, "2p"),
        (0, 2, "7z"),
        (0, 2, "9s"),
    ]
    assert log[2].t_removed == 23  # the first view without it
    assert log[3].t_window == (33, 36)


def test_only_the_last_tile_can_be_called():
    # 2p hidden twice in the middle of the pond: it is not called away (only the latest discard can be)
    seq = [
        obs(0, 5, row(["1m", "2p", "3s"])),
        obs(8, 12, row(["1m", "3s"], start=0)),
        obs(15, 20, row(["1m", "3s"])),
    ]
    log = track_pond(seq)
    assert [(s.tile, s.t_removed) for s in log] == [
        ("1m", None),
        ("2p", None),
        ("3s", None),
    ]


def test_a_single_partial_frame_starts_nothing():
    """Hand 0 of the second VOD: at the reveal a single moving frame showed West's two 1z pushed down a
    row, plus tiles of the row above. None of it is a discard, and the 1z are the ones already known.
    """
    tiles = ["4z", "8p", "7m", "9s", "3s", "3p"]
    seq = [
        obs(0, 5, row(tiles) + row(["6s", "3m", "1z", "1z"], 1)),
        obs(270, 285, row(tiles) + row(["6s", "3m", "1z", "1z"], 1)),
        obs(
            291.5,
            291.5,
            row(["4z", "8p", "3s", "3p"])
            + row(["6s", "3m", "7m", "9s"], 1)
            + row(["1z", "1z"], 2),
            n_used=1,
            partial=True,
        ),
    ]
    log = track_pond(seq)
    assert [s.tile for s in log] == tiles + ["6s", "3m", "1z", "1z"]


def test_moved_tiles_are_the_same_tiles():
    # a player shifts the second row left (and the reader puts its first tile in the wrong row): same slots
    first = row(["1m", "2m", "3m", "4m", "5m", "6m"])
    seq = [
        obs(0, 5, first + row(["6s", "3m", "1z"], 1)),
        obs(
            10,
            15,
            first[:5] + [slot("6m", 1, 0)] + row(["6s", "3m", "1z", "1z"], 1, start=1),
        ),
    ]
    log = track_pond(seq)
    assert [s.tile for s in log] == [
        "1m",
        "2m",
        "3m",
        "4m",
        "5m",
        "6m",
        "6s",
        "3m",
        "1z",
        "1z",
    ]


def test_misread_last_tile_is_not_a_call():
    """Hand 20 of the second VOD: North's 1s read as 3z for three views while the next discard was laid."""
    seq = [
        obs(0, 5, row(["1s"])),
        obs(8, 9, row(["1s"])),
        obs(12, 14, row(["3z", "1p"])),
        obs(16, 18, row(["3z", "1p"])),
        obs(20, 23, row(["3z", "1p"])),
        obs(30, 35, row(["1s", "1p", "1m"])),
        obs(40, 45, row(["1s", "1p", "1m"])),
    ]
    log = track_pond(seq)
    assert [(s.tile, s.t_removed) for s in log] == [
        ("1s", None),
        ("1p", None),
        ("1m", None),
    ]
    assert log[0].disagree >= 1


def test_refill_after_the_call_is_a_plain_discard():
    # the last tile already missing in an earlier view: the next new tile is simply the next discard
    seq = [
        obs(0, 5, row(["1m", "4p"])),
        obs(8, 12, row(["1m"])),
        obs(15, 20, row(["1m", "6p"])),
    ]
    log = track_pond(seq)
    assert [(s.tile, s.index, s.t_removed) for s in log] == [
        ("1m", 0, None),
        ("4p", 1, 8),
        ("6p", 1, None),
    ]


def test_new_row_needs_six_in_the_previous_row():
    seq = [
        obs(0, 5, row(["1m", "2m"])),
        obs(8, 12, row(["1m", "2m"]) + [slot("7z", 1, 0)]),  # a stray tile below row 0
        obs(15, 20, row(["1m", "2m"]) + [slot("7z", 1, 0)]),
        obs(23, 26, row(["1m", "2m", "3m"]) + [slot("7z", 1, 0)]),
    ]
    log = track_pond(seq)
    assert [(s.row, s.index, s.tile) for s in log] == [
        (0, 0, "1m"),
        (0, 1, "2m"),
        (0, 2, "3m"),
    ]


def test_partial_observation_never_removes():
    seq = [
        obs(0, 5, row(["1m", "2p"])),
        obs(8, 12, row(["1m", "2p"])),
        obs(15, 16, row(["1m"]), partial=True),  # 2p hidden under an arm
        obs(20, 21, row(["1m"]), partial=True),
        obs(25, 30, row(["1m", "2p", "3s"])),
    ]
    log = track_pond(seq)
    assert [(s.row, s.index, s.tile, s.t_removed) for s in log] == [
        (0, 0, "1m", None),
        (0, 1, "2p", None),
        (0, 2, "3s", None),
    ]
    assert log[2].t_window == (12, 25)  # the partial views did not narrow the window


def test_the_clearing_stops_the_tracker():
    tiles = ["1m", "2m", "3m", "4m", "5m"]
    seq = [
        obs(0, 5, row(tiles)),
        obs(10, 15, []),
        obs(20, 25, row(["9p"])),
    ]  # the next hand's first discard
    assert [s.tile for s in track_pond(seq)] == tiles


def _pslot(i, tile, t, removed=None, row=0):
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.9
    s = PondSlot(i, row, 0, p, t, (t - 3, t), t + 4, 3)
    s.t_removed = removed
    return s


def test_insert_slot_takes_the_position_the_stack_gives():
    log = [_pslot(0, "1m", 10), _pslot(1, "2m", 20), _pslot(2, "3m", 30)]
    insert_slot(
        log, _pslot(9, "9p", 15, removed=18)
    )  # a called-away tile found by a dense read
    # the called-away 9p shares the position that 2m refilled; nothing after it moves
    assert [(s.tile, s.index) for s in log] == [
        ("1m", 0),
        ("9p", 1),
        ("2m", 1),
        ("3m", 2),
    ]
    full = [_pslot(i, "1m", 10 * i) for i in range(6)]
    insert_slot(full, _pslot(9, "9p", 15, removed=16))
    assert len(full) == 7 and full[2].index == full[3].index == 2


def test_tail_runs_align_by_reading_order_not_by_grid():
    """The second VOD's hand 9: E chi'd N's 5s, N's eighth discard. The frame's geometry put the 5s in the third
    row (tiles are not flush), where a grid lookup found no place for it; aligned with the stack in reading
    order it is the tile after the last one, and it vanished while the pond stayed in view: a taken discard.
    """
    log = [
        _pslot(i, t, 10 * i + 5)
        for i, t in enumerate(["1m", "2m", "3m", "4m", "5m", "6m", "7m"])
    ]

    def box(tile, row, col):
        q = np.full(len(CLASSES), 0.002)
        q[CLASS_INDEX[tile]] = 0.9
        return {
            "role": "tile",
            "row": row,
            "col": col,
            "xyxy": [col * 40, row * 50, col * 40 + 38, row * 50 + 48],
            "p": (q / q.sum()).tolist(),
        }

    rest = [box(x.tile, 0 if k < 6 else 1, k % 6) for k, x in enumerate(log)]
    reads = [
        {"t": 100.0, "boxes": rest},
        {
            "t": 100.2,
            "boxes": rest + [box("5s", 2, 0)],
        },  # laid; the geometry says a third row
        {"t": 100.4, "boxes": rest + [box("5s", 2, 0)]},
        {"t": 100.6, "boxes": rest[:3]},  # a hand over the pond: says nothing
        {"t": 100.8, "boxes": rest + [box("5s", 1, 1)]},
        {"t": 101.0, "boxes": rest},
    ]  # gone, the pond in view: called
    runs = tail_runs(log, reads)
    assert [(r["tile"], r["n"], r["gone"]) for r in runs] == [("5s", 3, True)]
    # a slot the calm reads saw later is matched from the start of its window, not new
    later = log + [_pslot(7, "5s", 103)]
    assert tail_runs(later, reads) == []


def test_play_window_ends_at_the_clearing():
    ponds = {
        "pond:TL": [
            obs(0, 2, []),
            obs(10, 12, row(["1m"])),
            obs(60, 62, row(["1m", "2m"])),
            obs(90, 92, []),
        ],
        "pond:TR": [obs(0, 2, []), obs(40, 42, row(["3p"])), obs(95, 97, [])],
    }
    # two ponds emptied together: one clearing, after the last calm view that held tiles (TL at 62)
    assert clearings(ponds, 0, 100) == [62]
    assert play_window(ponds, 0, 100) == (0, 62)


def test_play_window_finds_a_clearing_a_stale_count_would_hide():
    """The second VOD's hand 4/1: three ponds are read empty over the shuffle while the fourth was last
    read 30 s earlier, holding the previous hand.
    """
    prev = row(["1m", "2m"])
    ponds = {
        "pond:TL": [
            obs(0, 4, prev),
            obs(30, 34, prev),
            obs(62, 66, []),
            obs(70, 74, []),
            obs(80, 84, row(["5p"])),
        ],
        "pond:TR": [
            obs(0, 4, prev),
            obs(20, 24, prev),
            obs(59, 63, []),
            obs(75, 79, []),
            obs(88, 92, row(["6s"])),
        ],
        "pond:BL": [
            obs(0, 4, prev),
            obs(25, 29, prev),
            obs(61, 65, []),
            obs(72, 76, []),
            obs(95, 99, row(["7z"])),
        ],
        # last seen holding tiles at 46 s, then not read again until 81 s, well after the others are empty
        "pond:BR": [
            obs(0, 4, prev),
            obs(42, 46, prev),
            obs(81, 85, []),
            obs(93, 97, []),
        ],
    }
    assert play_window(ponds, 0, 120) == (46, 120)


def test_play_window_ignores_the_push_and_a_one_pond_dip():
    """Hand 2 of the second VOD: one pond was seen only in partial frames over the clearing, and the next
    hand's first discard lands in it; the others empty together. A single dip that comes back is not one.
    """
    full = row(["1m", "2m", "3m", "4m", "5m", "6m"]) + row(["7m", "8m"], 1)
    ponds = {
        "pond:TL": [
            obs(900, 990, full),
            obs(994, 996, full),
            obs(1003, 1007, []),
            obs(1060, 1062, row(["9m"])),
        ],
        "pond:TR": [obs(900, 995, full), obs(1001, 1002, []), obs(1004, 1014, [])],
        "pond:BL": [obs(900, 990, full), obs(1002, 1015, [])],
        "pond:BR": [
            obs(900, 985, full),
            obs(990, 990, full, n_used=1, partial=True),
            obs(1003, 1003, [], n_used=1, partial=True),
            obs(1023, 1040, row(["7m"])),
        ],
        # a pond dipping for one view (an occlusion) far from the clearing
        "pond:X": [
            obs(500, 510, full),
            obs(520, 525, row(["1m"])),
            obs(530, 540, full),
        ],
    }
    assert clearings(ponds, 600, 1046) == [996]
    assert clearings(ponds, 400, 700) == []
    assert play_window(ponds, 605, 1046) == (605, 996)
