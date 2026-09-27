from video2tenhou.train.data import hand_of


def test_hand_of_goes_by_the_overlay_segment():
    # read windows start 60 s before the overlay switch and overlap; the overlay segments do not
    hands = [
        {"hand": 0, "game": 0, "t_start": 40.0, "t_end": 400.0, "t_overlay": [100.0, 400.0]},
        {"hand": 1, "game": 0, "t_start": 340.0, "t_end": 700.0, "t_overlay": [402.0, 700.0]},
        {"hand": 2, "game": 1, "t_start": 900.0, "t_end": 1200.0, "t_overlay": [960.0, 1200.0]},
    ]
    assert hand_of(350.0, hands) == 0            # inside the overlap: still hand 0's segment
    assert hand_of(401.0, hands) == 1            # between the segments (the overlay switching): the table is on hand 1
    assert hand_of(500.0, hands) == 1
    assert hand_of(800.0, hands) is None         # the break between hanchan belongs to no hand
    assert hand_of(950.0, hands) is None         # nor the window before the first segment of a hanchan
    assert hand_of(50.0, hands) is None


def test_hand_of_falls_back_to_the_window_of_a_legacy_table():
    hands = [{"game": 0, "t_start": 380.0, "t_end": 760.0}, {"game": 0, "t_start": 765.0, "t_end": 1060.0}]
    assert hand_of(400.0, hands) == 0 and hand_of(762.0, hands) == 1 and hand_of(1100.0, hands) is None
