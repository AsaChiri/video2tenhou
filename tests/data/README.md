# Committed regression evidence

`layout_boxes/` contains two dense annotated frames per view and corner from
the original PML calibration recording. These frame-coordinate boxes exercise
crop containment, all rotations, discard rows and column order without requiring
the video or a developer's personal labels. The remaining training annotations
stay in the ignored local `labels/` directory.

`week_11_calib.json` records the second broadcast's different camera placement.
Fixtures are test inputs, not writable project state or a model training corpus.
Serialized solver inputs use the current model schema. Removed metadata is
deleted from fixtures rather than handled by compatibility code; tile evidence,
constraints and reference results are preserved.

`week11_e4h1.json.gz` contains the recorded recognition outputs for week 11,
second hanchan, East 4 honba 1 (hand 19). It includes the site result, normalized
review facts, calm observations, and the dense windows needed by the decoder.
Player names and original media are omitted. Tile posteriors and geometry are
unchanged; this is not a synthetic expected-output fixture. The user's confirmed
regression is North's fifth and sixth turns: draw/discard 2p, then draw/discard
1z. A chi is included in the turn numbering. No draw fact forces either answer.

The integration test runs real pond tracking, call anchoring, turn merging,
hand reconstruction, score verification, Tenhou assembly and replay. Video
recognition is replaced by these recorded readings and the exact acquisition
plan that requested them, so CI needs neither model weights nor a GPU. Fixing
the acquisition plan avoids wall-clock solver contention selecting unrecorded
windows; separate tests cover acquisition thresholds and certified bounds.
Tile assignments and confidence are still solved normally. Every requested
window must exactly match recorded evidence, without slicing longer windows.
The test also preserves the previously reviewed site score
correction from 9 han/70 fu to 10 han/40 fu. To regenerate the evidence, capture
the same hand's observations and every requested dense reading before decoding;
never infer new ground truth by visually labeling its crops.

`week11_adjudicated_models.json.gz` contains numerical reconstruction inputs for
hands 7 and 24 from the full recording profile. The reviewer confirmed South's
18th discard as 1m in hand 7; in hand 24 they confirmed South's first draw as 4z
and exactly one 2p and one 4z in the starting hand. Only the discard and draw
become hard facts. The two starting-hand counts are assertions, not a fabricated
fact for the eleven unreviewed tiles. Tests reconstruct with these narrow facts,
check the winning score where applicable, and replay the exported hand.

`retention_known_1s.json` contains one human-annotated hand-camera reading and
the corresponding candidate detector/classifier probabilities. It covers a
correct `1s` detection lost by a later confidence filter. Tests distinguish raw
recognition from retained, restructured evidence and preserve the raw input.
The synthetic lower floor in the test verifies policy plumbing; it does not
qualify that floor for a deployed model.

`week11_open_kan.json.gz` contains anonymous numerical reconstruction inputs
and one exact 5 fps hand-camera window (8111–8181 seconds) for North's open-kan
replacement in week11 East4 honba1. The candidate previously inferred 5s with
a certified objective margin despite lacking direct replacement evidence. The
reviewer confirmed 1s. The test keeps that answer outside the model's facts,
selects the missing window through the production acquisition policy, maps its
real still views and reconstructs 1s while preserving the earlier 2p/1z draw
order and legal replay. The fixture contains neither weights nor source video;
posteriors are recorded unchanged, and player names are omitted.

`week11_contradicted_chi.json.gz` contains anonymous meld observations and an
exact recorded pond-camera window for a chi whose three positive sightings were
outnumbered by five later incomplete readings. The reviewer's chi answer is an
assertion outside the model inputs. Tests retain the contradicted hypothesis
only for independent anchoring, reconstruct the called-away red five from the
recorded pond window, and reject missing, wrong-source and wrong-tile anchors.
No source video, model weights or reviewer journal is included.

`week11_partial_hand_slots.json.gz` contains numerical partial hand views after
South's third discard in first-hanchan South4, 3honba. The reviewer confirmed
the draw as 7p. One visible box assigns probability to both 5s and 8s; another
box reads 8s. The test verifies that an unseen second 8s cannot improve the fit
by explaining the first box twice. Its starting state is a controlled test
setup, not a human annotation of the entire starting hand. Both 7p and 8s are
tested as diagnostic alternatives: these partial views alone cannot settle
the draw. No answer is injected into production evidence.
