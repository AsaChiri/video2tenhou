# video2tenhou — design

Turn a PacificML broadcast VOD of a physical-table riichi mahjong hanchan into
a tenhou.net/6 log. This document explains the physical constraints and
reconstruction rationale. See [maintenance](MAINTENANCE.md) for current entry
points, cache contracts and validation commands.

## 0. Contract

**Inputs**

| input | form | required |
|---|---|---|
| video | local mp4 of the broadcast (1080p preferred; any size is rescaled), or a video URL supported by yt-dlp to download | yes |
| site record | scoremj.com game ids, one per hanchan in the video, in video order | yes |
| calibration | two parts: the **layout** of the broadcast composite (one per layout; the PML layout ships with the repo) and the **fit** of this video (where the table and the camera panels landed), measured by stage 0 and confirmed in the tool | yes, the layout ships, the fit is measured |
| facts | human answers to review items from earlier runs on the same video | optional |

**Outputs**, per run, in `out/<video>/`:

| output | content |
|---|---|
| `g<k>.json` | tenhou.net/6-format log of hanchan k, every kyoku in `log` |
| `g<k>.html` | links for `g<k>.json`: the whole hanchan in tenhou's viewer, `https://tenhou.net/5/#json=<URL-encoded JSON>` (`&ts=N` starts at kyoku N), and one `https://tenhou.net/6/#json=` URL per kyoku carrying that kyoku alone, each with a copy button plus one that copies them all (one per line). tenhou.net/6 is tenhou's log *editor*: it loads only `log[ts]`, and the analysis tools that take its URLs take one kyoku per URL |
| `g<k>.confidence.json` | per event: margin, evidence, `human` flag, `lost` flag |
| `review.json` | open questions for the tool, each with evidence crops and a prefilled best guess |
| `report.md` | per hand: status (`complete` / `review` / `conflict`), counts, checksum results |

**Definition of done for a video**: every hand is `complete` (nothing open:
every tile of its log is backed by evidence or by a reviewer's answer), or a
conflict is listed that the program could not resolve. Nothing is silently
guessed: every event in the log that is not backed by evidence above the
margin is listed in the confidence file, and every tile no evidence covers
(section 6, `lost`) is a question in `review.json` until the reviewer
supplies it or confirms it cannot be seen. A log never leaves out a tile the
rules require (an indicator per kan, section 1): the best guess stands in it
while the question is open.

**Validation contract:** replay every exported hand, compare discards, calls,
riichi and dora against human-verified annotations where available, and check
winning-hand scores against the site record. Every uncertain draw or starting
hand must remain visible for review. The number of questions depends on camera
coverage and recognition quality; legal replay does not establish tile accuracy.

Live streams are not supported. Other composite geometries can be configured,
but the overlay font/text arrangement, centre-unit fitting and scoremj record
adapter retain PML assumptions; see [LAYOUTS.md](LAYOUTS.md) before adapting them.

## 1. The physical world: what can and cannot happen

These are hard constraints. The engine never re-derives them from pixels; it
uses them to reject or repair readings.

**Tiles.** 136 tiles: four of each of 34 kinds; one red 5 in each numbered
suit replaces one plain 5 (so plain 5m/5p/5s occur three times, 0m/0p/0s once).
Faces are identical between the two alternating tile sets (blue / yellow
backs). Tile notation in this project: `1m..9m 1p..9p 1s..9s 1z..7z`
(東南西北白發中), red fives `0m 0p 0s`, face-down `X`, unknown `?`.

**Rest.** A tile at rest does not change. A pond tile keeps its position and
identity until a hand takes it (a call). A hand tile leaves only by discard or
call; sorting permutes but does not change the multiset. Melds never change
after being laid except a kakan adding the fourth tile. Dora indicators only
grow (one more per kan). Therefore two readings of the same region with no
motion in between are readings of the same content: if they disagree, at
least one reading is wrong. Disagreement is not discarded as noise; it is
the primary detector of misrecognition. A slot whose readings disagree
within a rest interval gets a low margin and is checked against the other
signals (the pond for a discard, the meld camera for a call, the count rule,
later discards for a hand tile) before it can be trusted.

**Hand size.** Concealed + melded tiles = 13 between turns, 14 right after a
draw (each kan adds one tile to the player's total, and each kan is followed
by a rinshan draw). Haipai: 13 tiles for the three non-dealers and 14 for the
dealer. The dealer's 14 are treated as one set: some players sort before
taking the 14th, others after, so which of the 14 was "drawn" is not
observable and not needed. The log format wants 13 + 1; the split is made at
write time (section 4.9) and marked arbitrary in the confidence file.

**Draw.** After a draw the player holds 14 tiles, and the drawn tile is the
one at the leftmost or rightmost end of the row until the player sorts it in
or discards it. When a frame shows the 14-tile row with the new tile at an
end, that is a direct observation of the draw and is used as such;
inference from consecutive 13-tile states is only for turns where that
moment was not captured.

**Turn.** Dealer first, then counter-clockwise E → S → W → N. A turn is draw
then discard. A call (chi/pon/daiminkan) takes the last discard, skips the
draw, and passes the turn to the caller, who discards next; a daiminkan,
ankan or kakan is followed by a rinshan draw before the discard. The wall
has 136 − 52 − 14 = 70 live tiles. A rinshan draw comes from the dead wall,
which is kept at 14 by moving the last live tile into it, so every kan
costs the live wall one tile: wall draws (the dealer's 14th tile included)
plus kans never exceed 70, and an exhaustive draw happens exactly when they
reach 70. A turn sequence that needs more is a misread sequence, not a hand.

**Discards.** Placed in the discarder's pond in reading order from the
player's left, six per row, a new row after six. The first row is the one
nearest the centre unit and later rows grow toward the player (verified on
the labelled ponds of the reference VOD; kept as a calibration field so a
different table can override it). A called tile's position is refilled by the
next discard, so a row stays six long and nothing shifts (the earlier "empty
position" rule was wrong: none of the 733 labelled rows of the reference VOD
shows a gap, and dense reads show the next discard taking the place). A
row's positions are therefore an order of appearance, removed tiles
included, not the columns a frame shows. Only the last discard can be
called, so the tile a call takes is always the most recent one in its pond:
a pond is a **stack** read in reading order — it grows at the end, it loses
only its last tile and only to a call, and nothing is ever inserted into
the middle. The tiles themselves can be nudged: a player straightens a
row, an elbow pushes a block, and at the end of the hand everything is
pushed into the table. Positions move; the order of the tiles does not.
A riichi declaration tile lies
sideways; if that tile is called, the
next discard lies sideways instead. A discard is either the tile just drawn
(tsumogiri) or a tile that was already in the hand.

**Calls.** chi: three consecutive tiles of one numbered suit, only from the
player before you (kamicha); the called tile lies sideways at the left of the
meld. pon: three identical tiles from anyone; the sideways tile's position
(left / middle / right) tells which player it came from (kamicha / toimen /
shimocha). daiminkan: four identical, same position rule. ankan: four
identical from hand, two shown face-down. kakan: the fourth tile placed on
the pon's sideways tile. A call takes at least two tiles that were already in
the caller's hand (three for daiminkan, four for ankan). Every kan reveals a
new dora indicator; at most four kans per hand. So the kans of a hand are
exactly the indicators after the first, one for one: a kan with no new
indicator did not happen, and a new indicator with no kan read was revealed
by a kan the cameras missed.

**Dead wall.** The face-up indicators lie side by side in one row of the
dead wall. The row can be pushed as a whole (players straighten the walls)
but a face-up tile never turns back, so the number of face-up tiles only
grows, by one per kan, and their order along the row does not change. A
tile that vanishes from one place in the moment a tile appears at another,
with the count unchanged, is the same tile moved; if the two readings
disagree, one of them is a misreading.

**Riichi.** Only with a closed hand; a 1000-point stick goes to the table;
from then on the hand multiset is frozen (every later discard is the drawn
tile) except for an ankan. The PML overlay updates its stick counter and
the scores only between hands, so the moment of a declaration is seen only
through the turned tile; when the turned tile follows a discard of the same
player that was called, the declaration was either that called tile or the
turned one, and the hand's legality under the freeze decides.

**End of hand.** Win by tsumo (the drawn tile completes the hand; the winner
reveals 14 tiles minus melds, the drawn tile among them) or by ron (the last
discard completes it; the winner reveals 13 minus melds; the winning tile
stays in the discarder's pond and is never moved). At the reveal some
players move their melds closer to the hand row, so the meld camera and the
hand band may both change at that moment without any tile changing
identity. Atamahane: one ron winner. After a riichi win the ura indicator(s)
under the dora indicator(s) are revealed. Exhaustive draw: tenpai players
reveal. Then all tiles are pushed into the table; the next hand's walls come
up.

**Counting.** At any moment, for each kind: (tiles in the four hands + ponds +
melds + revealed indicators) ≤ 4, and the remainder is in the wall or the
dead wall. Over a whole hand the union of haipai + draws + indicators never
exceeds four of a kind.

**Ruleset (PML TEAMs = M-League, per the organiser's document).** Facts the
engine uses: atamahane; no abortive draws; kiriage mangan; aka dora three;
tenpai payment 3000 at an exhaustive draw; honba 300 per honba to the winner;
riichi sticks to the next winner; 25000 start; no agari-yame. The site record
carries the result, so scoring is only a *check*, computed with the `mahjong`
library under these settings.

## 2. What the broadcast shows

One 1920×1080 composite (calibration gives every rectangle):

- **Overhead camera**, rotated 45° into a centre diamond: the four ponds, the
  centre unit, the inner ends of the walls, the dead wall with the dora
  indicator(s). Pond tiles are about 45×60 px, flat, in one of four
  orientations per pond. The dora indicator is the face-up tile in the wall
  ring.
- **Four corner cameras** (TL, TR, BL, BR), each looking over one player's
  shoulder: that player's concealed hand faces the camera, tiles 45–90 px
  wide with perspective (near end large, far end small), sharp. The wall in
  front of the player and their melds (to the right of the hand) are partly
  visible. Arms cross the hand often.
- **Four meld cameras**, small insets at the inner corner of each quadrant,
  pointing at that player's meld area.
- **Overlay text**: per corner `score`, seat-wind kanji, `Name (nick)`; bottom
  centre round wind, kyoku number, honba, riichi sticks. White on dark.
- **Noise**: five face-cam insets, table LED displays, sponsor cards.

The site record supplies names (nick = site username), starting seats and
scores. The overlay is used for what the site cannot give: *when* in the
video each hand starts and ends, which corner holds which seat, and the
riichi-stick counter timing.

## 3. Architecture

### Browser workflow

`video2tenhou web` opens a loopback-only workspace for the complete recording
lifecycle. A project records its source, ordered scoremj game IDs and layout
in `work/projects/`; videos, caches, facts and outputs retain their stage
contracts below. Local file uploads stream to disk and video downloads
run in a child process. Only one analysis job runs at a time so competing
videos cannot exhaust GPU memory. Progress comes from real pipeline output;
failed and interrupted jobs remain visible and can be retried using caches.

Preparation measures the fit. The user opens the calibration view, checks
the crops, then starts analysis; geometry checks remain mandatory. Project
scoped review URLs bind every fact, frame and calibration request to that
project. Re-decoding refreshes exports as well as review data. Results show
open questions separately from completed logs and provide direct downloads
and viewer links. Advanced labelling stays inside the review workspace.

The studio uses a slate background (`#edf2f5`), white paper (`#ffffff`),
navy text (`#193343`), table green (`#245d51`), amber (`#94651c`) and red
(`#ac3544`). Segoe UI/Arial is the interface type; Georgia provides one
editorial title. A wide recording intake above a persistent project list
leads to a sequential preparation/review/export workspace. The memorable
element is a small real tile hand, not decorative status cards. Tile SVGs
are vendored from FluffyStuff/riichi-mahjong-tiles at a pinned commit with
their CC0 license. The layout collapses to one column on narrow screens,
and status changes use live regions without invented completion estimates.

A pipeline of stages with a file artifact between each, all under
`work/<video>/`, so every stage can be rerun and inspected alone.

```
video ─► [0 fit]        plate.png, labels/<video>/calib.json (where the table and the panels are in THIS video)
      ─► [1 header]     overlay.jsonl, hands.json (hand windows, corner→seat, site alignment)
      ─► [2 calm]       calm.jsonl (per region: calm intervals, occlusion score)
      ─► [3 read]       reads/<hand>/<region>.jsonl (per calm frame: boxes + class posteriors)
      ─► [4 observe]    obs/<hand>.json (per region per calm interval: one voted observation)
      ─► [5 decode]     decode/<hand>.json (events, solver margins, review items)
      ─► [6 write]      out/<video>/g<k>.json + confidence + review + report
      ◄─ [7 review tool] labels/<video>/facts.jsonl (human answers → hard constraints → rerun 5–6)
                        labels/<video>/boxes/*.json (human box/tile labels → retrain models)
```

Recognition in stage 3 and targeted video rereads in stage 5 can use the GPU;
geometry preparation and its checks also use recognition. Stage 5 combines
those readings with CPU constraint solving and alternative searches, so a
rebuild can require substantial work even when sparse recognition is cached.
Each stage validates the provenance relevant to its own artifacts, including
source, geometry, sampling, model/runtime and evidence-policy identities where
applicable. Review answers invalidate the affected reconstruction; changing
recognition inputs requires Analyze to refresh earlier evidence before Rebuild
can use it. Calibration is reused for its recording and can be adjusted by the
reviewer. Later stages use that geometry, and conversion checks it before tile
analysis (section 4.2a).

Package layout:

```
src/video2tenhou/
  cli.py            video2tenhou web | download | convert | review | calib
  video.py          frame access: sequential sampler (ffmpeg pipe) and seek
  layout.py         Calibration geometry, layout + per-video fit, contact sheet
  calibfit.py       stage 0: the table plate, the fit of this video, the geometry checks
  record.py         scoremj.com client → Game, HandResult
  overlay.py        overlay reader (template kanji + digits, tesseract for nicks)
  timeline.py       hand windows, corner→seat map, alignment with the site record
  calm.py           motion / occlusion scoring, calm intervals per region
  perception/
    detector.py     tile box detector (YOLO, classes face / back), one model for all views
    classifier.py   tile face classifier (CNN on crops, 39 classes, calibrated posteriors)
    reader.py       read_region(frame, region) → boxes with posteriors and orientation
  observe.py        calm-interval voting → observations (ponds, hands, melds, indicators)
  engine/
    rules.py        invariants of section 1 as checkable functions
    ponds.py        pond slot tracking → per-seat discard log with times, sideways, removals
    melds.py        meld observations → call events (type, tiles, source)
    calls.py        the call anchor: a meld event is a call only if a discard was taken (or a kan pattern)
    dense.py        5 fps pond and hand reads on demand: taken discards, skipped turns, uncertain draws
    hand.py         seats, corners and the site's seat names of one hand
    turns.py        merge of four discard logs into the turn sequence with calls
    solver.py       haipai / draw reconstruction as a constraint program (CP-SAT)
    scoring.py      yaku / han / fu check against the site record (mahjong library)
    indicators.py   the dead-wall row: dora / kan indicators and the kans they imply
    review.py       margins → review items and confidence rows; facts → constraints
    decode.py       orchestration per hand
  tenhou6.py        log model, writer, and replayer (legality check of a finished log)
  tool/             browser projects, owned processing jobs, review and labeling
  train/            dataset builders from local labels, detector and classifier training
  eval.py           metrics per stage on labels; end-to-end against verified logs
tests/              unit tests on synthetic states plus small real fixtures
docs/DESIGN.md      this document
labels/<video>/     local human ground truth (ignored)
models/             trained weights (ignored); models/external/ pretrained bases
work/, out/, samples/   caches, outputs, video (ignored)
```

## 4. Components

### 4.1 `video.py`

- `probe(path) -> VideoInfo(fps, width, height, duration)`.
- `sample(path, fps=2.0, start=0, end=None, scale=(1920,1080)) -> Iterator[(t, frame)]`:
  one sequential ffmpeg decode, frames rescaled to 1080p, never seeks; this
  is the only way stages 2–3 read video (a 2.4 h VOD is 17k frames at 2 fps).
- `frame_at(path, t) -> frame`: seek, for evidence crops and the tool.
- `download(url, out, start=None, end=None)`: yt-dlp best 1080p.

Why 2 fps: placing a tile takes over a second; a discard is visible for many
seconds before the next; motion between two samples 0.5 s apart is enough to
tell calm from disturbed.

### 4.2 `layout.py`

`Calibration` (JSON, 1080p coordinates): overlay rectangles; corner camera
rectangles and, inside each, the hand row band and the meld inset; overhead
centre, angle, scale; the four pond rectangles in the de-rotated overhead
frame, each with its upright rotation (0/90/180/270) and its row growth
direction; the wall ring for indicators; corner ↔ pond mapping. Functions:
`load(name_or_path)`, `region(frame, calib, name) -> (image, transform)` where
the transform maps region pixels back to frame pixels (used for evidence
crops and for converting labels), and `contact_sheet(frame, calib)` for
`video2tenhou calib check <video> --t <s>`.

A calibration has two layers. The **layout** (`src/video2tenhou/assets/calib/pml.json`) is
what the broadcast software draws and what the table is: the overlay
rectangles, the corner-camera quadrants, and — in the de-rotated overhead
square — the centre unit and the four pond rectangles. Those last are a
property of the *table*, a fixed physical object, not of a video. The **fit**
(`labels/<video>/calib.json`) is where that table and those panels landed in
*this* video: `overhead` = (centre, angle, scale), and a rectangle (plus the
roll of the hand row) per corner camera panel. `Calibration.load(name,
video)` returns the layout with the video's fit applied; with no fit file it
returns the layout's own numbers and says so.

Every pixel number in the project lives in one of the two files.

### 4.2a `calibfit.py` — stage 0: the fit of this video

Cameras are re-mounted between broadcasts. On the reference VOD the overhead
camera puts the centre unit at (960, 540) rotated 45°; on the second VOD of
the same layout it is at (1015, 537) rotated 46.5°, and the meld insets have
moved by up to 60 px. A 55 px error in the overhead is half a tile: pond
rectangles cut the row nearest the centre unit (the *first* row, whose loss
renumbers every discard of the hand) and reach into the neighbour's pond, so
one player's discards are read as another's. Nothing downstream can recover
from that, so the geometry is measured per video, checked, and confirmed by
a human before anything else runs.

**The table plate.** The median of N = 60 frames spread over the video
(`work/<video>/plate.png`). Tiles, hands and players average away; what is
left is the table, the felt markings, the centre unit, and the hard straight
borders of every panel the broadcast pastes into the composite. The faint
white ghosts inside the overhead are exactly where discards live over the
whole video: the plate is both the thing to fit and the evidence to check
against.

**The overhead fit.** The centre unit is the one fixed, unmistakable object
on the table: a dark, unsaturated square in an image that is otherwise teal
felt and white tiles. Stage 0 masks it on the plate near the frame centre,
then finds the (centre, angle, scale) whose de-rotation maps that mask onto
the layout's stored unit mask (`src/video2tenhou/assets/calib/pml_unit.png`, cut from the
reference plate through the reference fit), maximising IoU: a coarse sweep
over angle and scale, then a hill climb. On the reference video this
recovers its own calibration to 0.05 px and 0.01°; a fit under IoU 0.6 is
reported as a failure, not a number. Because the pond rectangles are table
constants, fitting the unit fits the ponds.

**Panels.** The meld inset, the hand band and the corner camera are panels of
the composite; their borders are visible on the plate but not always (a
teal-on-teal border has no gradient), so they are *guessed* and then
*confirmed*: the layout's rectangles are the starting point, the hand band
and its roll are fitted from the detector's boxes (the row line through the
largest cluster of tile boxes in the corner camera), and the tool's
**Calibrate** page shows every rectangle over the plate for the human to drag
(section 4.10).

**The check: a region border never cuts a tile.** A crop is right when every
tile is either wholly inside it or wholly outside; a tile split by the border
is the definition of a bad crop, and it is what produces a missing first row,
a neighbour's discard and a half-read meld. Stage 0 therefore runs the
detector on K = 8 frames spread over play, over the *whole* overhead and the
*whole* corner camera, and for every region reports:

- `cut`: boxes that straddle the region's border (> 15 % of the box on each
  side). A region fails when a third of the tiles at its border are cut, or
  when two or more are and they are over 8 % of what it holds; a single cut
  box among many held ones is a stray detection (a tile in a player's
  fingers crossing the border) and only warns.
- `held`: boxes wholly inside — a region that holds no tile in eight frames
  of play is not looking at the right place (a failure for ponds and hand
  bands; a warning for meld insets, where a hand may have no call).
- `foreign`: for a pond, boxes inside it that belong to another pond's block
  of tiles (the blocks are the plate's occupancy components: the bright
  ghosts of the discards, each assigned to the pond rectangle it lies in
  most). Any is a failure.
- the overhead fit itself: a unit match (IoU) under 0.6 is a failure.

When the fit suggests a panel rectangle, it replaces the current one only
when its check is better: the verdict first (ok, warn, fail), then more
tiles held, then fewer cut — a rectangle that sees nothing cuts nothing,
and must not win for that.

`video2tenhou calib fit <video>` writes the fit and prints this table;
`calib check <video>` prints it for the current fit and exits non-zero on a
failure; `convert` runs it before stage 1 and refuses to start when it fails,
naming the region and the command to fix it. The checks are cheap (one plate,
eight frames) against a decode that takes an hour.

### 4.3 `record.py`

GraphQL client for scoremj.com: `fetch_game(id) -> Game(players by starting
seat, final scores, hands: [HandResult(kyoku, honba, sticks, deltas, outcome,
winner, loser, han, fu, riichi seats, tenpai seats)])`. Pure data; cached to
`work/<video>/record.json`.

### 4.4 `overlay.py`, `timeline.py`

- `read_overlay(frame) -> Overlay(scores{corner}, winds{corner}, nicks{corner}, round, kyoku, honba, sticks)`
  with the sanity rule scores + 1000·sticks = 100000 (nicks only when asked).
- `segment(overlay.jsonl) -> hands.json`: hand windows keyed by (kyoku,
  honba) with debouncing; hanchan boundaries where kyoku resets; per hand the
  corner→seat map from the wind glyphs, cross-checked by nick ↔ site username
  and by scores ↔ site cumulative scores. Disagreement is a hard error (wrong
  game id or wrong video).
- Riichi-stick counter timeline (t, sticks). On the PML overlay it only
  changes between hands (the counter and the scores are updated with the
  result), so it carries the sticks at the start of each hand and nothing
  about when a declaration happened.
- Physical window of a hand: the overlay switches to the next hand while
  tiles are still on the table; the hand window is trimmed to the interval
  between the table clearing before and after. A clearing is read from the
  ponds' own counts (4.8 `ponds.py`): within a hand a pond's count never
  drops by more than one, so a calm view with several tiles fewer than the
  previous calm view of the same pond — confirmed by the next — is the
  table being cleared.

### 4.5 `calm.py`

For every region of interest (4 ponds, 4 hand bands, 4 meld insets, indicator
ring), per sampled frame: `motion` = mean absolute difference to the previous
sample after blur; `occlusion` = fraction of skin-coloured pixels plus a
"tile-like structure" score (edge density in the tile band). A frame is calm
for a region when motion < τm and occlusion < τo. A **calm interval** is a
maximal run of ≥ 3 calm samples (≥ 1 s). Output per region: intervals with
start, end, mean scores, and the **disturbed intervals** between them.
Thresholds are set on the reference VOD and reported as a histogram in
`calib check`.

**Read floor.** No region may go longer than `MAX_GAP` = 15 s (one turn
cycle) without a read; a blind span longer than that is where calls and
draws get lost (hand 20: the meld camera had no calm frame for 68 s because
a resting hand covered part of the inset, and the pon inside that span was
first seen 36 s late). Wherever the calm criterion leaves such a span, the
stage adds **partial intervals**: runs of ≥ 3 still samples regardless of
the skin score (an arm resting over part of the region), and if a span is
still longer than the floor, the stillest sample of every 7.5 s chunk —
provided it is still (motion under 1.5 τm): a frame in motion shows tiles
being moved, and at the end of a hand, tiles being pushed into the table
in every position but their own. They
are read like calm intervals and flagged `partial` through to the
observations. A partial observation is positive evidence only: a tile seen
is there; a tile absent may be hidden. Ponds never remove a slot on a
partial observation, never start a slot from a single partial reading, and
never narrow a later slot's window by it; a meld
seen only partially may be a kan seen as a pon, so its later growth is not
a kakan; a hand row with the resting count + 1 is full evidence (nothing
can be hidden), and so is a row with the resting count between the seat's
turns, where the hand holds exactly that many tiles: every tile is seen.
Only inside a turn's window is a resting-count row weak evidence (the hidden
tile may be the draw).

### 4.5a Evidence during disturbed intervals

The implementation uses targeted dense reads in `engine/dense.py` when calm
observations cannot settle an event. It does not implement a separate motion
tracker or an `activity.py` stage. Dense observations retain the same tile
posteriors and geometry as ordinary reads; merged discard and call chronology
constrains which hand state they can describe. Section 4.8 describes the
fallback windows and evidence weighting.

### 4.6 Perception

**Detector** (`perception/detector.py`): one local checkpoint serves all views.
Its hash-verified metadata selects the backend and inference settings.
The LibreYOLO YOLO9 adapter localizes face-up tiles and returns boxes in
region coordinates. Inputs are region crops scaled so tiles are 60–120 px wide
(scale factor per region in the calibration). Human labels span ponds, melds,
hands and indicators; hand labels use raw camera space, with rectified
labels mapped back through their stored matrices. See
[detector training and backend contracts](DETECTOR_BACKENDS.md) for the
reproducible training recipes and independent acceptance checks.

**Classifier** (`perception/classifier.py`): a CNN (torchvision resnet18,
ImageNet init) on crops rectified to 64×96 (tile aspect), 39 classes = 34
kinds + 3 reds + `X` (back) + `none` (not a tile). Every crop is first turned
upright using the known orientation of its region (ponds: per-pond rotation
plus the box aspect for sideways tiles; hands: the camera's roll from the
calibration). Augmentation: ±15° rotation, perspective jitter, scale, colour,
blur, cut-out for partial occlusion. Trained on all labelled crops of all
views, evaluated per view on held-out hands. Posteriors
are temperature-calibrated on the held-out set so the solver's costs mean
something.

**Reader** (`perception/reader.py`): `read_region(frame, calib, name) ->
Reading(t, region, boxes=[Box(xyxy, det_conf, sideways, posterior[39])])`. For
hands, boxes are ordered along the row line fitted through box centres; for
ponds, boxes are assigned to rows and columns in the upright pond frame
(rows = clusters along the growth axis, a column = the rank along the row:
a called tile leaves no gap and hand-laid tiles are unevenly spaced, so a
gap means nothing). A row of seven that no neighbouring pond explains
rejects the reading; at a side edge the neighbour's tile is the box at the
edge the row touches. For melds, boxes are grouped into rows by height,
never two boxes that overlap along the row (the same tile boxed twice keeps
its better box; two tiles overlapping along x are in different rows), and
split at gaps; a run of more than four boxes is several melds laid close
together and is split by the call rules (4.8 `melds.py`).

Why two models: the face problem is identical in every view once the crop is
upright, so all labels pool into one classifier; localisation differs by view
and is easy. Posteriors from a dedicated classifier are better calibrated
than YOLO class scores.

Inference keeps the same samples, full precision and crop geometry while
batching classifier work. Stage 3 buffers at most 48 upright region crops;
tile crops share a classifier call, still including both rotations of
sideways tiles. Detector calls stay at batch size one: larger CUDA batches
changed box coordinates enough to alter tile crop pixels in measurements.
The detector's multi-image API therefore also uses individual calls. Buffers
are flushed at each hand/window boundary and caches retain their time order.
Dense cache identity includes sampling rate and millisecond window bounds:
a 2 fps result cannot satisfy a later 5 fps request. Observation construction
filters and restructures each interval's readings once, sharing that prepared
evidence between the count-change check and the final votes.

### 4.7 `observe.py`

Turns per-frame readings into one observation per region per calm interval:

- **A change inside an interval.** The calm criterion is about motion, and a
  discard or a draw can be made quickly enough to leave an interval calm.
  When the box count of an interval's readings changes once and stays
  changed (two runs of at least two readings each), the interval is split
  there into two observations; otherwise readings off the mode are dropped
  as noise.
- **Pond observation**: slots (row, col) with the summed posteriors over the
  interval's frames and the sideways vote; readings with a row of seven are
  rejected.
- **Hand observation**: the count of boxes (mode over frames), and per slot
  (position along the row) the summed posterior; frames whose count differs
  from the mode are dropped. A 14-tile observation records the drawn tile:
  the end tile (left or right) whose removal leaves the 13-tile row of the
  previous observation, compared as multisets (the player may not have
  sorted, but the tiles are the same). The comparison must leave at most
  two differences (misreads); a reference row further off than that is of
  another state and names nothing. If both ends are possible, both are
  candidates, weighted by the end the player has used in the rest of the
  hand (players are consistent: on the second VOD 207 of 207 decided draws
  are at the right end). This is the direct draw observation of section 1;
  it goes into the decode as a strong reading, not a hard fact, so a misread
  end tile can still be repaired by the pond (tsumogiri) or by the next
  13-tile state. (Implemented where the states are mapped to turns,
  `solver.hand_evidence`, since the reference row is the previous state.)
- **Disagreement**: a slot whose readings inside the interval do not agree
  (top class changes or the best posterior is spread) is marked; the
  observation's quality is reduced and the slot enters the decode with its
  full distribution instead of an argmax.
- **Meld observation**: groups of 3–4 boxes with their sideways position and
  `X` tiles; per group the summed posteriors.
- **Indicator observation**: face-up boxes in the wall ring with posteriors.

An observation carries its interval, frame count, a quality score and the
`partial` flag of a read-floor interval (see 4.5: absence is then unknown). The
"tile at rest" rule is applied here: within a calm interval nothing may
change, so the vote is over identical content.

### 4.8 Engine

**`ponds.py`** — the stack of section 1, tracked through the pond
observations of one seat. Each observation is flattened into one sequence
in reading order (row by row, left to right); the rows the reader found
serve only to order it, because a pushed block or a straightened row moves
tiles across the reader's row boundaries without changing their order. The
sequence is aligned with the stack's visible tiles by order and identity
(one alignment over the whole pond, not row by row):

- a matched tile is a new reading of that slot; a reading that disagrees
  with the slot's identity is counted as a disagreement (section 1), never
  as a new tile;
- tiles after the last matched slot are **new discards**. A new slot needs
  a view at rest: one full observation of at least two readings, or two
  observations. A slot seen once, in a partial or one-reading view, is
  tentative, and is dropped unless the next view confirms it;
- an unmatched tile anywhere else is noise: nothing is ever inserted in the
  middle of a pond;
- only the **last** visible slot can leave, when two consecutive full views
  lack it while the slot before it is seen: it was called away between its
  last sighting and the first view without it (`t_removed`, the window the
  call is matched against). Any other slot a view lacks is hidden or
  misread, and stays. A misread last tile looks like a call and a refill;
  two rules tell them apart. The last tile gone and two new ones in a view
  less than 10 s after the previous one cannot be a call (the seat would
  have discarded twice): the first "new" tile is the last one misread. And
  a "called" slot whose refill, laid in the very view it vanished from,
  ends up the same tile after disagreeing readings was never called: the
  two slots are one tile;
- a full view that lacks at least three slots and more than half of the
  stack is the **clearing** — the end of the hand — and the tracker stops.

A slot's (row, index) is its position in the stack when it was laid
(six per row), which is the refill rule: a called-away tile's successor
takes its position. The discard log is ordered by the order of laying,
never by the reader's rows. A slot found later by a dense read enters the
log by its time (`insert_slot`) with the position the stack gives it.

`play_window` finds the hand's physical window in the same observations:
per pond, a calm view with at least three tiles fewer than the previous
calm view of that pond, and not contradicted by the next one, is a reset;
resets of two or more ponds within 40 s of each other are one clearing (a
single pond's reset needs the other ponds to have no calm view to confirm
or deny it). The window is the stretch between two clearings that overlaps
the overlay segment most; it starts after the last calm view that still
held the previous hand's tiles and ends with the last calm view before the
next clearing. Partial views take no part: they are exactly the frames of
the push.

**`melds.py`** — from meld observations of one seat: a call event when a new
group appears: `(t, type ∈ {chi,pon,daiminkan,ankan,kakan}, tiles, source ∈
{kamicha,toimen,shimocha,self})`, type decided by the group's composition
under the call rules (chi: consecutive same suit; pon/kan: identical), source
by the sideways position. A group is a legal meld only when its readings
support it: four boxes are a kan only when every one of them reads as that
kind (the majority never overwrites the rest — "2m 4m 4m 3m" is the chi
2m-3m-4m with its 4m boxed twice, not four 4m); otherwise the best legal
meld of three of them is taken and the fourth box is noise. A run of five
or more boxes is split into consecutive melds of three or four, choosing
the split whose melds the readings support best. A group that matches a
known meld of the seat, or holds it plus one stray box, is that meld seen
again: melds never change after being laid except by a kakan, which adds a
fourth tile of the pon's kind to a pon; a seat has only one pon or kan of a
kind, so two groups read as the same pon are one meld read twice, and a seat
calls at most once between two views of its camera. A meld never leaves the
table: later full views that lack a group more often than they show it (a view
showing two of its tiles still shows it) contradict that camera hypothesis.
Ordinary tracking excludes it, but call anchoring retains an external chi,
pon or kan hypothesis for an independent check: a compatible called tile must
actually disappear from the indicated player's pond in the event window.
The disputed hypothesis cannot claim an unrelated same-suit removal or become
a self-kan merely because no discard was found. This preserves positive call
evidence when later occlusion or regrouping falsely resembles absence. One
seen only in a single partial frame is *weak*.

The camera says *that* a seat laid a meld and roughly what it looks like; it
is the weakest of the three witnesses of a call, and never decides the call
on its own. **A call is anchored on the discard it took** (the user's rule:
"you should be able to tell from the discard"):

1. *Meld events.* Every new group in a seat's meld camera — a legal meld, a
   group that reads as no legal meld, or a **fragment** (two tiles of a meld
   with no legal third: the turned tile unboxed, or misread) — is an event
   of that seat, timed by the camera's last view without it and first view
   with it. An event cannot be first seen before the hand's first discard
   (20 s of slack: the first discard may have been called away at once) nor
   happen after its last one (only a tsumo winner's kan can).
2. *Was a discard taken?* Only the latest discard can be called, and it
   vanishes from its pond. The calm pond logs are searched first (a removal
   no other call has taken, in the event's window — and never a tile still
   seen in its pond after the meld was on the table: the meld holds the
   called tile, so it left the pond first); when they show none, the
   three other ponds are read densely (5 fps) over the 25 s before the event
   (as far back as a calm removal is matched) for any tile that appears at
   the end of a pond and vanishes — the end
   found by aligning each reading with the stack in reading order, as the
   tracker aligns a calm view, never by grid position (a tile laid askew
   can sit in the frame where a grid would put the next row). The
   discarder's pond reading of that tile is the **called tile** (ponds are
   read at 99 %, meld insets far worse). A chi or pon is of one suit, and
   the insets confuse numbers, not suits: a removal of another suit than
   the one the camera reads was not taken by this meld (a 0m for a meld read
   7z 7z 7z is some other call's).
3. *With a discard taken* the event is a chi, pon or daiminkan of that tile:
   the source is the discarder (chi only from the kamicha). The two (three)
   tiles from the caller's hand are **not** read off the camera: they are a
   choice among the legal melds built on the called tile — for 4p from the
   kamicha: 2p3p, 3p5p, 5p6p (chi), 4p4p (pon), 4p4p4p (daiminkan, four
   boxes) — costed by how well the camera's readings support each, and
   decided by the solver together with the caller's hand, which must hold
   those tiles before the call and not after (section 4.8 `solver.py`).
4. *With no discard taken* the event is an ankan (a pair of identical face-up
   tiles, the backs unboxed) or a kakan (a pon of the seat grown by one tile
   of its kind); the camera's pattern decides, the dead wall confirms when
   it is in view (it is often cut off at the edge of the overhead). An
   ankan is all four tiles of its kind, so it is checked against the
   counting rule once every call is known: a kind that another call or a
   pond at rest shows cannot be an ankan. A pair seen only as a fragment is
   then no kan at all (it was another meld's tiles, misread); a kan the
   camera saw whole keeps its place and loses its kind, which the solver
   chooses (never a five: those the camera names).
5. *Neither* — no discard taken and no kan pattern — is not a call. A
   clear chi or pon the camera held for several views, with no discard
   found even in a dense read, is a `call` question — but only when the
   caller's hand camera agrees a meld was laid: a chi, pon or daiminkan
   leaves the resting hand three tiles shorter, so a hand that still holds
   as many tiles as the seat's known calls allow says the camera re-read a
   meld already counted (meld insets regroup their tiles from view to view),
   and that is a note.

Every removal is a call, too: a tile that left a pond (step 2 of the pond
tracker) that no event explains was called by a seat whose camera missed the
meld. The merge (`turns.py`) decides the caller from the turn order — the seat
that discards next, out of the natural order — and the solver its tiles, as
in step 3. A reviewer's meld fact (type, tiles and source) replaces the
program's call nearest its time, whatever step produced it, and suppresses
any other call of that seat inferred within 40 s.

**Kans.** A kan is rarely read directly: the meld camera shows an ankan as
two face-up tiles between two face-down ones (the reader boxes only the
faces), and a kakan adds one tile to a pon. The reliable signal is the
dead wall: every kan reveals a new dora indicator, so each indicator after
the first, in the order of appearance, is a kan event just before it.

The indicators are tracked as the dead-wall row of section 1, per pond
region, through every view: a view showing as many face-up wall tiles as
the row holds is a new reading of each of them in order, whatever their
positions (the row may have been pushed); a view with one more holds one
new indicator, placed by aligning the others by order and identity; a view
with fewer lacks some (hidden or cut). A tile is real when it persists:
seen in at least three full views and in at least half of the full views
after its first sighting. Its identity is the vote over all its readings;
readings that disagree are a misrecognition, and the indicator is asked
about with both readings unless the site's han decide it. The region is
the one whose row is seen best; the indicators are ordered by first
sighting.

A kan and an indicator explain each other when they fall within 30 s. A kan
the discard anchor established (a daiminkan: a discard was taken and four
boxes shown; an ankan or kakan: no discard taken, the kan pattern shown)
stands, and so does its indicator: a kan never goes into the log without
one (section 1). When no view shows it — the dead wall is often cut off by
the edge of the overhead (the second VOD's hand 3: the dora 4s lies at the
crop's edge, and the indicator the ankan of 1s revealed is outside every
region) — the indicator is a `dora` question, asked with the kan's time,
and the log carries the best guess meanwhile: a kind with copies left whose
dora keeps the winner's han at the site's (the kind with the most copies
left among those). An
indicator no kan explains is a kan the cameras missed. Its maker is the player of the first discard sighted
after the indicator's last view without it (the rinshan draw is kept or
discarded, then they discard); a pond removal near that time that no call
has taken makes it a daiminkan on that tile, a pair of identical face-up
tiles in the maker's meld camera names an ankan, a prior pon of the maker
makes it a kakan, otherwise it is an ankan whose tile the solver chooses
(never a five: a kan of fives is all four fives including the red one, and
must be named by the camera). A self-kan turn has two draws (the normal
draw, the kan, the rinshan draw) and one discard; a daiminkan has the
rinshan draw only. Human indicator facts prepend to the observed list;
they never hide an observed later indicator.

**Principle: when visual evidence is unclear, look again more closely.** The
2 fps calm reads are the first pass. The decoder reads the video densely
(5 fps, every sampled frame, a short window) at the
place and time that decides it, before asking the human: a called tile that
no pond shows (below), a turn the merge had to skip (that player's pond over
the gap), or a draw with a close feasible alternative or no alternative found
before timeout (below). Dense reads are cached under `work/<video>/dense/`.
The 0.5 boundary is inclusive, with a small floating-point tolerance. Acquisition
tests the cost gap of an alternative actually found; review tests the certified
margin. A distant candidate with a weak proof bound stays reviewable without
automatically triggering a video scan. Margin reuse additionally requires the
same model, baseline objective and preferred draw.
Excluding the boundary can skip the additional resting-hand evidence needed
to distinguish adjacent draw orders; the recorded draw-order regression fixture
covers this case.

Hand observations also obey the merged turn order: a player cannot draw
before the preceding player discards. The preceding pond's last view without
its new tile provides a conservative earliest-draw bound (not the later
first sighting, which an arm can delay). A row with exactly one extra tile entirely before
that bound cannot be an after-draw state or identify a drawn end. It remains
soft partial evidence of the resting state, so misdetections cost disagreement
without moving genuine persistent tiles into a future draw.
Decode artifacts carry `decoder_version`; results from an older or unversioned
decoder are recomputed from observations rather than silently reused. Model
initialization is deferred when all requested artifacts have the current version.

**Uncertain draws: the hand before, the hand after, the discard.** A draw
is the hand after the turn, less the hand before it, plus the discard; so
for a draw with an alternative cost gap at or below 0.5, or no candidate before
timeout, the player's hand row is read at 5 fps
over the whole stretch that shows both hands at rest — from the seat's own
previous discard to the next player's discard (at most 60 s before the
discard and 20 s after it); the draws of one seat whose stretches overlap
are read in one window. There the motion test of the calm stage is not
used (a hand camera watches hands, which are rarely still for its liking):
a **still run** is at least 3 frames (0.6 s) over which the row's reading
does not change — the same count, the same tile in every position (a single
frame may be off in one box when the next reads as the run again; a change
that stays is a change) — and it enters exactly as a calm view of that
moment would (evidence rules of `solver.py` below): the resting count
before the draw is the hand before, the resting count after the discard is
the hand after, one tile more is the moment after the draw with the new
tile at an end, one or two short is an all-but-one (two) row, anything
else a sub-multiset.

An open-kan replacement with no direct draw posterior and no human draw fact
also requests this bounded window, even when its inferred choice has a large
alternative gap. Its tile may otherwise be determined entirely by uncertain
earlier hand counts. The pre-kan resting view and post-discard resting view
constrain the replacement through the existing tile-removal rules; a kan is
not treated as an ordinary draw when mapping intermediate hand states. The
request joins the same per-seat window merge and runs once per reconstruction.
New partial or full evidence triggers a re-solve; an empty/no-op read does not.
This additional rule applies to single-draw daiminkan turns. Self-kans with
two draws and winning daiminkan turns without an ordinary draw variable are
not reinterpreted by it. Human answers remain hard constraints and suppress
the extra acquisition.

**Called tiles never seen calm.** A tile that is called lies in the pond for
a second or two, often inside a disturbed interval, so the calm reads miss
it: that is step 2 of the call anchor. The three other ponds are read on
every frame at 5 fps over the 15 s before the meld appeared
(`read.dense_pond_reads`, cached); a tile of *any* kind that appears in at
least two consecutive frames after the pond's last known tile and then
vanishes is the discard taken. It enters the log with its time, position and
box, names the call's source and its called tile, and the merge treats it
like any other discard. The camera's reading of the called tile only breaks
ties between two such tiles.

**`turns.py`** — merge the four discard logs into one sequence. The order
inside each log is fixed; the interleaving is chosen by dynamic programming
over (positions in the four logs, whose turn it is), subject to the turn
automaton (E S W N; a call passes the turn to the caller, who discards next;
kans add a rinshan draw). The cost is **local**: a discard placed after
another costs the seconds by which it was sighted before the previous one
could have happened (the start of that one's window), capped, so a single
wrong time costs one inversion and never drags the seats after it along.
Three escape moves carry fixed costs above any honest inversion: a seat
passing its turn with no visible discard (a missed discard), a slot left
out of the sequence (a phantom: it is not a discard of this hand), and a
seat whose log has run out while another seat still has discards (that seat
missed a discard: a skip like any other, never a free pass). A removed
slot no call has taken passes the turn to its natural successor only at a
cost (a tile left the pond and nobody took it: a misread removal); passing
it to another seat instead is a **hidden call** by that seat, cheaper, whose
tiles the solver chooses (step 3 of the call anchor). The site's
result fixes the end: after a tsumo the winner is next to play (it drew the
winning tile), after a ron the seat after the loser; a sequence that ends
elsewhere pays a skip per seat in between. The wall is part of the search:
the state counts the draws (every turn that is not a call turn, the
dealer's first included, and every skip), no sequence draws more than the
live wall less the kans (less the winning draw of a tsumo), and an
exhaustive draw ends with the discard of the last draw — that one cannot
be called — so whatever the ponds show after it was laid after the hand
ended (the reveal) and is dropped like a trailing tile after a win. Every
inversion over the tolerance, skip and dropped slot is a note. The
riichi turn per seat is its sideways slot (the overlay counter does not
move during a hand on this layout, section 1).

**`solver.py`** — haipai and draws as a constraint program (OR-Tools CP-SAT).
Per seat s and kind k (37): integer `h0[s,k]` (haipai count, Σk = 13, or 14
for the dealer whose first turn then has no draw variable); per turn j of s
with a draw: booleans `d[s,j,k]` with Σk = 1; discards are known kinds from the
pond (a slot with a weak margin is a small choice set with costs); meld tiles
are known. Hand after turn j: `h[s,j,k] = h0 + Σ draws − Σ discards − Σ meld
removals`, all ≥ 0, Σk = 13 − 3·melds (14 after a kan set). Constraints: the
counting rule Σs(h0 + Σ draws) + indicators ≤ 4 per kind, red fives ≤ 1,
plain fives ≤ 3; after riichi `d = discard`; a call turn has no draw; a kan
has a rinshan draw; the winner's final hand must contain the tiles the site
result implies (checked by `scoring.py` afterwards; on failure the solver is
rerun with the failing hand excluded from the solution pool).

Evidence enters as costs, in this order of strength:
1. **direct draw observations** (a hand row with its newly drawn end tile): cost `w_d · (1 − p_k)` on `d[s,j,k]`, high weight;
   when this observation exists the solver effectively copies it and only
   overrides it if the pond or the next state contradicts it beyond doubt;
2. **full resting-count observations** between the seat's turns, calm or
   read-floor alike (a row showing the whole hand hides nothing):
   Σk |h[s,j,k] − e[k]| where `e[k]` is the expected count (sum of the slot
   posteriors), weighted by quality;
3. **all-but-one (or two) rows** — a calm view between the seat's turns
   that shows one or two tiles fewer than the hand holds (a hand resting
   over an end of the row, a tile standing behind another; hand 7 of the
   second VOD: East's right-end tile under a resting hand for minutes):
   match each observed box to at most one tile in the concealed hand, with
   each kind's hand count limiting how many boxes can use it. A match costs
   `1 − p(kind)` and an unmatched box costs one. This prevents two alternative
   identities of one box from becoming evidence for two separate tiles.
   The cost is doubled to retain the missing-plus-excess weight of a nearly
   complete row. Unobserved tiles add no identity evidence. Such a view does
   not by itself cover a draw: the hidden tile may be the drawn one;
4. **partial observations** — other rows with fewer tiles than the hand
   holds (an arm over part of the row, a read-floor view, a row inside a
   turn's window): the visible tiles must be a sub-multiset of the hand at
   that moment, using the same capacity-constrained box matching at half
   weight. An ambiguous box can use its second choice when another box needs
   the only available copy of its first choice; unseen tiles remain unknown.
   Matching uses the solver's ordinary integer probability resolution.
   Identical rounded posteriors share flow variables to avoid repeated work.
Merged discard and call chronology bounds which turn an observation can
describe; an observation cannot show a draw before that player can act.

Facts from the review tool fix variables. Margin of a decision = objective
increase when that decision is forbidden (one re-solve per draw turn; a hand
solves in seconds); the re-solve's own choice is the runner-up, which the
review item offers next to the best guess. A timed-out feasible re-solve uses
its certified objective lower bound, never the cost of its current candidate:
that candidate can be much worse than an undiscovered equal-cost alternative.
An UNKNOWN search can also supply a certified lower bound without finding a
candidate. Retain that bound; the model's nonnegative costs make an uninitialized
zero bound conservative. Invalid models do not supply confidence.
The candidate still supplies a possible runner-up. A haipai gets a margin the same
way, per seat, by forbidding that exact multiset. Variable pond identities are
also certified by forbidding the selected discard, including when that choice
matches the original pond reading. Draws whose margin is
below the threshold become review items; a draw with no direct
observation, no calm state on either side and no later discard or reveal
that pins it is a `lost` candidate (section 6).

Acquisition uses a separate `alternative_gap`: the cost of the best feasible
alternative found, an upper bound on its true minimum cost gap. A close candidate
(including exactly 0.5) requests a dense video read. An UNKNOWN solve requests
one unless its certified lower bound already settles the choice. A distant
candidate with only a weak proof bound remains uncertain for review but does not
trigger another video read solely because the counterfactual search timed out.
An infeasible alternative has infinite gap. This scheduling heuristic preserves
the previous evidence-acquisition policy without treating a candidate gap as
certified confidence; confidence, completion and review use only the lower bound.

Draw alternatives clone the model built for that solve, inserting the exclusion
at exactly the same constraint position as a full rebuild. Variables, objective,
constraint order and hints are unchanged. Haipai alternatives rebuild because
they introduce variables. A confident prior draw, discard or starting-hand margin
is reusable only if the entire model fingerprint, baseline objective and selected
tile or multiset match; changing soft
evidence can change relative confidence even when the preferred tiles stay put.
The next main solve uses the previous haipai, draws and available repair/meld
choices as hints, never constraints. This gives a timed search a useful starting
assignment after dense evidence arrives while letting new facts override it.
Transient hints are excluded from the model fingerprint; cloned alternatives
clear inherited hints before adding their own, so every variable is hinted once.
A legal incumbent whose optimality was not proved remains exportable but gets
a `solver_incomplete` review item. `solver.optimal` preserves that distinction
even when discard repair changes the display status to `repaired`.
Covered draws and starting hands with low certified margins are grouped into
one `uncertain_tiles` review item. Human-fixed choices, explicit Can't tell
answers and existing individual questions are excluded. Image coverage alone
does not make a reconstruction complete; the grouped item links to its choices.
Uncertified variable discards receive individual `discard` questions, even if
the solver kept the raw pond reading. Fixed pond identities are not assigned
invented margins, and human-confirmed discards stay hard constraints in repair.
An explicit Can't tell answer for a starting hand uses `lost` with
`field=haipai`, scoped to that seat. It marks only the starting-hand confidence
row as lost; it neither supplies tiles nor answers that player's first draw.
The main solve has a 60-second ceiling; simple hands finish as soon as optimality
is proved. Alternative solves keep their four-second ceiling and conservative
bounds. They can stop earlier once a certified objective lower bound exceeds
the inclusive review threshold: further optimization of a rejected candidate
cannot change that confidence decision. An incumbent cost never triggers this
stop. Exported margins can therefore be lower-bound certificates rather than
exact alternative cost differences. Evidence expansion can require multiple solves, so the
main-search ceiling is not a limit on the total time to reconstruct a hand.

**Repair.** A pond reading is evidence too, not a fact: when the hand has no
legal reconstruction with every discard as read (the fifth tile of a kind,
a discard the hand cannot hold), the solver is run again with every
discard a choice over all kinds, costed by its posterior, so the cheapest
set of re-readings that makes the hand legal is found. Those re-readings are
then taken as the discards and the hand is solved as usual (margins and
all); each unconfirmed change is a discard review question showing both the
pond reading and the reconstruction's choice. A discard a call took
is not re-read: the meld camera saw the same tile, so its kind is attested
twice and a contradiction lies elsewhere. Only when no re-reading helps is
the hand a conflict (section 6), with the diagnosis of which constraint
cannot hold.

**Meld tiles.** A call's tiles from the hand are a choice (step 3 of the call
anchor): one boolean per legal composition, exactly one true, the chosen
tiles leaving the hand at the call, each option costed by the camera's
readings. The hand states before and after the call, the count rule and the
later discards then decide what the camera could not; the choice is written
back into the call before scoring and writing.

**The result as a constraint.** The site record says who won, so the
winner's final hand *is* a winning hand: its concealed tiles (the winning
tile included, the melds taken out) decompose into sets and one pair —
integer variables per pon kind, per chi start and per pair kind whose
tiles add up to the concealed counts (red fives counted as fives), with the
sets and the melds four in all — or into seven pairs. At an exhaustive draw
each seat the site lists as tenpai holds, after its last discard, 13 tiles
that one more tile of some kind (a choice variable) would complete. Both are
hard constraints: a reconstruction in which the winner does not win is not a
reconstruction of this hand.

**Riichi on a called tile.** When a seat's turned tile follows its own
discard that was called away, the declaration was that called tile or the
turned one (section 1). Both are solved; the freeze decides between them,
and a choice that is not clear is a `riichi` review item.

Why a solver instead of turn-by-turn arithmetic: a single bad reading then
costs one local error instead of desynchronising every later turn, and every
rule of section 1 is one line of constraint.

**`scoring.py`** — the winner's hand (solver's final state + winning tile +
melds) scored with the `mahjong` library under the ruleset; must reproduce
the site's han/fu (ura dora and the riichi/ippatsu/tsumo context included).
The fu counts at a mangan or more too: the payment ignores it but the site
records it, and a 6/30 reconstruction of a 6/20 pinfu tsumo holds the wrong
hand; only a yakuman's fu means nothing. The yaku list goes into the log. A kakan is the pon it grew from with its
fourth tile: it replaces that pon among the melds, it is not a meld beside
it. At a draw, the site's tenpai seats must be tenpai with their final
states.

**The site can be wrong about han and fu.** The record is typed in by a
scorer, and the slip that costs nothing is the common one: a han/fu pair
that pays what the hand paid (10/40 and 9/70 are both a baiman). So a
`result` question compares payments too: when the reconstructed hand pays
exactly what the site's han/fu pay (same limit, or the same points), the
question says so first, and the reviewer may confirm that the site is wrong
and the reconstruction right (a `site_wrong` fact holding the confirmed
han/fu; it can be given when the payments differ as well, and the question
then names both). From then on the confirmed han/fu replace the site's
everywhere — the result constraint, the score check, the log's value text
and its yaku list — while the deltas stay the site's (the points did move).
The decoder notes the correction; it is the reviewer's decision, never the
program's.

The site's han and fu are a constraint too, applied after the solve: the
reconstruction is the cheapest one whose winning hand scores the site's
value. When the best one scores otherwise, the alternatives are tried in
order of the evidence they cost, and the first that scores the site's
value is taken (a note, not a question):
1. an unseen tsumo tile: the tiles that complete the hand with that score,
   ranked by what the reveal shows beyond the concealed hand (a 6s the hand
   already holds is no evidence that 6s won; the 14th tile read 9p for 9s
   is), a question only when two of them rank close;
2. the next-best reconstructions: the solver is run again with the
   winner's final hand forbidden, then that one too, up to a few times
   and within a cost budget — the hand whose tiles the views read
   ambiguously (a 5s that is also read 8s) is one of them, and the reveal
   frames usually price it close;
3. a five read plain for red or the reverse — the aka dora is one han,
   the classifier's weakest distinction, and at most one red of a suit
   exists anywhere — within the same budget;
4. the dora indicator's next reading.
Each accepted alternative is bound as the winner's final hand and the hand
solved again, so the draws that brought it follow. What none of them
explains is a `result` question, unless the unknown ura of a riichi win
can.

**`review.py`** — questions are the scarcest resource: the program solves
what the video and the rules settle, and asks a person only the question
whose answer settles the most. Uncertain covered draws and starting hands
appear in one grouped `uncertain_tiles` item and individually in the confidence
file with their margins and alternatives. The group keeps completion status
honest without flooding the queue with one question per draw. A tile that
nothing covers is different (section 6, `lost`): a draw no
frame shows and nothing later pins, a discard placed without ever being
sighted, a kan indicator outside every view. The program never writes it
silently as a legal guess: it is a question (`draw`, `discard`, `dora`),
with the guess prefilled and in the log meanwhile, until the reviewer
supplies the tile or answers *Can't tell* (a `lost` fact). A hand that the
video and the rules settle asks nothing but its ura; one that they do not
asks the few questions that settle it — each open decision that no other
answer would settle, ranked by how much of the hand it settles, never one
question per uncertain draw. The kinds, in that order of priority:

1. `ura` — the ura indicators of a riichi win (the video rarely shows them
   readably);
2. a **conflict** — the hand has no legal reconstruction even after repair:
   the one decision the diagnosis names (a discard's identity, a call's
   type, tiles and source, a reviewer's fact that contradicts the video),
   asked with the control that answers it, never as a list of violations;
3. a **result** — the winning hand does not score the site's han/fu for a
   reason other than the unknown ura: the winner's revealed hand, with the
   reveal frame and the winner's melds, and whether the reconstruction pays
   what the site paid (then the site's han/fu is likely the slip, and
   *The site is wrong* is the first answer offered); or the tile the log ends on is
   uncertain: a tsumo whose winning draw the reveal reads no better than
   another tile that completes the hand with the same score (reference
   hand 10: the 14th tile read 9p, the winners 6s and 9s) — which one;
4. a **call** the anchor could not settle (a meld event whose discard no
   read found, or a hidden call): its type, tiles and source.

Item kinds: `ura`, `conflict`, `result`, `call`, `discard`, `dora`, `kan`,
`riichi`, `order`, `draw`, `haipai`, `lost`, `uncertain_tiles`, `solver_incomplete`. Each item: hand, seat, time(s),
question, candidates with costs, evidence crops (region, t), the best guess,
and the control it is answered with. Facts file entries mirror the kinds and
are applied as hard constraints on the next decode, after everything the
program inferred (a fact always wins). The confidence rows of section 5 hold
every decision: one per draw, discard, haipai, call and indicator, with its
margin, the evidence it rests on, `human` when a fact fixed it and `lost`
when nothing did.

**`indicators.py`** — the dead-wall row and the kans it implies (above).

**`decode.py`** — runs ponds → melds → indicators → turns → solver →
scoring → review for one hand and writes `decode/<hand>.json`.

### 4.9 `tenhou6.py`

Model + writer for tenhou.net/6 (tiles 11–47, 51–53; tsumogiri 60; riichi
`r`; call strings `c`/`p`/`m`/`a`/`k` with the called position; results 和了 /
流局 with deltas, han/fu text and yaku). The deltas are the site's with the
riichi deposits taken out: the site charges a declarer's 1000 in the hand's
deltas, the viewer charges it at the `r` discard. The value text is
computed from the site's han and fu (the reviewer's, when a `site_wrong`
fact corrected them) under the ruleset, in tenhou's form:
`30符2飜2000点` for a ron, `30符2飜500-1000点` for a non-dealer tsumo,
`30符2飜1000点∀` for a dealer tsumo, `満貫…` and above by name.

And a **replayer**: given a log, simulates every hand turn by turn in the
real interleaving (the dealer first; after each discard, a call in another
seat's draw list naming that tile from that seat takes the turn) and
reports violations: haipai and final hand sizes, a draw or discard out of
turn, a discard not in hand, more discards than draws, a call on a tile
that is not the last discard or from the wrong seat (chi only from
kamicha), a call without its tiles in hand, count over four with the
indicators, riichi with an open hand, a hand-tile discard after riichi, a
winner whose hand does not win, wall draws plus kans over 70, and
deltas that do not balance. `convert` replays each kyoku on its own, leaves
out of the log every kyoku that is a conflict or that the replayer rejects
(section 6: nothing is written for it), and files each violation as a
review item of that hand.

Dealer split: the dealer's 14-tile haipai is written as 13 tiles plus a
first draw. The first draw is the dealer's first discard when that tile is
among the 14 (the first discard then shows as tsumogiri, which is the only
thing the split affects); otherwise any tile of the 14. The confidence file
marks the split as arbitrary.

### 4.10 Review and label tool (`tool/`)

The local studio uses the standard-library HTTP server. Its recording selector
and Review, Results and Settings tabs own navigation; embedded review suppresses
its standalone page header and global statistics.

- **Review** opens directly for processed recordings. A compact hand/question
  navigator leads to one full-frame video beside the relevant answer controls.
  Tile palettes, starting/final-hand editors, indicators and meld editors save
  explicit human facts. Additional crops and less common corrections stay
  collapsed. Review prioritizes blocking conflicts, then choices without observed
  evidence, then increasing certified confidence margin. Unscored diagnostics
  follow; equal-ranked choices retain stable hand/item order. Grouped draws and
  starting hands keep their original choice identifiers. Each choice needs its
  own answer. Search
  incompleteness is a retry action, never an arbitrary tile acknowledgement.
- **Rebuild changes** appears when saved changes have not been applied. The
  server selects those hands and rebuilds them in one background job, then
  refreshes whole-project exports. While changes are pending, this is the sole
  rebuild action in embedded review. Otherwise, **Rebuild hand** retries the
  selected hand's reconstruction. A hanchan with pending changes withholds
  replay/download links until rebuilding has incorporated those changes.
- **Hand inspection** shows editable starting hands and a chronological turn
  table. Selecting a turn opens its evidence and correction controls. Displayed
  hand/hanchan numbers are one-based; stored IDs remain zero-based. Low-confidence
  draws remain marked until their particular choice has a human answer.
- **Results** provides hanchan replay/download actions and individual hand links.
  Reports are collapsed below them.
- **Calibration** is the first step after a new recording is prepared, and is
  available later through Settings. The full-frame plate shows region borders;
  drag to move or resize them. **Check borders** saves only changed regions
  before testing them. An unchanged check does not write a human fit. **Discard
  changes** reloads saved geometry. Overhead controls and crop previews are
  collapsed. Unsaved geometry blocks navigation and analysis; saved geometry
  changes require analysis to refresh the affected video evidence.
- **Training labels** remain an advanced feature of standalone review. Region
  views can prefill model boxes; a human corrects identities and geometry before
  saving frame-coordinate annotations. Existing human labels retain an
  `.orig.json` backup. These annotations are training inputs, not review answers.

Storage: `labels/<video>/facts.jsonl` (kinds draw, discard, haipai,
final_hand, ura, dora, riichi, lost, site_wrong, note) and `labels/<video>/boxes/*.json`.
The API (`/api/hands`, `/api/items`, `/api/hand/<i>`, `/api/frame`,
`/api/read`, `/api/facts`, `/api/label`, `/api/decode/<i>`) returns JSON free
of NaN so the browser can parse it; static files are served with
`Cache-Control: no-store`.

### 4.11 Training (`train/`)

`python -m video2tenhou.train.data` builds upright region crops from
frame-coordinate labels and splits by hand, keeping adjacent views out of
separate splits. The standard face detector uses `train.face_data` to assemble
its dataset and `train.train_libreyolo` to train it. `train.train_classifier`
trains the tile classifier; `train.classifier_context` provides the context
refinement recipe. These module names are relative to `video2tenhou`.
Model identities are part of the reading-cache key. See the exact commands
and expected inputs in [maintenance](MAINTENANCE.md#training).

### 4.12 Evaluation (`eval.py`) and CLI

`python -m video2tenhou.eval` exposes detector, perception and observation
checks against local annotations. `convert` runs stages 1–6 with caching;
`web` opens the full workflow, `review` opens one recording, `calib check`
renders calibration checks, and `download` fetches a VOD.

## 5. Data formats (all JSON / JSONL, times in seconds of the video)

- `overlay.jsonl`: `{t, scores:{TL..}, winds:{TL..}, round, kyoku, honba, sticks, nicks?}`
- `hands.json`: `[{hand, game, kyoku, honba, sticks, t_start, t_end, t_overlay:[t0, t1], corner_wind:{TL:"E",...}, scores:{seat:..}, site_index}]`.
  `t_start`/`t_end` is the **read window**: it starts 60 s before the
  overlay switch (the overlay lags the table), so consecutive windows
  overlap. `t_overlay` is the overlay segment itself; segments are disjoint
  and a label is attributed to a hand by them (`train.data.hand_of`). The
  physical play window is found in the ponds at decode time
  (`engine.ponds.play_window`), not stored here.
- `work/<video>/plate.png`: the table plate, the median of 60 frames spread over the video.
- `labels/<video>/calib.json`: this video's fit —
  `{video, layout, overhead:{center:[x,y], angle, scale, iou}, hand:{corner:{rect,roll}}, meld:{corner:{rect}}, cam:{corner:rect}, source, ts}`.
  Absent means the layout's own numbers are used, which is only right for the video the layout was drawn on.
- `calm.jsonl`: `{region, t0, t1, n, motion, skin, calm: bool, partial: bool}`; `calm_scores.npz` carries
  region geometry (`calm.geometry_key`) and source-content SHA-256. The source hash is refreshed at
  stage entry, so replacing bytes at the same path invalidates scores even if size/timestamps match.
  Changed geometry, changed source or missing provenance requires recomputation; threshold
  changes can reuse the same proven scores. Scores are published by atomic replacement after a
  complete write; corrupt or incomplete arrays are cache misses, and a failed recomputation leaves
  the previous complete score file intact.
- `reads/<hand>/<region>.jsonl`: `{t, region, size:[w, h], rejected, boxes:[{xyxy, conf, sideways, role, row?, col?, group?, p:[39 floats]}]}`;
  `role` ∈ tile | indicator | extra (a hand's meld beside the row) | other (not of this region); `rejected`
  marks a pond reading whose rows cannot exist (seven positions). `reads/<hand>/done.json`:
  `{hand, readings, detector, classifier, cap, geometry, identity}`; cache validation also binds source,
  exact sampling plan and recognition/runtime identity. A changed recognition input re-reads the hand.
  `dense/<t0>_<t1>_<hash>.json`: the signature binds the exact window, frame rate, requested regions,
  source, recognition/runtime identity, geometry and dense-stage evidence policy.
- `obs/<hand>.json`: `{region: [{t0, t1, iv_t0, iv_t1, n_readings, n_used, count, quality, partial, slots:[{key, tile, p:[39], sideways, disagree, xyxy}], indicators:[..], extra:[[slot]]}]}`;
  `t0`/`t1` are the first and last reading used (reads are clipped to the hand window), `iv_t0`/`iv_t1`
  the calm interval; `extra` holds the groups a hand reading found beside the row (melds moved at a reveal).
- `decode/<hand>.json`: `{decoder_version, hand, play_window, turns:[{i, seat, j, kind, t, draw, margin, discard, tsumogiri, riichi, call?, own_call?}], haipai:{seat:[..]}, haipai_margin:{seat:m}, dora:[..], ura:[..], indicators:[..], result, score, solver:{status ∈ optimal|feasible|repaired|infeasible}, items:[...], confidence:[...]}`;
  an item is `{kind, seat?, t?, j?, text, tile?, candidates?:[{tile, cost}], evidence?:[{region, t}]}`
- `labels/<video>/facts.jsonl`: `{kind, hand, seat?, turn?, slot?, tile|tiles|order, note, author, ts}`
- `labels/<video>/boxes/<kind>_<corner>_<t>.json` (kind = pond | hand | meld):
  `{video, t, kind, corner, boxes:[{quad:[[x,y]×4] in 1080p frame pixels, tile, sideways, role: tile|indicator, row?, col?}], rows? (pond tokens, "-" = gap), tiles? (hand, left to right), melds?, revealed?, occluded?, source}`.
  Frame coordinates keep the labels independent of any panel geometry; the
  dataset builder maps quads into the current calibration's regions.
- `g<k>.confidence.json`: `{hand:[{seat, turn, field ∈ draw|discard|haipai|call|dora, value, margin, human, lost, evidence:[{region,t}]}]}`
  (a margin of `null` means none was computed; the dealer's 13 + 1 split is a `first_draw` row marked arbitrary).
  A hand left out of the log (a conflict, or rejected by the replayer) has no rows.

Tiles in all internal files use the notation of section 1; tenhou numbers
appear only in the log.

## 6. Failure taxonomy

- **Noise** (a reading disagrees with the state at rest): out-voted in
  `observe`, or repaired by the solver; reported only through margins.
- **Review** (evidence exists but the margin is small or two signals
  disagree): a review item with the crops; the human answers; the fact is
  permanent for that video.
- **Lost** (no evidence covers the decision and no later evidence pins it:
  the draw moment was not captured, no calm state exists on either side, and the tile is never
  discarded, called or revealed; or a kan indicator outside every view):
  the solver still emits a tile (the cheapest legal one) so the log stays
  valid, marks it `lost`, and asks it: a `draw`, `discard` or `dora`
  question with the whole frame over the disturbed interval, so the human
  can either supply the tile from what they see or confirm it is
  unrecoverable. Until then the hand is `review`, never `complete`.
- **Bad geometry** (a region's border cuts tiles, or a pond holds a
  neighbour's block): not a decode problem at all — every reading of that
  region is wrong and the hand is unrecoverable from it. Stage 0's check
  catches it before stage 1 and `convert` stops with the region named; it is
  never reported as a review item, because no answer to a question about
  tiles can repair a crop.
- **Conflict** (constraints unsatisfiable even after repair: site record and
  video disagree, replayer rejects the log, calibration does not fit): the
  run stops for that hand with a `conflict` entry in the report; nothing is
  written for it.

## 7. Validation

See [maintenance](MAINTENANCE.md#tests) for the executable validation workflow
and [performance](PERFORMANCE.md) for measured workloads and remaining limits.
Validate changed behavior on held-out inputs as well as regression fixtures.

## 8. Design rationale

- Two-stage perception (detector + shared face classifier) instead of one
  class-aware detector per view.
- Hands are read in raw camera space per calm interval; no global row warp.
- Draws use end-tile evidence and changes in the hand multiset, constrained
  by turn chronology; dense reads supply additional evidence when needed.
- The dealer's haipai is 14 tiles; the 13 + 1 split is a write-time detail.
- Readings that disagree at rest are the misrecognition detector, not noise
  to drop.
- Reconstruction is a constraint program over the whole hand, not
  turn-by-turn arithmetic.
- The site record is a required input; the overlay is for timing and seat
  mapping.

## Package and data boundaries

Runtime calibration and overlay templates ship inside `video2tenhou/assets`.
Writable data is rooted at `VIDEO2TENHOU_HOME` (the launch directory by default),
never in site-packages: `models/`, `labels/`, `work/`, and `out/` are local.
Only curated regression fixtures belong in `tests/data`; personal labels, video,
trained weights and analysis logs do not belong in the source distribution.
The browser studio owns import, calibration, analysis, review and export; CLI
commands remain available for scripted use. See the adopter and maintenance guides
for the supported workflow and validation boundaries.
