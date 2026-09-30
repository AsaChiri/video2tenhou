# Adapting another broadcast layout

Copy `src/video2tenhou/assets/calib/pml.json` to a local file such as
`layouts/my-table.json`. Keep the shipped layout unchanged. Select the file in
advanced studio setup, or pass `--calib layouts/my-table.json` to CLI commands.
The studio resolves relative paths from its data directory; CLI paths are relative
to the launch directory. Use an absolute path when those directories differ.

This configures geometry for a four-player composite with overhead, hand and
meld cameras. A single-camera broadcast, new score provider or different
rules also needs code or model adaptation. The starting dealer must occupy
the top-left chair; starting South, West and North occupy BL, BR and TR. The record adapter requires scoremj;
another provider must supply the `record.Game` contract.

## Coordinate contract

Coordinates use a normalized 1920×1080 BGR frame. Rectangles are
`[x, y, width, height]`, not opposite corners or fractions. TL/TR/BL/BR identify
fixed camera corners. Winds change by round; player identity does not.

| Field | Meaning |
|---|---|
| `frame` | Keep `[1920, 1080]`; frames are normalized to this size. |
| `cam` | Full player camera panels in frame pixels. |
| `hand`, `meld` | Frame-pixel `rect`, rendering `scale`; hand also has `roll` in degrees. |
| `overhead.center`, `angle`, `scale`, `side` | Frame-to-table transform and output square size. |
| `overhead.unit` | Centre-unit rectangle in transformed overhead pixels. |
| `pond` | Per-corner `rect` in overhead pixels, clockwise quarter-turn `rot`, and `scale`. |

In the owner's upright pond view, row zero is nearest the centre unit. Rows
grow toward the player; columns run from the player's left to right. Verify
all four corners. Labels store full-frame quads so crop changes preserve
annotation coordinates.

## Fit, inspect, validate

1. Use representative frames with tiles in every view to set rough panel
   rectangles and the overhead transform in the custom JSON.
2. Run `uv run video2tenhou calib fit recording.mp4 --calib layouts/my-table.json`.
   The fitter expects a dark centre unit on saturated felt near frame centre.
   Other tables need an adapted fitter or a manually measured fit.
3. Inspect the table preview in browser calibration. Leave margins around tiles
   and exclude neighbouring ponds. Save the per-video fit.
4. Run `uv run video2tenhou calib check recording.mp4 --calib layouts/my-table.json`.
   Add `--t 120 --t 1800 --t 3600` with representative early, middle and late play times
   to generate additional contact sheets under `work/calib/`. The border check
   independently samples recognized play. Inspect warnings; failures block
   normal analysis. `--skip-fit-check` is for controlled debugging.
5. Analyze a representative game and compare against independent human ground
   truth, including red fives, called tiles, kans and low light, before calling
   the layout supported.

The layout describes stable table geometry. `labels/<video-stem>/calib.json`
records placement for one broadcast and overrides named overhead/panel fields;
pond rectangles remain properties of the table. Give CLI videos unique stems
so their local labels do not collide.

## Table fitting and model changes

The centre-unit template is PML-specific. `calibfit.write_unit_template`
rebuilds it from a trusted reference calibration. Review the resulting fit
and border checks before shipping it with a layout.

For different tile faces or perspectives, label examples in the browser and
retrain. Hold out whole hands or recordings, not neighbouring frames. Keep
calibration, model checksums, label conventions and measured accuracy together
for every supported layout.
