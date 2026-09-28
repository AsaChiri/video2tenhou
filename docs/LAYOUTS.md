# Adapting another broadcast layout

Copy `src/video2tenhou/assets/calib/pml.json` to a local file such as
`layouts/my-table.json`. Keep the shipped layout unchanged. Select the file in
advanced studio setup, or pass `--calib layouts/my-table.json` to CLI commands.
The studio resolves relative paths from its data directory; CLI paths are relative
to the launch directory. Use an absolute path when those directories differ.

This configures geometry for a four-player composite with overhead, hand and
meld cameras and a PML-style overlay. A single-camera broadcast, new score
provider, different overlay font/arrangement or different rules also needs
code or model adaptation. The record adapter requires scoremj;
another provider must supply the `record.Game` contract.

## Coordinate contract

Coordinates use a normalized 1920×1080 BGR frame. Rectangles are
`[x, y, width, height]`, not opposite corners or fractions. TL/TR/BL/BR identify
fixed camera corners. Winds change by round; player identity does not.

| Field | Meaning |
|---|---|
| `frame` | Keep `[1920, 1080]`; frames are normalized to this size. |
| `overlay.strip` | Search strip containing each wind glyph and score/name block. |
| `overlay.score`, `name`, `wind` | Nominal rectangles for previews/template work. |
| `overlay.round_wind`, `round_num`, `honba`, `sticks` | Header crops read directly. |
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

## Overlay and model changes

`timeline.scan` passes the selected calibration to `overlay.read_overlay`.
Corner text is located relative to its wind glyph inside the configured strip.
Moving a strip works; rearranging the text around the glyph requires changing
`read_corner`. Glyph sizes, white text masking and templates are PML assumptions
in `overlay.py`. Different fonts need new templates/reader and regression frames.
The centre-unit template is also PML-specific.

Source-maintenance helpers for rebuilding these packaged assets are
`overlay.save_digit_templates`, `overlay.save_wind_templates`, and
`calibfit.write_unit_template`. Supply known text/wind labels and a trusted
reference calibration; these are offline asset-authoring utilities, not part of
normal video processing. Review regenerated assets and regression frames before
shipping them with a layout.

For different tile faces or perspectives, label examples in the browser and
retrain. Hold out whole hands or recordings, not neighbouring frames. Keep
calibration, model checksums, label conventions and measured accuracy together
for every supported layout.
