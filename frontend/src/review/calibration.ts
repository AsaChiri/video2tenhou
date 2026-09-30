import type { Calibration, Drag, Overhead, Point, Rectangle } from "../types";
const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value));
export const MIN_REGION_SIZE = 20;
// Coordinates remain in source-frame pixels, independently of canvas display size.
export function hitRegion(
  data: Calibration,
  x: number,
  y: number,
): Drag | null {
  for (const [name, region] of Object.entries(data.regions)) {
    if (!region.movable) continue;
    const corner = region.quad.findIndex(
      (point) => Math.abs(x - point[0]) < 14 && Math.abs(y - point[1]) < 14,
    );
    if (corner >= 0)
      return { name, mode: "corner", i: corner, x, y, rect: [...region.rect] };
  }
  for (const [name, region] of Object.entries(data.regions)) {
    const [left, top, width, height] = region.rect || [];
    if (
      region.movable &&
      x > left &&
      x < left + width &&
      y > top &&
      y < top + height
    )
      return { name, mode: "move", x, y, rect: [...region.rect] };
  }
  return null;
}
export function dragRegion(
  drag: Drag,
  x: number,
  y: number,
  frame: Point,
): Rectangle {
  const [left, top, width, height] = drag.rect;
  const [frameWidth, frameHeight] = frame;
  if (drag.mode === "move") {
    const w = Math.min(width, frameWidth),
      h = Math.min(height, frameHeight);
    return [
      clamp(Math.round(left + x - drag.x), frameWidth - w),
      clamp(Math.round(top + y - drag.y), frameHeight - h),
      w,
      h,
    ];
  }
  x = clamp(x, frameWidth);
  y = clamp(y, frameHeight);
  let ax = clamp(left, frameWidth),
    ay = clamp(top, frameHeight),
    cx = clamp(left + width, frameWidth),
    cy = clamp(top + height, frameHeight);
  if (drag.i === 0) {
    ax = x;
    ay = y;
  } else if (drag.i === 1) {
    cx = x;
    ay = y;
  } else if (drag.i === 2) {
    cx = x;
    cy = y;
  } else {
    ax = x;
    cy = y;
  }
  return [
    Math.round(Math.min(ax, cx)),
    Math.round(Math.min(ay, cy)),
    Math.round(Math.abs(cx - ax)),
    Math.round(Math.abs(cy - ay)),
  ];
}
export function calibrationError(
  data: Calibration,
  overhead?: Overhead,
): string | null {
  const [width, height] = data.frame;
  for (const [name, region] of Object.entries(data.regions)) {
    if (!region.movable) continue;
    const [x, y, w, h] = region.rect;
    if (
      !region.rect.every(Number.isFinite) ||
      w < MIN_REGION_SIZE ||
      h < MIN_REGION_SIZE
    )
      return `${name}: each side must be at least ${MIN_REGION_SIZE} pixels.`;
    if (x < 0 || y < 0 || x + w > width || y + h > height)
      return `${name}: move or resize the box to keep it inside the image.`;
  }
  if (overhead) {
    if (
      ![...overhead.center, overhead.angle, overhead.scale].every(
        Number.isFinite,
      ) ||
      overhead.scale <= 0
    )
      return "Enter finite overhead coordinates and a positive scale.";
    const [x, y] = overhead.center;
    if (x < 0 || y < 0 || x > width || y > height)
      return "Keep the overhead centre inside the image.";
  }
  return null;
}
export function calibrationBody(
  data: Calibration,
  dirty: Iterable<string>,
  overhead?: Overhead,
) {
  const error = calibrationError(data, overhead);
  if (error) throw new Error(error);
  const body: Record<
    string,
    Record<string, { rect: Rectangle; roll?: number }>
  > = {};
  for (const name of dirty) {
    const [kind, corner] = name.split(":"),
      region = data.regions[name];
    (body[kind] ??= {})[corner] = { rect: region.rect };
    if (kind === "hand" && region.roll != null)
      body[kind][corner].roll = region.roll;
  }
  return overhead ? { ...body, overhead } : body;
}
