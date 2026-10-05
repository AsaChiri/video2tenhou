import type { Decode, HandEntry, ReviewItem, Turn } from "../types";

/** Video window showing a turn's draw: from before the previous turn to the discard. */
export function drawWindow(
  turn: Pick<Turn, "t" | "t_prev">,
  entry: Pick<HandEntry, "t_start">,
  decode: Pick<Decode, "t_last">,
): [number, number] {
  const t = turn.t;
  return [
    Math.max(entry.t_start || 0, (turn.t_prev ?? t - 15) - 6),
    t + (t >= (decode.t_last ?? 0) - 1 ? 30 : 4),
  ];
}

/** When the end of a hand (final hands, indicators) is shown. */
export const handEnd = (decode: Decode, entry: HandEntry) =>
  decode.t_last ?? decode.play_window?.[1] ?? entry.t_end;

/** Video window for answering a review question. */
export function questionWindow(
  item: ReviewItem & { t: number },
  entry: HandEntry,
  decode: Decode,
  turn?: Turn,
): [number, number] {
  const t = item.t;
  if (item.kind === "draw")
    return drawWindow({ t, t_prev: turn?.t_prev }, entry, decode);
  if (item.kind === "discard") return [t - 8, t + 6];
  if (["result", "ura", "dora"].includes(item.kind)) {
    if (item.guess) return [t - 5, t + 25];
    const end = handEnd(decode, entry);
    return [end - 3, end + 30];
  }
  return [Math.max(entry.t_start || 0, t - 10), t + 10];
}
