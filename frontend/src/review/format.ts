import type { HandEntry, ReviewChoice } from "../types";
export const seats = [..."ESWN"];
const winds: Record<string, string> = {
  E: "East",
  S: "South",
  W: "West",
  N: "North",
};
export const cornerOf = (entry: Partial<HandEntry>, seat?: string | null) =>
  Object.entries(entry.corner_wind || {}).find(
    ([, wind]) => wind === seat,
  )?.[0];
export function seatName(entry: Partial<HandEntry>, seat?: string | null) {
  const corner = cornerOf(entry, seat);
  const name = (corner ? entry.nicks?.[corner] : undefined) || corner;
  return `${winds[seat || ""] || seat || ""}${name ? ` (${name})` : ""}`;
}
/** Video time as m:ss. */
export function time(seconds?: number | null) {
  const value = Math.max(0, Math.round(seconds || 0));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}
/** Parse seconds, m:ss or h:mm:ss; null when the text is not a time. */
export function parseTime(text: string): number | null {
  const value = text.trim();
  if (!/^\d+(?::[0-5]\d){0,2}(?:\.\d+)?$/.test(value)) return null;
  return value.split(":").reduce((total, part) => total * 60 + Number(part), 0);
}
export const roundName = (entry: Pick<HandEntry, "kyoku" | "honba">) =>
  `${["East", "South", "West", "North"][Math.floor(entry.kyoku / 4)] || "Round"} ${(entry.kyoku % 4) + 1}, ${entry.honba} honba`;
const decisionLabels: Record<string, string> = {
  haipai: "Starting hand",
  result: "Final hand",
  call: "Meld",
  kan: "Kan",
  riichi: "Riichi",
  dora: "Dora indicators",
  ura: "Ura indicators",
  conflict: "Conflict",
  order: "Missed discard",
};
export function decisionLabel(item: ReviewChoice) {
  const kind = item.field || item.kind || "";
  if (kind === "draw" || kind === "discard")
    return `${kind === "draw" ? "Draw" : "Discard"} ${(item.j ?? 0) + 1}`;
  return decisionLabels[kind] || kind;
}
const handStatus: Record<string, string> = {
  conflict: "Conflict",
  unresolvable: "Not solved",
  review: "Questions",
  complete: "Complete",
};
export const handStatusLabel = (status: string | null, pending: boolean) =>
  pending ? "Updating" : status ? handStatus[status] || status : "Not analyzed";
const factLabels: Record<string, string> = {
  draw: "Draw",
  discard: "Discard",
  haipai: "Starting hand",
  final_hand: "Final hand",
  meld: "Meld",
  meld_remove: "Meld removed",
  missing_discard: "Missed discard",
  riichi: "Riichi seats",
  riichi_turn: "Riichi discard",
  kan_time: "Kan after discard",
  ura: "Ura indicators",
  dora: "Dora indicators",
  lost: "Can't tell",
  site_wrong: "Site score corrected",
  dismiss: "Question dismissed",
  note: "Note",
};
export const factLabel = (kind: string) => factLabels[kind] || kind;
export const paletteRows = [..."mps"].map((suit) =>
  [1, 2, 3, 4, 5, 0, 6, 7, 8, 9].map((rank) => rank + suit),
);
paletteRows.push(Array.from({ length: 7 }, (_, index) => `${index + 1}z`));
const suitPrefixes: Record<string, string> = { m: "Man", p: "Pin", s: "Sou" };
export function tileImage(tile: string) {
  const [rank, suit] = tile || "";
  const prefix = suitPrefixes[suit];
  const name =
    prefix && /^\d[mps]$/.test(tile)
      ? prefix + (rank === "0" ? "5-Dora" : rank)
      : /^[1-7]z$/.test(tile)
        ? ["Ton", "Nan", "Shaa", "Pei", "Haku", "Hatsu", "Chun"][
            Number(rank) - 1
          ]
        : tile === "X"
          ? "Back"
          : null;
  return name ? `/tiles/${name}.svg` : null;
}
