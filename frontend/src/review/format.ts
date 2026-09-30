import type { HandEntry, ReviewChoice } from "../types";
export const seats = [..."ESWN"];
const winds: Record<string, string> = {
  E: "East",
  S: "South",
  W: "West",
  N: "North",
};
export const cornerOf = (entry: Partial<HandEntry>, seat?: string) =>
  Object.entries(entry.corner_wind || {}).find(
    ([, wind]) => wind === seat,
  )?.[0];
export function seatName(entry: Partial<HandEntry>, seat?: string) {
  const corner = cornerOf(entry, seat);
  const name = (corner ? entry.nicks?.[corner] : undefined) || corner;
  return `${winds[seat || ""] || seat || ""}${name ? ` (${name})` : ""}`;
}
export function time(seconds?: number) {
  const value = Math.round(seconds || 0);
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}
export const roundName = (entry: HandEntry) =>
  `${["East", "South", "West", "North"][Math.floor(entry.kyoku / 4)] || "Round"} ${(entry.kyoku % 4) + 1}, ${entry.honba} honba`;
const decisionLabels: Record<string, string> = {
  solver_incomplete: "Search incomplete",
  uncertain_tiles: "Uncertain tiles",
};
export const decisionLabel = (item: ReviewChoice) =>
  item.field === "haipai" || item.kind === "haipai"
    ? "Starting hand"
    : ["draw", "discard"].includes(item.field || item.kind || "")
      ? `${item.field === "draw" || item.kind === "draw" ? "Draw" : "Discard"} ${(item.j ?? 0) + 1}`
      : decisionLabels[item.kind || ""] || item.kind || item.field;
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
