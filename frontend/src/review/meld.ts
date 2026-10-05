import type { FactBody } from "../types";
interface MeldInput {
  hand?: number;
  seat?: string;
  t?: number;
  type: string;
  tiles: string[];
  source?: string | null;
}
const isKan = (type: string) => ["kan", "ankan", "kakan"].includes(type);
/** Index of the sideways (called) tile as laid, given the caller's source. */
function calledPosition(type: string, source?: string | null) {
  if (type === "ankan") return null;
  if (type === "chi") return 0;
  const positions: Record<string, number> = {
    kamicha: 0,
    toimen: 1,
    shimocha: type === "pon" ? 2 : 3,
  };
  return positions[source || ""];
}
function checked(type: string, tiles: string[]) {
  const count = isKan(type) ? 4 : 3;
  if (count === 4 && tiles.length === 3) tiles.push(tiles[0]);
  if (
    tiles.length !== count ||
    tiles.some((tile) => !/^(?:[0-9][mps]|[1-7]z)$/.test(tile))
  )
    throw new Error(`A ${type} has ${count} tiles: fill every tile first.`);
  return tiles;
}
/** A meld fact from tiles entered with the called tile first. */
export function meldFact({
  hand,
  seat,
  t,
  type,
  tiles,
  source,
}: MeldInput): FactBody {
  tiles = checked(type, [...tiles]);
  if (type === "chi") source = "kamicha";
  const position = calledPosition(type, source);
  if (position) tiles.splice(position, 0, tiles.splice(0, 1)[0]);
  return {
    kind: "meld",
    hand,
    seat,
    t,
    type,
    tiles,
    source: type === "ankan" ? undefined : (source ?? undefined),
    called_pos: position,
  };
}
/** Confirm a meld exactly as the reconstruction laid it out. */
export function confirmedMeld({
  hand,
  seat,
  t,
  type,
  tiles,
  source,
}: MeldInput): FactBody {
  return {
    kind: "meld",
    hand,
    seat,
    t,
    type,
    tiles: checked(type, [...tiles]),
    source: type === "ankan" ? undefined : (source ?? undefined),
    called_pos: calledPosition(type, source),
  };
}
