import type { FactBody } from "../types";
interface MeldInput {
  hand?: number;
  seat?: string;
  t?: number;
  type: string;
  tiles: string[];
  source?: string;
}
export function meldFact({
  hand,
  seat,
  t,
  type,
  tiles,
  source,
}: MeldInput): FactBody {
  tiles = [...tiles];
  if (type === "chi") source = "kamicha";
  const count = ["kan", "ankan", "kakan"].includes(type) ? 4 : 3;
  if (count === 4 && tiles.length === 3) tiles.push(tiles[0]);
  if (
    tiles.length !== count ||
    tiles.some((tile) => !/^(?:[0-9][mps]|[1-7]z)$/.test(tile))
  )
    throw new Error(`A ${type} has ${count} tiles: fill every tile first.`);
  const positions: Record<string, number> = {
    kamicha: 0,
    toimen: 1,
    shimocha: type === "pon" ? 2 : 3,
  };
  const position =
    type === "ankan" ? null : type === "chi" ? 0 : positions[source || ""];
  if (position) tiles.splice(position, 0, tiles.splice(0, 1)[0]);
  return {
    kind: "meld",
    hand,
    seat,
    t,
    type,
    tiles,
    source: type === "ankan" ? null : source,
    called_pos: position,
  };
}
