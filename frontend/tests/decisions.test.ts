import type { Decision } from "../src/types";
import { describe, expect, it } from "vitest";
import { isAnswered, reviewDecisions } from "../src/review/decisions";
import { meldFact } from "../src/review/meld";
import { gameIds, timeRange } from "../src/studio/validation";

describe("review decisions", () => {
  it("does not answer a different identified turn merely because its timestamp is close", () => {
    expect(
      isAnswered({ hand: 0, kind: "draw", seat: "E", j: 0, t: 30 }, [], {
        0: [{ kind: "draw", seat: "E", j: 2, t: 32, tile: "2p" }],
      }),
    ).toBe(false);
  });
  it("keeps every unresolved choice in review regardless of the candidate gap", () => {
    const item = {
      hand: 0,
      kind: "uncertain_tiles",
      choices: [
        {
          field: "draw",
          seat: "E",
          j: 0,
          margin: 0,
          runner_up: "2p",
          alternative_gap: 80,
        },
        {
          field: "draw",
          seat: "E",
          j: 1,
          margin: 0,
          runner_up: "2p",
          alternative_gap: 0.2,
        },
        {
          field: "draw",
          seat: "E",
          j: 2,
          margin: 0,
          runner_up: null,
          alternative_gap: 0,
        },
        { field: "haipai", seat: "E", j: -1, margin: 0 },
      ],
    };
    const decisions = reviewDecisions([item], [], {});
    expect(decisions).toHaveLength(4);
    expect(decisions[0].decision.field).toBe("haipai");
    expect(decisions.map((row) => row.decision.j)).toEqual([-1, 0, 1, 2]);
    expect(isAnswered(item, [], {})).toBe(false);
  });
  const hands = [0, 1].map((hand) => ({ hand, decoded_at: 10 }));
  const items = [
    { hand: 0, idx: 0, kind: "ura", margin: null },
    {
      hand: 0,
      idx: 1,
      kind: "uncertain_tiles",
      choices: [
        { field: "haipai", seat: "N", j: -1, margin: 0.4, value: [] },
        {
          field: "draw",
          seat: "N",
          j: 0,
          margin: 0,
          alternative_gap: 90,
          value: "2p",
        },
        {
          field: "draw",
          seat: "N",
          j: 1,
          margin: 0.2,
          alternative_gap: 0,
          value: "1z",
        },
        { field: "draw", seat: "N", j: 2, margin: 0.2, value: "9m" },
      ],
    },
    { hand: 0, idx: 2, kind: "discard", seat: "N", j: 9, margin: 0.1 },
    { hand: 1, idx: 0, kind: "draw", seat: "N", j: 3, margin: 0.5, lost: true },
    { hand: 1, idx: 1, kind: "discard", seat: "N", j: 4, margin: 0 },
    { hand: 1, idx: 2, kind: "conflict" },
    { hand: 1, idx: 3, kind: "draw", seat: "N", j: 5, margin: 0.15 },
    { hand: 1, idx: 4, kind: "solver_incomplete", alternative_gap: 0 },
    { hand: 1, idx: 5, kind: "draw", seat: "N", j: 6, lost: true },
  ];
  const facts = {
    0: [{ kind: "draw", seat: "N", j: 0, ts: 9 }],
    1: [{ kind: "draw", seat: "N", j: 5, ts: 11 }],
  };
  const keys = (rows: Decision[]) =>
    rows.map(({ item, choice }) => `${item.hand}:${item.idx}:${choice ?? ""}`);
  it("orders conflicts, missing evidence and certified margins without reindexing source choices", () => {
    const original = JSON.stringify(items);
    expect(keys(reviewDecisions(items, hands, facts))).toEqual([
      "1:2:",
      "1:0:",
      "1:5:",
      "0:1:1",
      "1:1:",
      "0:2:",
      "0:1:2",
      "0:1:3",
      "0:1:0",
      "0:0:",
    ]);
    expect(keys(reviewDecisions(items, hands, facts, 0))).toEqual([
      "0:1:1",
      "0:2:",
      "0:1:2",
      "0:1:3",
      "0:1:0",
      "0:0:",
    ]);
    expect(JSON.stringify(items)).toBe(original);
  });
  it("requires an individual fresh answer for every grouped decision", () => {
    const answers = {
      0: [
        { kind: "lost", field: "haipai", seat: "N", ts: 11 },
        { kind: "draw", seat: "N", j: 0, ts: 11 },
      ],
    };
    expect(isAnswered(items[1], hands, answers)).toBe(false);
    expect(keys(reviewDecisions(items, hands, answers, 0))).not.toContain(
      "0:1:0",
    );
    expect(keys(reviewDecisions(items, hands, answers, 0))).toContain("0:1:2");
    answers[0].push(
      { kind: "lost", seat: "N", j: 1, ts: 11 },
      { kind: "lost", seat: "N", j: 2, ts: 11 },
    );
    expect(isAnswered(items[1], hands, answers)).toBe(true);
  });
  it("keeps incomplete solver searches out of the question queue", () => {
    expect(
      reviewDecisions(
        [{ hand: 0, kind: "solver_incomplete", text: "retry" }],
        hands,
        {},
      ),
    ).toEqual([]);
  });
});
describe("input contracts", () => {
  it("preserves red fives and places the called tile according to the source", () => {
    expect(
      meldFact({
        hand: 0,
        seat: "N",
        t: 20,
        type: "pon",
        tiles: ["0m", "5m", "5m"],
        source: "shimocha",
      }),
    ).toMatchObject({ tiles: ["5m", "5m", "0m"], called_pos: 2 });
    expect(() =>
      meldFact({ type: "pon", tiles: ["?"], source: "kamicha" }),
    ).toThrow("3 tiles");
  });
  it("validates game IDs and time ranges before an upload", () => {
    expect(gameIds("12, https://scoremj.com/game?id=13")).toEqual([12, 13]);
    expect(() => gameIds("https://evil.com/12")).toThrow();
    expect(timeRange("1:20", "90")).toEqual({ start: "1:20", end: "90" });
    expect(() => timeRange("2:00", "90")).toThrow("after start");
    expect(() => timeRange("x", "")).toThrow("Use seconds");
  });
});
