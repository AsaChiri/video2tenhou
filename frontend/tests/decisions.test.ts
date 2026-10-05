import type { Decision, ReviewItem } from "../src/types";
import { describe, expect, it } from "vitest";
import { reviewDecisions } from "../src/review/decisions";
import { parseTime, time } from "../src/review/format";
import { confirmedMeld, meldFact } from "../src/review/meld";
import { gameIds, timeRange } from "../src/studio/validation";

describe("review decisions", () => {
  it("keeps every unresolved choice in review regardless of the candidate gap", () => {
    const item: ReviewItem = {
      id: "uncertain_tiles::",
      hand: 0,
      kind: "uncertain_tiles",
      choices: [
        {
          id: "draw:E:0",
          field: "draw",
          seat: "E",
          j: 0,
          margin: 0,
          runner_up: "2p",
          alternative_gap: 80,
        },
        {
          id: "draw:E:1",
          field: "draw",
          seat: "E",
          j: 1,
          margin: 0,
          runner_up: "2p",
          alternative_gap: 0.2,
        },
        {
          id: "draw:E:2",
          field: "draw",
          seat: "E",
          j: 2,
          margin: 0,
          runner_up: null,
          alternative_gap: 0,
        },
        { id: "haipai:E:-1", field: "haipai", seat: "E", j: -1, margin: 0 },
      ],
    };
    const decisions = reviewDecisions([item]);
    expect(decisions).toHaveLength(4);
    expect(decisions[0].choice?.field).toBe("haipai");
    expect(decisions.map((row) => row.choice?.j)).toEqual([-1, 0, 1, 2]);
    expect(decisions.map((row) => row.key)).toEqual([
      "0:haipai:E:-1",
      "0:draw:E:0",
      "0:draw:E:1",
      "0:draw:E:2",
    ]);
  });
  const items: ReviewItem[] = [
    { hand: 0, id: "ura::", kind: "ura", margin: null },
    {
      hand: 0,
      id: "uncertain_tiles::",
      kind: "uncertain_tiles",
      choices: [
        { id: "a", field: "haipai", seat: "N", j: -1, margin: 0.4, value: [] },
        {
          id: "b",
          field: "draw",
          seat: "N",
          j: 0,
          margin: 0,
          alternative_gap: 90,
          value: "2p",
        },
        {
          id: "c",
          field: "draw",
          seat: "N",
          j: 1,
          margin: 0.2,
          alternative_gap: 0,
          value: "1z",
        },
        { id: "d", field: "draw", seat: "N", j: 2, margin: 0.2, value: "9m" },
      ],
    },
    {
      hand: 0,
      id: "discard:N:9",
      kind: "discard",
      seat: "N",
      j: 9,
      margin: 0.1,
    },
    {
      hand: 1,
      id: "draw:N:3",
      kind: "draw",
      seat: "N",
      j: 3,
      margin: 0.5,
      lost: true,
    },
    { hand: 1, id: "discard:N:4", kind: "discard", seat: "N", j: 4, margin: 0 },
    { hand: 1, id: "conflict::", kind: "conflict" },
    { hand: 1, id: "draw:N:5", kind: "draw", seat: "N", j: 5, margin: 0.15 },
    { hand: 1, id: "draw:N:6", kind: "draw", seat: "N", j: 6, lost: true },
  ];
  const keys = (rows: Decision[]) => rows.map((row) => row.key);
  it("orders conflicts, missing evidence and certified margins without changing source items", () => {
    const original = JSON.stringify(items);
    expect(keys(reviewDecisions(items))).toEqual([
      "1:conflict::",
      "1:draw:N:3",
      "1:draw:N:6",
      "0:b",
      "1:discard:N:4",
      "0:discard:N:9",
      "1:draw:N:5",
      "0:c",
      "0:d",
      "0:a",
      "0:ura::",
    ]);
    expect(JSON.stringify(items)).toBe(original);
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
  it("confirms a shown meld exactly as laid out", () => {
    expect(
      confirmedMeld({
        seat: "N",
        t: 20,
        type: "pon",
        tiles: ["5m", "0m", "5m"],
        source: "toimen",
      }),
    ).toEqual({
      kind: "meld",
      hand: undefined,
      seat: "N",
      t: 20,
      type: "pon",
      tiles: ["5m", "0m", "5m"],
      source: "toimen",
      called_pos: 1,
    });
  });
  it("validates game IDs and time ranges before an upload", () => {
    expect(gameIds("12, https://scoremj.com/game?id=13")).toEqual([12, 13]);
    expect(() => gameIds("https://evil.com/12")).toThrow();
    expect(timeRange("1:20", "90")).toEqual({ start: "1:20", end: "90" });
    expect(() => timeRange("2:00", "90")).toThrow("after start");
    expect(() => timeRange("x", "")).toThrow("Use seconds");
  });
  it("shows and reads video times as m:ss", () => {
    expect(time(75.4)).toBe("1:15");
    expect(time(3725)).toBe("62:05");
    expect(parseTime("1:15")).toBe(75);
    expect(parseTime("90")).toBe(90);
    expect(parseTime("1:75")).toBeNull();
  });
});
