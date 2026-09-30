import { defineComponent } from "vue";
import { mount, flushPromises } from "@vue/test-utils";
import { expect, it, vi } from "vitest";
import { api } from "../src/shared/api";
import type { Fact, ReviewItem } from "../src/types";
import { useReviewState } from "../src/review/useReviewState";
vi.mock("../src/shared/api", () => ({ api: vi.fn() }));
function setup() {
  let state: ReturnType<typeof useReviewState> | undefined;
  const wrapper = mount(
    defineComponent({
      setup() {
        state = useReviewState("/review/id");
        return () => null;
      },
    }),
  );
  if (!state) throw new Error("Review state was not initialized.");
  return { state, wrapper };
}
function mockServer() {
  let pending: number[] = [],
    running = false,
    fail = false;
  const saved: Fact[] = [];
  let selected: number[] = [];
  let items: ReviewItem[] = [];
  vi.mocked(api).mockImplementation(async (path, body) => {
    path = path.replace("/review/id/api/", "");
    if (path === "hands")
      return [
        { hand: 0, decoded_at: 0 },
        { hand: 1, decoded_at: 0 },
      ];
    if (path === "items") return items;
    if (path.startsWith("hand/")) return { decode: { confidence: [] } };
    if (path === "revision") return ["new"];
    if (path === "facts") {
      if (body) {
        const row = { ...(body as Fact), ts: 10 };
        saved.push(row);
        pending = [...new Set([...pending, row.hand])];
        return row;
      }
      return saved;
    }
    if (path === "decode_pending") {
      if (body) {
        running = true;
        selected = [...pending];
        return {
          running: true,
          pending,
          hands: selected,
          hands_total: selected.length,
        };
      }
      if (running) {
        running = false;
        if (!fail) pending = pending.filter((hand) => !selected.includes(hand));
      }
      return {
        running: false,
        pending,
        error: fail ? "Rebuild failed; Analyze recording" : null,
      };
    }
    throw Error(path);
  });
  return {
    addPending: (hand: number) => pending.push(hand),
    fail: (value = true) => (fail = value),
    items: (value: ReviewItem[]) => (items = value),
  };
}
it("rebuilds server pending deletions together with new answers and retains failed work", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const { state } = setup();
  await state.load();
  await state.save({ hand: 0, kind: "draw", tile: "2p" });
  server.addPending(1);
  await state.load();
  expect(state.pending.value).toEqual([0, 1]);
  await state.rebuild();
  expect(state.job.value.running).toBe(true);
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.pending.value).toEqual([]);
  await state.save({ hand: 0, kind: "discard", tile: "1m" });
  server.fail();
  await state.rebuild();
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.pending.value).toEqual([0]);
  expect(state.error.value).toContain("Analyze recording");
});
it("holds refreshes during playback or unsaved edits and releases timers on unmount", async () => {
  vi.useFakeTimers();
  mockServer();
  const { state, wrapper } = setup();
  await state.load();
  state.revision.value = '["old"]';
  state.players.value = 1;
  await state.poll();
  expect(state.updates.value).toBe(true);
  expect(state.revision.value).toBe('["old"]');
  state.players.value = 0;
  state.dirty.value = true;
  await state.poll();
  expect(state.revision.value).toBe('["old"]');
  const calls = vi.mocked(api).mock.calls.length;
  await state.rebuild();
  expect(api).toHaveBeenCalledTimes(calls);
  state.dirty.value = false;
  await state.poll();
  expect(state.revision.value).toBe('["new"]');
  await state.rebuild();
  wrapper.unmount();
  const count = vi.mocked(api).mock.calls.length;
  await vi.advanceTimersByTimeAsync(10000);
  expect(api).toHaveBeenCalledTimes(count);
});
it("reconnects to a running job without posting another one", async () => {
  vi.useFakeTimers();
  mockServer();
  const { state } = setup();
  await state.rebuild("decode_pending", true);
  await flushPromises();
  expect(
    vi
      .mocked(api)
      .mock.calls.filter(
        ([path, body]) => path.endsWith("decode_pending") && body,
      ),
  ).toEqual([]);
});
it("keeps checking a running job discovered by a refresh until it finishes", async () => {
  vi.useFakeTimers();
  let running = true;
  vi.mocked(api).mockImplementation(async (path) => {
    if (path.endsWith("decode_pending"))
      return { running, pending: running ? [0] : [] };
    if (path.endsWith("revision")) return [1];
    return [];
  });
  const { state } = setup();
  await state.refresh();
  expect(state.updating.value).toBe(true);
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.updating.value).toBe(true);
  running = false;
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.updating.value).toBe(false);
  expect(
    vi.mocked(api).mock.calls.every(([, body]) => body === undefined),
  ).toBe(true);
});
it("lets another hand be answered while updating and batches its queued answer without repeating related questions", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  server.items([
    { hand: 0, idx: 0, kind: "draw", seat: "N", j: 0, margin: 1 },
    {
      hand: 1,
      idx: 0,
      kind: "uncertain_tiles",
      choices: [
        { field: "draw", seat: "E", j: 0, margin: 0 },
        { field: "draw", seat: "E", j: 1, margin: 0.1 },
      ],
    },
    { hand: 0, idx: 1, kind: "draw", seat: "N", j: 1, margin: 0.5 },
  ]);
  const { state } = setup();
  state.guided.value = true;
  await state.refresh();
  expect(state.selection.value?.item.hand).toBe(1);
  expect(state.selection.value?.decision.j).toBe(0);
  await state.save({ hand: 1, kind: "draw", seat: "E", j: 0, tile: "2p" });
  await flushPromises();
  expect(state.updating.value).toBe(true);
  expect(state.selection.value?.item.hand).toBe(0);
  expect(state.selection.value?.decision.j).toBe(1);
  // A draft and its video must survive the unrelated hand's background update.
  const current = state.selection.value;
  state.dirty.value = true;
  state.players.value = 1;
  const count = vi.mocked(api).mock.calls.length;
  await state.poll();
  expect(api).toHaveBeenCalledTimes(count);
  expect(state.selection.value).toBe(current);
  // Reconstruction settles both draws in hand 1 from the one supplied answer.
  server.items([
    { hand: 0, idx: 0, kind: "draw", seat: "N", j: 0, margin: 1 },
    { hand: 0, idx: 1, kind: "draw", seat: "N", j: 1, margin: 0.5 },
  ]);
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.updating.value).toBe(false);
  expect(state.pending.value).toEqual([]);
  expect(state.decisions.value).toHaveLength(2);
  expect(state.selection.value?.decision.j).toBe(1);
  expect(state.selection.value).toBe(current);
  expect(state.dirty.value).toBe(true);
  expect(
    vi
      .mocked(api)
      .mock.calls.filter(
        ([path, body]) => path.endsWith("decode_pending") && body,
      ),
  ).toHaveLength(1);
});
it("queues answers from different hands in one follow-up batch while a job runs", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const { state } = setup();
  state.guided.value = true;
  server.items(
    [0, 1, 2].map((hand) => ({
      hand,
      idx: 0,
      kind: "draw",
      seat: "E",
      j: 0,
      margin: hand,
    })),
  );
  await state.refresh();
  await state.save({ hand: 0, kind: "draw", seat: "E", j: 0, tile: "2p" });
  await flushPromises();
  await state.save({ hand: 1, kind: "draw", seat: "E", j: 0, tile: "3p" });
  await state.save({ hand: 2, kind: "draw", seat: "E", j: 0, tile: "4p" });
  await flushPromises();
  const posts = () =>
    vi
      .mocked(api)
      .mock.calls.filter(
        ([path, body]) => path.endsWith("decode_pending") && body,
      );
  expect(posts()).toHaveLength(1);
  expect(state.selection.value).toBeNull();
  await vi.advanceTimersByTimeAsync(2000);
  expect(posts()).toHaveLength(2);
  expect(state.job.value.hands).toEqual([1, 2]);
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.pending.value).toEqual([]);
  expect(posts()).toHaveLength(2);
});
it("keeps skipped questions unresolved across refreshes and returns to them explicitly", async () => {
  const server = mockServer();
  const item = { hand: 1, idx: 0, kind: "draw", seat: "E", j: 0, margin: 0 };
  server.items([item]);
  const { state } = setup();
  await state.refresh();
  state.next();
  expect(state.selection.value).toBeNull();
  expect(state.decisions.value).toHaveLength(1);
  server.items([{ ...item, idx: 9 }]);
  await state.refresh();
  expect(state.questions.value).toHaveLength(0);
  expect(state.selection.value).toBeNull();
  state.revisit();
  expect(state.selection.value?.decision.j).toBe(0);
});
it("offers timed-out confidence checks and incomplete processing directly in main review", async () => {
  const server = mockServer();
  server.items([
    {
      hand: 0,
      idx: 0,
      kind: "uncertain_tiles",
      choices: [
        {
          field: "draw",
          seat: "E",
          j: 0,
          margin: 0,
          alternative_gap: 80,
          runner_up: "2p",
        },
        {
          field: "draw",
          seat: "E",
          j: 1,
          margin: 0,
          alternative_gap: 0,
          runner_up: null,
        },
      ],
    },
    { hand: 1, idx: 0, kind: "solver_incomplete" },
  ]);
  const { state, wrapper } = setup();
  await state.refresh();
  expect(state.questions.value).toHaveLength(2);
  expect(state.selection.value?.decision.j).toBe(0);
  state.next();
  expect(state.selection.value?.decision.j).toBe(1);
  state.next();
  expect(
    state.questions.value.some(
      (row) => row.decision.kind === "solver_incomplete",
    ),
  ).toBe(false);
  expect(api).not.toHaveBeenCalledWith(expect.stringContaining("/hand/"));
  wrapper.unmount();
});
it("retains saved answers on an automatic update failure and clears the error on retry", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const { state } = setup();
  state.guided.value = true;
  server.fail();
  await state.save({ hand: 0, kind: "draw", seat: "E", j: 0, tile: "2p" });
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.updating.value).toBe(false);
  expect(state.pending.value).toEqual([0]);
  expect(state.error.value).toContain("Rebuild failed");
  expect(state.facts.value[0]).toHaveLength(1);
  server.fail(false);
  await state.rebuild();
  await vi.advanceTimersByTimeAsync(2000);
  expect(state.error.value).toBe("");
  expect(state.pending.value).toEqual([]);
});
