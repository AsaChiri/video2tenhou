import { defineComponent, ref } from "vue";
import { mount, flushPromises } from "@vue/test-utils";
import { expect, it, vi } from "vitest";
import { api } from "../src/shared/api";
import type { Fact, ReviewItem, WorkspaceJob } from "../src/types";
import { useJobStatus } from "../src/shared/useJob";
import { useReviewState } from "../src/review/useReviewState";
vi.mock("../src/shared/api", () => ({ api: vi.fn() }));
function setup() {
  let state: ReturnType<typeof useReviewState> | undefined;
  const wrapper = mount(
    defineComponent({
      setup() {
        state = useReviewState("id", useJobStatus(ref("id")));
        return () => null;
      },
    }),
  );
  if (!state) throw new Error("Review state was not initialized.");
  return { state, wrapper };
}
/**
 * A server whose rebuild jobs finish at the next status poll. Answers make
 * their hand pending until a successful update; dismissals never do.
 */
function mockServer() {
  const pending = new Set<number>(),
    saved: Fact[] = [];
  let items: ReviewItem[] = [],
    job: WorkspaceJob | null = null,
    shown = false,
    fail = false,
    revision = 0,
    clock = 0;
  const status = () => ({ job, revision: String(revision) });
  vi.mocked(api).mockImplementation(async (path, body) => {
    if (path.startsWith("/api/job")) {
      if (job?.running && !shown) shown = true;
      else if (job?.running) {
        job = { ...job, running: false, finished: ++clock };
        if (fail) job.error = "Hand 1 has no tile readings. Analyze recording.";
        else for (const hand of job.hands || []) pending.delete(hand);
        revision++;
      }
      return status();
    }
    path = path.replace("/review/id/api/", "");
    if (path === "hands")
      return [0, 1, 2].map((hand) => ({ hand, pending: pending.has(hand) }));
    if (path === "items") return items;
    if (path === "facts") {
      if (!body) return saved;
      const row = { ...(body as Fact), ts: ++clock };
      saved.push(row);
      if (row.kind !== "dismiss") pending.add(row.hand);
      revision++;
      return row;
    }
    if (path === "rebuild") {
      shown = true;
      if (pending.size)
        job = {
          kind: "rebuild",
          project: "id",
          stage: "Updating hands",
          hands: [...pending].sort(),
          running: true,
          started: ++clock,
          finished: null,
          error: null,
          log_lines: 0,
        };
      return status();
    }
    throw Error(path);
  });
  const posts = () =>
    vi.mocked(api).mock.calls.filter(([path]) => path.endsWith("/rebuild"));
  return {
    addPending: (hand: number) => pending.add(hand),
    fail: (value = true) => (fail = value),
    items: (value: ReviewItem[]) => (items = value),
    touch: () => revision++,
    running: (hands: number[]) =>
      (job = {
        kind: "rebuild",
        project: "id",
        stage: "Updating hands",
        hands,
        running: true,
        started: ++clock,
        finished: null,
        error: null,
        log_lines: 0,
      }),
    posts,
  };
}
const draw = (hand: number, j: number, margin = 0): ReviewItem => ({
  id: `draw:E:${j}`,
  hand,
  kind: "draw",
  seat: "E",
  j,
  margin,
});
it("updates server pending deletions together with new answers and keeps failed work", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const { state } = setup();
  await state.refresh();
  await state.save({ hand: 0, kind: "draw", tile: "2p" });
  server.addPending(1);
  await state.refresh();
  expect(state.pending.value).toEqual([0, 1]);
  await state.rebuild();
  expect(state.updating.value).toBe(true);
  expect(state.job.value?.hands).toEqual([0, 1]);
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.updating.value).toBe(false);
  expect(state.pending.value).toEqual([]);
  await state.save({ hand: 0, kind: "discard", tile: "1m" });
  server.fail();
  await state.rebuild();
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.pending.value).toEqual([0]);
  expect(state.error.value).toContain("Analyze recording");
});
it("holds refreshes during playback or unsaved edits and releases timers on unmount", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const { state, wrapper } = setup();
  await vi.advanceTimersByTimeAsync(0);
  await state.refresh();
  const loads = state.loads.value;
  state.players.value = 1;
  server.touch();
  await vi.advanceTimersByTimeAsync(5000);
  expect(state.updates.value).toBe(true);
  expect(state.loads.value).toBe(loads);
  state.players.value = 0;
  state.updates.value = false;
  state.dirty.value = true;
  server.touch();
  await vi.advanceTimersByTimeAsync(5000);
  expect(state.loads.value).toBe(loads);
  await state.rebuild("all");
  expect(state.message.value).toContain("Save the current answer");
  expect(server.posts()).toHaveLength(0);
  state.dirty.value = false;
  server.touch();
  await vi.advanceTimersByTimeAsync(5000);
  expect(state.loads.value).toBe(loads + 1);
  wrapper.unmount();
  const count = vi.mocked(api).mock.calls.length;
  await vi.advanceTimersByTimeAsync(10000);
  expect(api).toHaveBeenCalledTimes(count);
});
it("follows a running update without starting another one", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  server.running([1]);
  const { state } = setup();
  state.guided.value = true;
  await vi.advanceTimersByTimeAsync(0);
  await state.refresh();
  expect(state.updating.value).toBe(true);
  expect(state.pending.value).toEqual([1]);
  state.maybeStart();
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.updating.value).toBe(false);
  expect(server.posts()).toEqual([]);
});
it("lets another hand be answered while updating and keeps an unrelated draft", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  server.items([
    draw(0, 0, 1),
    {
      id: "uncertain_tiles::",
      hand: 1,
      kind: "uncertain_tiles",
      choices: [
        { id: "draw:E:0", field: "draw", seat: "E", j: 0, margin: 0 },
        { id: "draw:E:1", field: "draw", seat: "E", j: 1, margin: 0.1 },
      ],
    },
    draw(0, 1, 0.5),
  ]);
  const { state } = setup();
  state.guided.value = true;
  await vi.advanceTimersByTimeAsync(0);
  await state.refresh();
  expect(state.selection.value?.key).toBe("1:draw:E:0");
  await state.save({ hand: 1, kind: "draw", seat: "E", j: 0, tile: "2p" });
  await flushPromises();
  expect(state.updating.value).toBe(true);
  // Both questions of the updating hand wait; another hand is offered.
  expect(state.selection.value?.key).toBe("0:draw:E:1");
  const current = state.selection.value;
  state.dirty.value = true;
  state.players.value = 1;
  // The update settles both draws of hand 1 from the one answer.
  server.items([draw(0, 0, 1), draw(0, 1, 0.5)]);
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.updating.value).toBe(false);
  expect(state.pending.value).toEqual([]);
  expect(state.decisions.value).toHaveLength(2);
  expect(state.selection.value).toBe(current);
  expect(state.dirty.value).toBe(true);
  expect(server.posts()).toHaveLength(1);
});
it("batches answers from different hands saved while an update runs", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  server.items([0, 1, 2].map((hand) => draw(hand, 0, hand)));
  const { state } = setup();
  state.guided.value = true;
  await vi.advanceTimersByTimeAsync(0);
  await state.refresh();
  await state.save({ hand: 0, kind: "draw", seat: "E", j: 0, tile: "2p" });
  await flushPromises();
  await state.save({ hand: 1, kind: "draw", seat: "E", j: 0, tile: "3p" });
  await state.save({ hand: 2, kind: "draw", seat: "E", j: 0, tile: "4p" });
  await flushPromises();
  expect(server.posts()).toHaveLength(1);
  expect(state.selection.value).toBeNull();
  await vi.advanceTimersByTimeAsync(1000);
  expect(server.posts()).toHaveLength(2);
  expect(state.job.value?.hands).toEqual([1, 2]);
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.pending.value).toEqual([]);
  expect(server.posts()).toHaveLength(2);
});
it("dismisses a question without updating its hand", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const conflict: ReviewItem = {
    id: "conflict::120",
    hand: 2,
    kind: "conflict",
    t: 120,
  };
  server.items([conflict, draw(0, 0, 1)]);
  const { state } = setup();
  state.guided.value = true;
  await vi.advanceTimersByTimeAsync(0);
  await state.refresh();
  expect(state.selection.value?.key).toBe("2:conflict::120");
  await state.dismiss(conflict);
  await flushPromises();
  expect(api).toHaveBeenCalledWith("/review/id/api/facts", {
    kind: "dismiss",
    hand: 2,
    item: "conflict::120",
  });
  expect(state.pending.value).toEqual([]);
  expect(state.selection.value?.key).toBe("0:draw:E:0");
  expect(server.posts()).toHaveLength(0);
});
it("keeps skipped questions unresolved across refreshes and returns to them explicitly", async () => {
  const server = mockServer();
  server.items([draw(1, 0)]);
  const { state } = setup();
  await state.refresh();
  state.next();
  expect(state.selection.value).toBeNull();
  expect(state.decisions.value).toHaveLength(1);
  await state.refresh();
  expect(state.questions.value).toHaveLength(0);
  expect(state.selection.value).toBeNull();
  state.revisit();
  expect(state.selection.value?.choice).toBeNull();
  expect(state.selection.value?.item.j).toBe(0);
});
it("retains saved answers on an automatic update failure and clears the error on retry", async () => {
  vi.useFakeTimers();
  const server = mockServer();
  const { state } = setup();
  state.guided.value = true;
  await vi.advanceTimersByTimeAsync(0);
  server.fail();
  await state.save({ hand: 0, kind: "draw", seat: "E", j: 0, tile: "2p" });
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.updating.value).toBe(false);
  expect(state.pending.value).toEqual([0]);
  expect(state.error.value).toContain("no tile readings");
  expect(state.facts.value[0]).toHaveLength(1);
  server.fail(false);
  await state.rebuild();
  expect(state.error.value).toBe("");
  await vi.advanceTimersByTimeAsync(1000);
  expect(state.pending.value).toEqual([]);
});
