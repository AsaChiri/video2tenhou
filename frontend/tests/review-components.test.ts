import { mount, flushPromises } from "@vue/test-utils";
import type { Component } from "vue";
import { ref } from "vue";
import { describe, expect, it, vi } from "vitest";
import { reviewKey } from "../src/review/context";
import AnswerEditor from "../src/review/AnswerEditor.vue";
import ConflictEditor from "../src/review/ConflictEditor.vue";
import TilePalette from "../src/review/TilePalette.vue";
import EvidenceDetails from "../src/review/EvidenceDetails.vue";
import FactsPanel from "../src/review/FactsPanel.vue";
import HandsPanel from "../src/review/HandsPanel.vue";
import QuestionPanel from "../src/review/QuestionPanel.vue";
const entry = {
  hand: 0,
  game: 0,
  kyoku: 0,
  honba: 0,
  t_start: 20,
  t_end: 100,
  corner_wind: { TL: "N" },
};
function state<T extends object>(extra: T = {} as T) {
  return {
    save: vi.fn().mockResolvedValue({}),
    dismiss: vi.fn().mockResolvedValue(undefined),
    remove: vi.fn().mockResolvedValue(undefined),
    next: vi.fn(),
    dirty: ref(false),
    players: ref(0),
    error: ref(""),
    facts: ref({}),
    items: ref([]),
    pending: ref([]),
    busy: ref(false),
    loads: ref(0),
    url: (path: string) => `/review/id/api/${path}`,
    ...extra,
  };
}
function render(
  component: Component,
  props: Record<string, unknown>,
  review: object,
) {
  return mount(component, {
    props,
    global: { provide: { [reviewKey]: review } },
  });
}
function button(wrapper: ReturnType<typeof render>, text: string) {
  const found = wrapper.findAll("button").find((b) => b.text() === text);
  if (!found) throw new Error(`Missing button: ${text}`);
  return found;
}
describe("review components", () => {
  it("requires explicit confirmation of a final concealed hand", async () => {
    const review = state();
    const wrapper = render(
      AnswerEditor,
      {
        item: {
          id: "result:N:60",
          hand: 0,
          kind: "result",
          seat: "N",
          t: 60,
          tiles: Array(13).fill("1m"),
        },
        entry,
      },
      review,
    );
    const save = button(wrapper, "Save final hand");
    await save.trigger("click");
    expect(review.save).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Confirm that you compared");
    await wrapper.get("input[type=checkbox]").setValue(true);
    await save.trigger("click");
    await flushPromises();
    expect(review.save).toHaveBeenCalledWith({
      hand: 0,
      kind: "final_hand",
      seat: "N",
      t: 60,
      tiles: Array(13).fill("1m"),
    });
  });
  it("saves an unavailable starting hand without answering a draw", async () => {
    const review = state();
    const wrapper = render(
      AnswerEditor,
      {
        item: {
          id: "haipai:N:-1",
          hand: 0,
          kind: "haipai",
          seat: "N",
          t: 20,
          tiles: Array(13).fill("1m"),
        },
        entry,
      },
      review,
    );
    await button(wrapper, "Can't tell from the video").trigger("click");
    await flushPromises();
    expect(review.save).toHaveBeenCalledWith({
      hand: 0,
      kind: "lost",
      field: "haipai",
      seat: "N",
      t: 20,
    });
  });
  it.each(["draw", "discard"])(
    "saves a %s directly from the only answer palette",
    async (kind) => {
      const review = state();
      const wrapper = render(
        AnswerEditor,
        {
          item: {
            id: `${kind}:N:2`,
            hand: 0,
            kind,
            seat: "N",
            j: 2,
            t: 30,
            tile: "1m",
          },
          entry,
        },
        review,
      );
      expect(wrapper.findAllComponents(TilePalette)).toHaveLength(1);
      await wrapper.get('[aria-label="Choose 0p"]').trigger("click");
      await flushPromises();
      expect(review.save).toHaveBeenCalledWith({
        hand: 0,
        kind,
        seat: "N",
        t: 30,
        tile: "0p",
        ...(kind === "draw" ? { j: 2, t_discard: 30 } : {}),
      });
      expect(review.next).not.toHaveBeenCalled();
    },
  );
  it("dismisses a question that needs no tile answer instead of saving a note", async () => {
    const review = state();
    const item = {
      id: "conflict::90",
      hand: 0,
      kind: "conflict",
      t: 90,
      culprit: null,
      text: "No reconstruction of this hand is legal.",
    };
    const wrapper = render(AnswerEditor, { item, entry }, review);
    expect(wrapper.text()).toContain(
      "No reconstruction of this hand is legal.",
    );
    await button(wrapper, "Leave as conflict").trigger("click");
    await flushPromises();
    expect(review.dismiss).toHaveBeenCalledWith(item);
    expect(review.save).not.toHaveBeenCalled();
  });
  it("confirms a shown call as a meld fact", async () => {
    const review = state();
    const wrapper = render(
      AnswerEditor,
      {
        item: {
          id: "call:N:40",
          hand: 0,
          kind: "call",
          seat: "N",
          t: 40,
          type: "pon",
          tiles: ["5m", "0m", "5m"],
          source: "toimen",
          text: "What was the meld?",
        },
        entry,
      },
      review,
    );
    await button(wrapper, "It is right").trigger("click");
    await flushPromises();
    expect(review.save).toHaveBeenCalledWith({
      kind: "meld",
      hand: 0,
      seat: "N",
      t: 40,
      type: "pon",
      tiles: ["5m", "0m", "5m"],
      source: "toimen",
      called_pos: 1,
    });
    expect(review.dismiss).not.toHaveBeenCalled();
  });
  it("provides correction controls for an export tile-count conflict", () => {
    const wrapper = render(
      AnswerEditor,
      {
        item: {
          id: "conflict::",
          hand: 0,
          kind: "conflict",
          stage: "export",
          text: "This hand cannot be exported.",
          violations: [
            {
              kind: "out_of_turn",
              seat: "N",
              tile: null,
              t: 75,
              text: "calls where a draw is due (out of turn)",
            },
            {
              kind: "indicators",
              seat: null,
              tile: null,
              t: null,
              text: "2 dora indicator(s) for 0 kan(s): a log needs 1",
            },
          ],
          over: [
            {
              tile: "0s",
              count: 2,
              limit: 1,
              sources: [
                {
                  kind: "haipai",
                  seat: "N",
                  corner: "TL",
                  t: 20,
                  tile: "0s",
                  tiles: ["0s"],
                },
                { kind: "indicator", field: "dora", index: 0, tile: "0s" },
              ],
            },
          ],
        },
        entry,
        decode: { dora: ["0s"] },
      },
      state(),
    );
    expect(wrapper.text()).toContain("This hand cannot be exported.");
    expect(wrapper.text()).toContain(
      "North (TL) calls where a draw is due (out of turn) (1:15)",
    );
    expect(wrapper.text()).toContain(
      "2 dora indicator(s) for 0 kan(s): a log needs 1",
    );
    expect(wrapper.text()).toContain("has 1 of 0s");
    expect(wrapper.text()).toContain("Save starting hand");
    expect(wrapper.findComponent(TilePalette).exists()).toBe(true);
    expect(wrapper.text()).not.toContain("Leave as conflict");
  });
  it("shows the conflict's own prompt first and no hidden diagnosis", () => {
    const wrapper = render(
      ConflictEditor,
      {
        item: {
          id: "conflict:N:50",
          hand: 0,
          kind: "conflict",
          culprit: "fact",
          fact_seat: "N",
          fact_kind: "final_hand",
          text: "N's saved final hand conflicts with its calls. Check that fact.",
        },
        entry,
        decode: {},
      },
      state(),
    );
    expect(wrapper.find("p").text()).toBe(
      "N's saved final hand conflicts with its calls. Check that fact.",
    );
    expect(wrapper.text()).toContain("Saved final hand of North");
    expect(wrapper.text()).not.toContain("diagnosis");
  });
  it("shows all three source ponds when a call source is unknown", () => {
    const wrapper = render(
      AnswerEditor,
      {
        item: { id: "call:N:30", hand: 0, kind: "call", seat: "N", t: 30 },
        entry: {
          ...entry,
          corner_wind: { TL: "E", TR: "S", BL: "W", BR: "N" },
        },
      },
      state(),
    );
    expect(
      wrapper
        .findAllComponents(EvidenceDetails)
        .map((component) => component.props("region")),
    ).toEqual(["pond:TL", "pond:TR", "pond:BL"]);
  });
  it("does not fetch collapsed evidence", async () => {
    const wrapper = render(
      EvidenceDetails,
      { region: "hand:TL", start: 10, end: 20, label: "Camera" },
      state(),
    );
    expect(wrapper.find("video").exists()).toBe(false);
    wrapper.element.open = true;
    await wrapper.trigger("toggle");
    expect(wrapper.get("video").attributes("src")).toContain(
      "/review/id/api/clip?",
    );
  });
  it("marks an answer the reconstruction ignored on its own row", () => {
    const fact = { kind: "draw", hand: 0, seat: "N", t: 30, tile: "2p", ts: 7 };
    const wrapper = render(
      FactsPanel,
      {
        hand: 0,
        entry,
        ignored: [
          { ts: 7, kind: "draw", reason: "no turn of N at 0:30" },
          { ts: null, kind: "meld", reason: "no meld near 1:10" },
        ],
      },
      state({ facts: ref({ 0: [fact] }) }),
    );
    const rows = wrapper.findAll("li");
    expect(rows[0].text()).toContain("Not applied: no turn of N at 0:30");
    expect(rows[0].text()).toContain("Delete");
    expect(rows[1].text()).toContain("Not applied: no meld near 1:10");
  });
  it("escapes untrusted player names, facts and notes through Vue interpolation", async () => {
    const payload = '<img src=x onerror="alert(1)">';
    const review = state({
      facts: ref({ 0: [{ kind: "note", hand: 0, seat: "N", text: payload }] }),
      hands: ref([]),
      api: vi.fn().mockResolvedValue({
        entry: { ...entry, nicks: { TL: payload } },
        decode: { notes: [payload], turns: [], confidence: [] },
        ignored: [{ ts: null, kind: "draw", reason: payload }],
      }),
    });
    const wrapper = render(HandsPanel, { hand: 0 }, review);
    await flushPromises();
    expect(wrapper.find("img[src=x]").exists()).toBe(false);
    expect(wrapper.text()).toContain(payload);
    expect(wrapper.get(".notes").text()).toBe(payload);
  });
  it("ignores a stale hand response after choosing another question", async () => {
    const pending: ((value: unknown) => void)[] = [];
    const first = { id: "draw:N:0", hand: 0, kind: "draw", seat: "N", j: 0 };
    const second = { id: "draw:N:1", hand: 0, kind: "draw", seat: "N", j: 1 };
    const review = state({
      selection: ref({ key: "0:draw:N:0", item: first, choice: null }),
      api: vi.fn(() => new Promise((resolve) => pending.push(resolve))),
    });
    const wrapper = render(QuestionPanel, {}, review);
    review.selection.value = { key: "0:draw:N:1", item: second, choice: null };
    await flushPromises();
    pending[1]({ entry, decode: { turns: [] }, ignored: [] });
    await flushPromises();
    expect(wrapper.text()).toContain("Draw 2");
    pending[0]({ entry, decode: { turns: [] }, ignored: [] });
    await flushPromises();
    expect(wrapper.text()).toContain("Draw 2");
    expect(wrapper.findAll("video")).toHaveLength(1);
  });
});
