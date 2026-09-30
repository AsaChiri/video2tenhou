import { mount, flushPromises } from "@vue/test-utils";
import type { Component } from "vue";
import { ref } from "vue";
import { describe, expect, it, vi } from "vitest";
import { reviewKey } from "../src/review/context";
import AnswerEditor from "../src/review/AnswerEditor.vue";
import TilePalette from "../src/review/TilePalette.vue";
import EvidenceDetails from "../src/review/EvidenceDetails.vue";
import FactsPanel from "../src/review/FactsPanel.vue";
import QuestionPanel from "../src/review/QuestionPanel.vue";
const entry = { hand: 0, t_start: 20, t_end: 100, corner_wind: { TL: "N" } };
function state<T extends object>(extra: T = {} as T) {
  return {
    save: vi.fn().mockResolvedValue({}),
    next: vi.fn(),
    dirty: ref(false),
    players: ref(0),
    error: ref(""),
    facts: ref({}),
    pending: ref([]),
    job: ref({}),
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
describe("review components", () => {
  it("requires explicit confirmation of a final concealed hand", async () => {
    const review = state();
    const wrapper = render(
      AnswerEditor,
      {
        item: {
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
    const save = wrapper
      .findAll("button")
      .find((button) => button.text() === "Save final hand");
    if (!save) throw new Error("Save button is missing.");
    await save.trigger("click");
    expect(review.save).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("Confirm that you compared");
    await wrapper.get("input[type=checkbox]").setValue(true);
    if (!save) throw new Error("Save button is missing.");
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
    const lost = wrapper
      .findAll("button")
      .find((button) => button.text() === "Can't tell from the video");
    if (!lost) throw new Error("Missing unavailable-answer button.");
    await lost.trigger("click");
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
        { item: { hand: 0, kind, seat: "N", j: 2, t: 30, tile: "1m" }, entry },
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
  it("provides correction controls for an export tile-count conflict", () => {
    const wrapper = render(
      AnswerEditor,
      {
        item: {
          hand: 0,
          kind: "conflict",
          stage: "export",
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
    expect(wrapper.text()).toContain("1 copies of 0s");
    expect(wrapper.text()).toContain("Save starting hand");
    expect(wrapper.findComponent(TilePalette).exists()).toBe(true);
    expect(wrapper.text()).not.toContain("Acknowledge");
  });
  it("shows all three source ponds when a call source is unknown", () => {
    const wrapper = render(
      AnswerEditor,
      {
        item: { hand: 0, kind: "call", seat: "N", t: 30 },
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
  it("escapes untrusted player names, facts and notes through Vue interpolation", () => {
    const payload = '<img src=x onerror="alert(1)">';
    const review = state({
      facts: ref({ 0: [{ kind: "note", seat: "N", text: payload }] }),
    });
    const wrapper = render(
      FactsPanel,
      {
        hand: 0,
        entry: { ...entry, nicks: { TL: payload } },
        notes: [payload],
      },
      review,
    );
    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.text()).toContain(payload);
  });
  it("ignores a stale hand response after choosing another question", async () => {
    const pending: ((value: unknown) => void)[] = [];
    const review = state({
      selection: ref({
        item: { hand: 0, idx: 0, kind: "draw", seat: "N", j: 0, t: 30 },
        choice: null,
      }),
      api: vi.fn(() => new Promise((resolve) => pending.push(resolve))),
    });
    const wrapper = render(QuestionPanel, {}, review);
    review.selection.value = {
      item: { hand: 0, idx: 1, kind: "draw", seat: "N", j: 1, t: 60 },
      choice: null,
    };
    await flushPromises();
    pending[1]({ entry, decode: { turns: [] } });
    await flushPromises();
    expect(wrapper.text()).toContain("Draw 2");
    pending[0]({ entry, decode: { turns: [] } });
    await flushPromises();
    expect(wrapper.text()).toContain("Draw 2");
    expect(wrapper.findAll("video")).toHaveLength(1);
  });
});
