import { mount, flushPromises } from "@vue/test-utils";
import type { VueWrapper } from "@vue/test-utils";
import type { Calibration } from "../src/types";
import { ref } from "vue";
import { expect, it, vi } from "vitest";
import CalibrationEditor from "../src/review/CalibrationEditor.vue";
import { reviewKey } from "../src/review/context";
import {
  calibrationBody,
  dragRegion,
  hitRegion,
} from "../src/review/calibration";
const saved: Calibration = {
  frame: [1000, 1000],
  regions: {
    "hand:TL": {
      kind: "hand",
      movable: true,
      rect: [100, 100, 100, 100],
      quad: [
        [100, 100],
        [200, 100],
        [200, 200],
        [100, 200],
      ],
      roll: 3,
    },
    "meld:TL": {
      kind: "meld",
      movable: true,
      rect: [300, 300, 100, 100],
      quad: [
        [300, 300],
        [400, 300],
        [400, 400],
        [300, 400],
      ],
    },
  },
  checks: { "hand:TL": { level: "ok", held: 0, cut: 0 } },
  fit: {},
};
async function editor() {
  const api = vi.fn().mockImplementation(async () => structuredClone(saved));
  const wrapper = mount(CalibrationEditor, {
    global: {
      provide: {
        [reviewKey]: { api, url: (p: string) => `/api/${p}`, hands: ref([]) },
      },
    },
  });
  await flushPromises();
  wrapper.get("canvas").element.getBoundingClientRect = () =>
    new DOMRect(0, 0, 1000, 1000);
  return { wrapper, api };
}
async function drag(wrapper: VueWrapper) {
  await wrapper
    .get("canvas")
    .trigger("pointerdown", { clientX: 150, clientY: 150, pointerId: 1 });
  await wrapper
    .get("canvas")
    .trigger("pointermove", { clientX: 160, clientY: 150, pointerId: 1 });
  await wrapper.get("canvas").trigger("pointerup");
}
function button(wrapper: VueWrapper, label: string) {
  const found = wrapper
    .findAll("button")
    .find((button) => button.text() === label);
  if (!found) throw new Error(`Missing button: ${label}`);
  return found;
}
it("resizes in source coordinates and sends only changed regions plus overhead", () => {
  const data = structuredClone(saved),
    dirty = new Set(["hand:TL"]);
  const hit = hitRegion(
    { ...data, regions: { "hand:TL": data.regions["hand:TL"] } },
    150,
    150,
  );
  if (!hit) throw new Error("Expected a draggable region.");
  data.regions["hand:TL"].rect = dragRegion(hit, 160, 150, data.frame);
  expect(
    calibrationBody(data, dirty, { center: [500, 450], angle: 1, scale: 1.2 }),
  ).toEqual({
    hand: { TL: { rect: [110, 100, 100, 100], roll: 3 } },
    overhead: { center: [500, 450], angle: 1, scale: 1.2 },
  });
});
it.each([
  [-500, -500, [0, 0, 100, 100]],
  [1500, 1500, [900, 900, 100, 100]],
])("keeps moved boxes inside the image at %s,%s", (x, y, expected) => {
  const hit = hitRegion(saved, 150, 150)!;
  expect(dragRegion(hit, x as number, y as number, saved.frame)).toEqual(
    expected,
  );
});
it.each([
  [100, 100, -50, -50, [0, 0, 200, 200]],
  [200, 200, 1100, 1100, [100, 100, 900, 900]],
])(
  "keeps resized corners inside the image",
  (startX, startY, x, y, expected) => {
    const hit = hitRegion(saved, startX as number, startY as number)!;
    expect(dragRegion(hit, x as number, y as number, saved.frame)).toEqual(
      expected,
    );
  },
);
it("never saves a negative coordinate after dragging beyond the preview", async () => {
  const { wrapper, api } = await editor();
  const canvas = wrapper.get("canvas");
  await canvas.trigger("pointerdown", {
    clientX: 150,
    clientY: 150,
    pointerId: 1,
  });
  await canvas.trigger("pointermove", {
    clientX: -10,
    clientY: 150,
    pointerId: 1,
  });
  await canvas.trigger("pointerup");
  await button(wrapper, "Save changes").trigger("click");
  await flushPromises();
  expect(api).toHaveBeenCalledWith("calib", {
    hand: { TL: { rect: [0, 100, 100, 100], roll: 3 } },
  });
});
it("blocks saving and checking an out-of-bounds overhead centre", async () => {
  const { wrapper, api } = await editor();
  await wrapper.get('input[type="number"]').setValue(-2);
  expect(wrapper.get('[role="alert"]').text()).toContain("inside the image");
  expect(button(wrapper, "Save changes").attributes("disabled")).toBeDefined();
  expect(button(wrapper, "Check borders").attributes("disabled")).toBeDefined();
  await button(wrapper, "Apply").trigger("click");
  expect(api).toHaveBeenCalledTimes(1);
});
it("rejects an invalid saved rectangle until it is corrected", () => {
  const data = structuredClone(saved);
  data.regions["hand:TL"].rect[0] = -2;
  expect(() => calibrationBody(data, ["hand:TL"])).toThrow("inside the image");
});
it("saves dragged regions before checking borders", async () => {
  const { wrapper, api } = await editor();
  await drag(wrapper);
  await button(wrapper, "Check borders").trigger("click");
  await flushPromises();
  expect(api.mock.calls.slice(1)).toEqual([
    ["calib", { hand: { TL: { rect: [110, 100, 100, 100], roll: 3 } } }],
    ["calib"],
    ["calib/check", {}],
  ]);
  expect(wrapper.emitted("dirty")?.some(([dirty]) => dirty)).toBe(true);
});
it("checks unchanged geometry without creating a human calibration", async () => {
  const { wrapper, api } = await editor();
  await button(wrapper, "Check borders").trigger("click");
  await flushPromises();
  expect(api.mock.calls.slice(1)).toEqual([["calib/check", {}]]);
});
it("keeps a failed save dirty and prevents border checking and remeasurement", async () => {
  const { wrapper, api } = await editor();
  await drag(wrapper);
  await button(wrapper, "Measure table again").trigger("click");
  expect(api).toHaveBeenCalledTimes(1);
  api.mockRejectedValue(new Error("Cannot save"));
  await button(wrapper, "Check borders").trigger("click");
  await flushPromises();
  expect(wrapper.text()).toContain("Cannot save");
  expect(wrapper.emitted("dirty")?.at(-1)).toEqual([true]);
  expect(api.mock.calls.some(([path]) => path === "calib/check")).toBe(false);
  expect(
    button(wrapper, "Save changes").attributes("disabled"),
  ).toBeUndefined();
});
it("discards without a write and releases the navigation guard", async () => {
  const { wrapper, api } = await editor();
  await drag(wrapper);
  await button(wrapper, "Discard changes").trigger("click");
  await flushPromises();
  expect(api.mock.calls).toEqual([["calib"], ["calib"]]);
  expect(wrapper.emitted("dirty")?.at(-1)).toEqual([false]);
});
it("uses the current layout overhead before the first fit exists", async () => {
  const { wrapper, api } = await editor();
  api.mockResolvedValue({
    ...structuredClone(saved),
    fit: null,
    overhead: { center: [960, 540], angle: 45, scale: 1 },
  });
  await drag(wrapper);
  await button(wrapper, "Discard changes").trigger("click");
  await flushPromises();
  const values = wrapper
    .findAll<HTMLInputElement>(".calibration-details input")
    .map((input) => input.element.value);
  expect(values).toEqual(["960", "540", "45", "1"]);
});
