<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  watch,
} from "vue";
import { useReview } from "./context";
import { useJob } from "../shared/useJob";
import { errorText, useAction } from "../shared/useAction";
import { time } from "./format";
import {
  calibrationBody,
  calibrationError,
  dragRegion,
  hitRegion,
  MIN_REGION_SIZE,
} from "./calibration";
import EvidenceDetails from "./EvidenceDetails.vue";
import type { Calibration, Drag, JobStatus, Overhead, Point } from "../types";
const emit = defineEmits(["dirty"]);
const review = useReview(),
  jobs = useJob(),
  { busy: saving, error: failure, run } = useAction(),
  canvas = ref<HTMLCanvasElement | null>(null),
  data = ref<Calibration | null>(null),
  dirty = reactive(new Set<string>()),
  status = ref(""),
  selected = ref<string | undefined>(),
  preview = ref<string | null>(null);
const overhead = reactive<Overhead>({ center: [0, 0], angle: 0, scale: 1 });
const overheadDirty = ref(false);
const measuring = computed(() => {
  const job = jobs.job.value;
  return (
    !!job?.running &&
    job.project === review.projectId &&
    (job.kind === "fit" || job.kind === "check")
  );
});
const busy = computed(() => saving.value || measuring.value);
const validationError = computed(() =>
  data.value
    ? calibrationError(data.value, overheadDirty.value ? overhead : undefined)
    : null,
);
const blocked = computed(
  () => busy.value || dirty.size > 0 || overheadDirty.value,
);
const message = computed(() => failure.value || status.value);
let plate: HTMLImageElement,
  drag: Drag | null = null,
  disposed = false;
watch(blocked, (value) => emit("dirty", value), { immediate: true });
const colors: Record<string, string> = {
  pond: "#ff9800",
  hand: "#43a047",
  meld: "#d500f9",
  unit: "#e53935",
  ok: "#2e7d32",
  warn: "#ef6c00",
  fail: "#c62828",
};
const checkLabels: Record<string, string> = {
  ok: "passed",
  warn: "look at it",
  fail: "adjust it",
};
function applyData(value: Calibration) {
  data.value = value;
  Object.assign(overhead, {
    center: [
      ...(value.overhead?.center || value.fit?.overhead?.center || [0, 0]),
    ],
    angle: value.overhead?.angle ?? value.fit?.overhead?.angle ?? 0,
    scale: value.overhead?.scale ?? value.fit?.overhead?.scale ?? 1,
  });
  nextTick(draw);
}
async function load() {
  const result = await review.api<Calibration>("calib");
  if (!disposed && !(dirty.size > 0 || overheadDirty.value)) applyData(result);
}
function loadPlate() {
  plate.src = review.url(`plate?v=${Date.now()}`);
}
function draw() {
  if (!canvas.value || !data.value) return;
  const [width, height] = data.value.frame;
  canvas.value.width = width;
  canvas.value.height = height;
  const context = canvas.value.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  if (plate?.naturalWidth) context.drawImage(plate, 0, 0, width, height);
  for (const [name, region] of Object.entries(data.value.regions)) {
    const check = data.value.checks?.[name],
      color = colors[check?.level || region.kind];
    context.strokeStyle = color;
    context.lineWidth = selected.value === name ? 5 : 3;
    context.beginPath();
    region.quad.forEach(([x, y], index) =>
      index ? context.lineTo(x, y) : context.moveTo(x, y),
    );
    context.closePath();
    context.stroke();
    context.fillStyle = color;
    context.font = "bold 18px system-ui";
    context.fillText(name, region.quad[0][0] + 6, region.quad[0][1] + 20);
    if (region.movable)
      for (const [x, y] of region.quad) context.fillRect(x - 7, y - 7, 14, 14);
  }
}
function point(event: PointerEvent): Point {
  if (!canvas.value || !data.value)
    throw new Error("Calibration is not loaded.");
  const bounds = canvas.value.getBoundingClientRect(),
    scale = data.value.frame[0] / bounds.width;
  return [
    (event.clientX - bounds.left) * scale,
    (event.clientY - bounds.top) * scale,
  ];
}
function down(event: PointerEvent) {
  if (!data.value || busy.value) return;
  drag = hitRegion(data.value, ...point(event));
  selected.value = drag?.name;
  if (drag) canvas.value?.setPointerCapture?.(event.pointerId);
  draw();
}
function move(event: PointerEvent) {
  if (!drag || busy.value || !data.value) return;
  const rect = dragRegion(drag, ...point(event), data.value.frame);
  if (rect[2] < MIN_REGION_SIZE || rect[3] < MIN_REGION_SIZE) return;
  const [x, y, w, h] = rect,
    region = data.value.regions[drag.name];
  region.rect = rect;
  region.quad = [
    [x, y],
    [x + w, y],
    [x + w, y + h],
    [x, y + h],
  ];
  dirty.add(drag.name);
  if (data.value.checks) delete data.value.checks[drag.name];
  status.value = "Unsaved border changes.";
  draw();
}
async function persist() {
  if (!data.value) return;
  const body = calibrationBody(
    data.value,
    dirty,
    overheadDirty.value ? overhead : undefined,
  );
  if (!Object.keys(body).length) return;
  await review.api("calib", body);
  dirty.clear();
  overheadDirty.value = false;
  applyData(await review.api<Calibration>("calib"));
}
const save = () =>
  run(async () => {
    await persist();
    status.value = "Saved.";
  });
const discard = () =>
  run(async () => {
    const result = await review.api<Calibration>("calib");
    dirty.clear();
    overheadDirty.value = false;
    applyData(result);
    status.value = "Changes discarded.";
  });
function measure(kind: "fit" | "check") {
  if (kind === "fit" && blocked.value) {
    status.value = "Save or discard your border changes first.";
    return;
  }
  return run(async () => {
    if (kind === "check") await persist();
    jobs.show(await review.api<JobStatus>(`calib/${kind}`, {}));
    status.value =
      kind === "check" ? "Checking borders…" : "Measuring the table…";
  });
}
jobs.onFinished((job) => {
  if (job.project !== review.projectId || !["fit", "check"].includes(job.kind))
    return;
  status.value =
    job.error ||
    (job.kind === "check"
      ? "Borders checked."
      : "Table measured. Check the borders before analysis.");
  if (job.kind === "fit") loadPlate();
  load().catch((error) => (status.value = errorText(error)));
});
function nudge(x: number, y: number, angle = 0) {
  if (!data.value) return;
  overhead.center[0] = Math.max(
    0,
    Math.min(data.value.frame[0], overhead.center[0] + x),
  );
  overhead.center[1] = Math.max(
    0,
    Math.min(data.value.frame[1], overhead.center[1] + y),
  );
  overhead.angle += angle;
  overheadDirty.value = true;
  save();
}
const previewAt = computed(
  () =>
    review.hands.value[Math.floor(review.hands.value.length / 2)]?.t_start ||
    600,
);
onMounted(() => {
  plate = new Image();
  plate.onload = draw;
  plate.onerror = () => {
    status.value = "Prepare the recording to show the table preview.";
  };
  loadPlate();
  load().catch((error) => (status.value = errorText(error)));
});
onBeforeUnmount(() => {
  disposed = true;
  plate.onload = null;
  plate.onerror = null;
  emit("dirty", false);
});
</script>
<template>
  <section class="calibration-editor">
    <h2>Calibrate this video</h2>
    <p>Drag a box or its corners to adjust a region, then check the borders.</p>
    <div class="actions">
      <button
        class="primary"
        :disabled="!blocked || busy || !!validationError"
        @click="save"
      >
        Save changes</button
      ><button v-if="blocked" :disabled="busy" @click="discard">
        Discard changes</button
      ><button :disabled="busy || !!validationError" @click="measure('check')">
        Check borders</button
      ><span role="status">{{ message }}</span>
    </div>
    <p v-if="validationError" role="alert">{{ validationError }}</p>
    <div class="calibration-preview">
      <canvas
        ref="canvas"
        aria-label="Camera regions on the table preview"
        @pointerdown="down"
        @pointermove="move"
        @pointerup="drag = null"
        @pointercancel="drag = null"
      />
    </div>
    <details class="calibration-details">
      <summary>Overhead adjustment and crop previews</summary>
      <button :disabled="busy" @click="measure('fit')">
        Measure table again</button
      ><template v-if="data"
        ><div class="lay">
          <div @input="overheadDirty = true">
            <label
              >Overhead centre X<input
                v-model.number="overhead.center[0]"
                min="0"
                :max="data.frame[0]"
                type="number" /></label
            ><label
              >Overhead centre Y<input
                v-model.number="overhead.center[1]"
                min="0"
                :max="data.frame[1]"
                type="number" /></label
            ><label
              >Overhead angle<input
                v-model.number="overhead.angle"
                type="number"
                step="any" /></label
            ><label
              >Overhead scale<input
                v-model.number="overhead.scale"
                type="number"
                step="any"
            /></label>
            <div class="actions">
              <button :disabled="busy" @click="nudge(-2, 0)">Left</button
              ><button :disabled="busy" @click="nudge(2, 0)">Right</button
              ><button :disabled="busy" @click="nudge(0, -2)">Up</button
              ><button :disabled="busy" @click="nudge(0, 2)">Down</button
              ><button :disabled="busy" @click="nudge(0, 0, -0.25)">
                Rotate −</button
              ><button :disabled="busy" @click="nudge(0, 0, 0.25)">
                Rotate +</button
              ><button :disabled="busy || !!validationError" @click="save">
                Apply
              </button>
            </div>
          </div>
          <div>
            <div v-for="(region, name) in data.regions" :key="name">
              <template v-if="region.movable"
                ><b>{{ name }}</b
                ><span v-if="data.checks?.[name]">
                  {{ checkLabels[data.checks[name].level] }}</span
                ><button @click="preview = name">
                  Show the crop
                </button></template
              >
            </div>
          </div>
        </div>
        <template v-if="preview"
          ><EvidenceDetails
            v-for="offset in [0, 120, 240]"
            :key="`${preview}:${offset}`"
            :region="preview"
            :at="previewAt + offset"
            :label="`${preview} at ${time(previewAt + offset)}`" /></template
      ></template>
    </details>
  </section>
</template>
<style scoped>
canvas {
  touch-action: none;
}
.actions {
  margin-bottom: 16px;
}
</style>
