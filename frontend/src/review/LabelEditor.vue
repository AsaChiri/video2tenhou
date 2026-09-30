<script setup lang="ts">
import { onBeforeUnmount, ref } from "vue";
import type { LabelBox, Point, Reading } from "../types";
import { useReview } from "./context";
import TilePalette from "./TilePalette.vue";
const review = useReview(),
  region = ref("hand:TL"),
  at = ref(0),
  boxes = ref<LabelBox[]>([]),
  canvas = ref<HTMLCanvasElement | null>(null),
  selected = ref<number | null>(null),
  status = ref(""),
  busy = ref(false),
  loaded = ref<{ region: string; t: number } | null>(null);
let image: HTMLImageElement | undefined,
  size: Point = [0, 0],
  drag: Point | null = null,
  request = 0;
onBeforeUnmount(() => {
  request++;
  if (image) image.onload = null;
});
function draw() {
  if (!canvas.value || !image?.naturalWidth) return;
  canvas.value.width = size[0];
  canvas.value.height = size[1];
  const context = canvas.value.getContext("2d");
  if (!context) return;
  context.drawImage(image, 0, 0);
  context.lineWidth = 2;
  context.font = "bold 18px sans-serif";
  for (const box of boxes.value) {
    const [x, y, right, bottom] = box.xyxy;
    context.strokeStyle =
      box.role === "indicator" ? "#f00" : box.sideways ? "#f0f" : "#0f0";
    context.strokeRect(x, y, right - x, bottom - y);
    context.fillStyle = "#000";
    context.fillRect(x, y - 18, 34, 18);
    context.fillStyle = "#ff0";
    context.fillText(box.tile, x + 2, y - 3);
  }
}
async function load() {
  const token = ++request,
    target = { region: region.value, t: at.value };
  busy.value = true;
  status.value = "Reading (the models run on the frame)…";
  try {
    const result = await review.api<Reading>(
      `read?t=${target.t}&region=${target.region}`,
    );
    if (token !== request) return;
    boxes.value = result.boxes;
    size = result.size;
    selected.value = null;
    loaded.value = target;
    image = new Image();
    image.onload = () => {
      if (token === request) draw();
    };
    image.src = review.url(`frame?t=${target.t}&region=${target.region}`);
    status.value = `${boxes.value.length} boxes prefilled — check each tile`;
  } catch (error) {
    status.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}
function point(event: PointerEvent): Point {
  if (!canvas.value) throw new Error("Canvas is not mounted.");
  const bounds = canvas.value.getBoundingClientRect();
  return [
    ((event.clientX - bounds.left) * size[0]) / bounds.width,
    ((event.clientY - bounds.top) * size[1]) / bounds.height,
  ];
}
function down(event: PointerEvent) {
  const [x, y] = point(event),
    hit = boxes.value.findIndex(
      (box) =>
        x >= box.xyxy[0] &&
        x <= box.xyxy[2] &&
        y >= box.xyxy[1] &&
        y <= box.xyxy[3],
    );
  if (event.shiftKey && hit >= 0) {
    boxes.value.splice(hit, 1);
    selected.value = null;
    draw();
    return;
  }
  if (hit >= 0) {
    selected.value = hit;
    return;
  }
  drag = [x, y];
  canvas.value?.setPointerCapture?.(event.pointerId);
}
function up(event: PointerEvent) {
  if (!drag) return;
  const [x, y] = point(event),
    [left, top] = drag;
  drag = null;
  if (Math.abs(x - left) <= 8 || Math.abs(y - top) <= 8) return;
  boxes.value.push({
    xyxy: [
      Math.min(x, left),
      Math.min(y, top),
      Math.max(x, left),
      Math.max(y, top),
    ],
    tile: "?",
    sideways: false,
    role: "tile",
  });
  selected.value = boxes.value.length - 1;
  draw();
}
function pick(tile: string) {
  if (selected.value === null) return;
  boxes.value[selected.value].tile = tile;
  draw();
}
async function save() {
  if (!loaded.value || busy.value) return;
  if (
    boxes.value.some((box) => box.tile === "?") &&
    !window.confirm('Some boxes are still "?". Save anyway?')
  )
    return;
  busy.value = true;
  try {
    const [kind, corner] = loaded.value.region.split(":");
    const result = await review.api<{ saved: string }>("label", {
      t: loaded.value.t,
      kind,
      corner,
      boxes: boxes.value,
    });
    status.value = "Saved " + result.saved;
  } catch (error) {
    status.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <section>
    <h2>Label tiles for training</h2>
    <p>
      Load a region and time. Click a box to change its tile, drag on empty
      space to add a box, or shift-click a box to delete it.
    </p>
    <label
      >Region<select v-model="region">
        <template v-for="kind in ['hand', 'pond', 'meld']" :key="kind"
          ><option v-for="corner in ['TL', 'TR', 'BL', 'BR']" :key="corner">
            {{ kind }}:{{ corner }}
          </option></template
        >
      </select></label
    ><label
      >Time (seconds)<input v-model.number="at" type="number" min="0"
    /></label>
    <div class="actions">
      <button :disabled="busy" @click="load">Load</button
      ><button :disabled="busy || !loaded" @click="save">Save label</button
      ><span role="status">{{ status }}</span>
    </div>
    <canvas
      ref="canvas"
      aria-label="Training tile boxes"
      @pointerdown="down"
      @pointerup="up"
      @pointercancel="drag = null"
    />
    <div v-if="selected != null">
      <p>Tile of this box: {{ boxes[selected].tile }}</p>
      <TilePalette :guess="boxes[selected].tile" @pick="pick" /><button
        @click="pick('?')"
      >
        Unknown</button
      ><button
        @click="
          boxes[selected].sideways = !boxes[selected].sideways;
          draw();
        "
      >
        Turned sideways</button
      ><button @click="selected = null">Close</button>
    </div>
  </section>
</template>
<style scoped>
canvas {
  max-width: 100%;
  touch-action: none;
  border: 1px solid #666;
}
</style>
