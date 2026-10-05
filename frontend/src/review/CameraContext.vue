<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { CameraData } from "../types";
import { useReview } from "./context";
import { errorText } from "../shared/useAction";
import TileFace from "./TileFace.vue";
const props = defineProps({ hand: Number, seat: String, at: Number });
const review = useReview(),
  open = ref(false),
  data = ref<CameraData | null>(null),
  error = ref("");
const rows = computed<[string, string[] | undefined][]>(() => [
  [
    "Camera before",
    data.value?.read_before?.at(-1)?.tiles?.map((tile) => tile.tile),
  ],
  [
    "Camera after",
    data.value?.read_after?.[0]?.tiles?.map((tile) => tile.tile),
  ],
  ["Reconstruction before draw", data.value?.turn?.hand_before],
  ["Reconstruction after discard", data.value?.turn?.hand_after],
]);
watch(
  [open, () => props.hand, () => props.seat, () => props.at],
  async (_, __, cleanup) => {
    let current = true;
    cleanup(() => (current = false));
    if (!open.value) return;
    try {
      const result = await review.api<CameraData>(
        `context?hand=${props.hand}&seat=${props.seat}&t=${props.at}`,
      );
      if (current) data.value = result;
    } catch (failure) {
      if (current) error.value = errorText(failure);
    }
  },
);
</script>
<template>
  <details @toggle="open = ($event.target as HTMLDetailsElement).open">
    <summary>Compare camera readings and reconstruction</summary>
    <p v-if="error" role="alert">{{ error }}</p>
    <template v-if="data"
      ><div v-for="[label, tiles] in rows" :key="label">
        <h3>{{ label }}</h3>
        <span v-for="(tile, index) in tiles" :key="index" class="tile"
          ><TileFace :tile="tile" /></span
        ><span v-if="!tiles">No calm reading</span>
      </div></template
    >
  </details>
</template>
