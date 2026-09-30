<script setup lang="ts">
import { computed, onBeforeUnmount, onDeactivated, ref, watch } from "vue";
import { useReview } from "./context";
import { time } from "./format";
const props = defineProps({
  start: Number,
  end: Number,
  region: { type: String, default: "frame" },
});
const review = useReview(),
  video = ref<HTMLVideoElement | null>(null),
  before = ref(0),
  after = ref(0);
const start = computed(() => Math.max(0, (props.start ?? 0) + before.value));
const end = computed(() => (props.end ?? 0) + after.value);
const url = computed(() =>
  review.url(
    `clip?t0=${start.value.toFixed(1)}&t1=${end.value.toFixed(1)}&region=${encodeURIComponent(props.region)}`,
  ),
);
let playing = false;
function playback(value: boolean) {
  if (value !== playing) {
    playing = value;
    review.players.value += value ? 1 : -1;
  }
}
watch(
  () => [props.start, props.end, props.region],
  () => {
    before.value = after.value = 0;
    playback(false);
  },
);
onBeforeUnmount(() => {
  video.value?.pause();
  playback(false);
});
onDeactivated(() => {
  video.value?.pause();
  playback(false);
});
</script>
<template>
  <div class="watch">
    <video
      ref="video"
      controls
      preload="metadata"
      :src="url"
      @play="playback(true)"
      @pause="playback(false)"
      @ended="playback(false)"
      @emptied="playback(false)"
    />
    <div class="actions">
      <button @click="before -= 15">15 s earlier</button
      ><span>{{ time(start) }}–{{ time(end) }}</span
      ><button @click="after += 15">15 s later</button>
    </div>
  </div>
</template>
