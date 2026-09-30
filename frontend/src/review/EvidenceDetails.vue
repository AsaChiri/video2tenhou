<script setup lang="ts">
import { ref } from "vue";
import type { PropType } from "vue";
import { useReview } from "./context";
import EvidencePlayer from "./EvidencePlayer.vue";
defineProps({
  label: String,
  region: String,
  start: Number,
  end: Number,
  at: Number,
  box: Array as PropType<number[]>,
  scale: { type: Number, default: 0.6 },
});
const review = useReview(),
  open = ref(false);
</script>
<template>
  <details
    class="extra-evidence"
    @toggle="open = ($event.target as HTMLDetailsElement).open"
  >
    <summary>{{ label }}</summary>
    <template v-if="open"
      ><EvidencePlayer
        v-if="end != null"
        :start="start"
        :end="end"
        :region="region" />
      <figure v-else>
        <a :href="review.url(`frame?t=${at}&region=frame`)" target="_blank"
          ><div class="crop">
            <img
              :src="
                review.url(
                  `frame?t=${at}&region=${encodeURIComponent(region || 'frame')}&scale=${scale}`,
                )
              "
              :alt="label"
            /><span
              v-if="box"
              class="crop-box"
              :style="{
                left: box[0] * scale + 'px',
                top: box[1] * scale + 'px',
                width: (box[2] - box[0]) * scale + 'px',
                height: (box[3] - box[1]) * scale + 'px',
              }"
            /></div
        ></a></figure
    ></template>
  </details>
</template>
<style scoped>
.crop {
  position: relative;
  display: inline-block;
}
.crop-box {
  position: absolute;
  border: 3px solid #ff1744;
  pointer-events: none;
}
img {
  max-width: 100%;
}
</style>
