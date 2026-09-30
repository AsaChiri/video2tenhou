<script setup lang="ts">
import { ref } from "vue";
import TileFace from "./TileFace.vue";
import TilePalette from "./TilePalette.vue";
const tiles = defineModel<string[]>({ default: () => [] });
defineProps({ disabled: Boolean });
const selected = ref<number | null>(null);
function pick(tile: string) {
  const next = [...tiles.value];
  if (selected.value === null) next.push(tile);
  else next[selected.value] = tile;
  selected.value = null;
  tiles.value = next;
}
</script>
<template>
  <div class="slots">
    <button
      v-for="(tile, index) in tiles"
      :key="index"
      type="button"
      class="tile"
      :class="{ sel: selected === index }"
      :disabled="disabled"
      :aria-label="`Replace tile ${index + 1}: ${tile}`"
      @click="selected = index"
    >
      <TileFace :tile="tile" /></button
    ><span>{{ tiles.length }} tiles</span
    ><button
      type="button"
      :disabled="disabled || !tiles.length"
      @click="
        tiles = tiles.slice(0, -1);
        selected = null;
      "
    >
      Remove last
    </button>
  </div>
  <TilePalette :disabled="disabled" @pick="pick" />
</template>
