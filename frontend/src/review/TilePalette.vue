<script setup lang="ts">
import { paletteRows } from "./format";
import TileFace from "./TileFace.vue";
defineProps({ guess: String, disabled: Boolean });
const emit = defineEmits(["pick"]);
</script>
<template>
  <div class="palette">
    <div v-for="(row, index) in paletteRows" :key="index" class="palette-row">
      <button
        v-for="tile in row"
        :key="tile"
        type="button"
        class="tile"
        :class="{ guess: tile === guess }"
        :aria-label="`Choose ${tile}`"
        :disabled="disabled"
        @click="emit('pick', tile)"
      >
        <TileFace :tile="tile" />
      </button>
    </div>
  </div>
</template>
