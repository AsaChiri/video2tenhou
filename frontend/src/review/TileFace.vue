<script setup lang="ts">
import { tileImage } from "./format";
defineProps({ tile: { type: String, default: "?" } });
</script>
<template>
  <span class="g"
    ><img
      v-if="tileImage(tile)"
      :src="tileImage(tile) || undefined"
      :alt="tile"
      loading="lazy"
    /><span v-else>?</span></span
  >{{ tile }}
</template>
