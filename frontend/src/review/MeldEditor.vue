<script setup lang="ts">
import { ref } from "vue";
import type { PropType } from "vue";
import { meldFact } from "./meld";
import { useReview } from "./context";
import TileListEditor from "./TileListEditor.vue";
const props = defineProps({
  hand: { type: Number, required: true },
  seat: String,
  at: Number,
  type: { type: String, default: "pon" },
  tiles: { type: Array as PropType<string[]>, default: () => [] },
  source: String,
});
const review = useReview(),
  type = ref(props.type),
  source = ref(props.source || "kamicha"),
  tiles = ref(props.tiles.filter((tile) => tile !== "X" && tile !== "?")),
  busy = ref(false),
  error = ref("");
async function save(remove = false) {
  if (busy.value) return;
  busy.value = true;
  try {
    await review.save(
      remove
        ? {
            kind: "meld_remove",
            hand: props.hand,
            seat: props.seat,
            t: props.at,
            type: props.type,
          }
        : {
            ...meldFact({
              hand: props.hand,
              seat: props.seat,
              t: props.at,
              type: type.value,
              tiles: tiles.value,
              source: source.value,
            }),
            hand: props.hand,
          },
    );
  } catch (failure) {
    error.value = failure instanceof Error ? failure.message : String(failure);
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <div
    class="card"
    @input="review.dirty.value = true"
    @change="review.dirty.value = true"
  >
    <label
      >Type
      <select v-model="type">
        <option
          v-for="kind in ['chi', 'pon', 'kan', 'ankan', 'kakan']"
          :key="kind"
        >
          {{ kind }}
        </option>
      </select></label
    ><label
      >Called from
      <select v-model="source" :disabled="type === 'ankan' || type === 'chi'">
        <option value="kamicha">Kamicha (previous player)</option>
        <option value="toimen">Toimen (opposite)</option>
        <option value="shimocha">Shimocha (next player)</option>
      </select></label
    >
    <p>Enter the called tile first; click a slot to replace it.</p>
    <TileListEditor
      v-model="tiles"
      :disabled="busy"
      @update:model-value="review.dirty.value = true"
    />
    <p v-if="error" role="alert">{{ error }}</p>
    <div class="actions">
      <button :disabled="busy" @click="save()">Save this meld</button
      ><button :disabled="busy" @click="save(true)">
        This {{ props.type }} is not real: remove it
      </button>
    </div>
  </div>
</template>
