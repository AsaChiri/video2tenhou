<script setup lang="ts">
import { ref } from "vue";
import type { Fact } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import { seatName, time } from "./format";
import TileFace from "./TileFace.vue";
import TilePalette from "./TilePalette.vue";
defineProps({
  hand: { type: Number, required: true },
  entry: { type: Object, default: () => ({}) },
  notes: { type: Array as PropType<string[]>, default: () => [] },
});
const review = useReview(),
  editing = ref<number | null>(null);
async function remove(fact: Fact) {
  try {
    await review.remove(fact);
  } catch (error) {
    review.error.value = error instanceof Error ? error.message : String(error);
  }
}
async function replace(fact: Fact, tile: string) {
  try {
    const body = { ...fact };
    delete body.ts;
    delete body.author;
    await review.save({ ...body, tile });
    await review.remove(fact);
    editing.value = null;
  } catch (error) {
    review.error.value = error instanceof Error ? error.message : String(error);
  }
}
</script>
<template>
  <details>
    <summary>
      Facts for this hand ({{ review.facts.value[hand]?.length || 0 }})
    </summary>
    <ul>
      <li v-for="fact in review.facts.value[hand] || []" :key="fact.ts">
        <b>{{ fact.kind }}</b> {{ seatName(entry, fact.seat) }}
        {{ fact.t != null ? time(fact.t) : "" }} {{ fact.type }}
        <span v-if="fact.tile" class="tile"><TileFace :tile="fact.tile" /></span
        ><span
          v-for="(tile, index) in fact.tiles || []"
          :key="index"
          class="tile"
          ><TileFace :tile="tile" /></span
        >{{ fact.text || fact.outcome || fact.seats?.join(" ")
        }}<span v-if="fact.kind === 'site_wrong'"
          >site {{ fact.site?.join("/") }} → {{ fact.han }}/{{ fact.fu }}</span
        ><span v-if="fact.author !== 'tool'">
          ({{ fact.author || fact.source || "Imported answer" }})</span
        ><button v-if="fact.ts" @click="remove(fact)">Delete</button
        ><button v-if="fact.ts && fact.tile" @click="editing = fact.ts">
          Edit tile</button
        ><TilePalette
          v-if="editing === fact.ts"
          :guess="fact.tile"
          @pick="replace(fact, $event)"
        />
      </li>
    </ul>
  </details>
  <details v-if="notes.length">
    <summary>Decoder notes for this hand ({{ notes.length }})</summary>
    <ul>
      <li
        v-for="(note, index) in notes"
        :key="index"
        :class="{
          warn: /ignored|not applied|never observed|matched no call|contradicts/.test(
            note,
          ),
        }"
      >
        {{ note }}
      </li>
    </ul>
  </details>
</template>
