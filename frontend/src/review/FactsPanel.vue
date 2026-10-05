<script setup lang="ts">
import { computed, ref } from "vue";
import type { Fact, HandEntry, IgnoredAnswer } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import { factLabel, seatName, time } from "./format";
import { errorText } from "../shared/useAction";
import TileFace from "./TileFace.vue";
import TilePalette from "./TilePalette.vue";
const props = defineProps({
  hand: { type: Number, required: true },
  entry: { type: Object as PropType<HandEntry>, required: true },
  ignored: { type: Array as PropType<IgnoredAnswer[]>, default: () => [] },
});
const review = useReview(),
  editing = ref<number | null>(null);
const facts = computed(() => review.facts.value[props.hand] || []);
const reason = (fact: Fact) =>
  props.ignored.find((row) => row.ts != null && row.ts === fact.ts)?.reason;
const unmatched = computed(() =>
  props.ignored.filter(
    (row) => row.ts == null || !facts.value.some((fact) => fact.ts === row.ts),
  ),
);
async function remove(fact: Fact) {
  try {
    await review.remove(fact);
  } catch (error) {
    review.error.value = errorText(error);
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
    review.error.value = errorText(error);
  }
}
</script>
<template>
  <details :open="ignored.length > 0">
    <summary>Saved answers ({{ facts.length }})</summary>
    <ul>
      <li v-for="fact in facts" :key="fact.ts">
        <b>{{ factLabel(fact.kind) }}</b> {{ seatName(entry, fact.seat) }}
        {{ fact.t != null ? time(fact.t) : "" }} {{ fact.type }}
        <span v-if="fact.tile" class="tile"><TileFace :tile="fact.tile" /></span
        ><span
          v-for="(tile, index) in fact.tiles || []"
          :key="index"
          class="tile"
          ><TileFace :tile="tile" /></span
        >{{ fact.kind === "note" ? fact.text : fact.seats?.join(" ")
        }}<span v-if="fact.kind === 'site_wrong'"
          >site {{ fact.site?.join("/") }} → {{ fact.han }}/{{ fact.fu }}</span
        ><span v-if="fact.author !== 'tool'">
          ({{ fact.author || fact.source || "Imported answer" }})</span
        ><span v-if="reason(fact)" class="warn">
          Not applied: {{ reason(fact) }}</span
        ><button v-if="fact.ts" @click="remove(fact)">Delete</button
        ><button v-if="fact.ts && fact.tile" @click="editing = fact.ts">
          Edit tile</button
        ><TilePalette
          v-if="editing === fact.ts"
          :guess="fact.tile"
          @pick="replace(fact, $event)"
        />
      </li>
      <li v-for="(row, index) in unmatched" :key="`ignored-${index}`">
        <b>{{ factLabel(row.kind) }}</b>
        <span class="warn"> Not applied: {{ row.reason }}</span>
      </li>
    </ul>
  </details>
</template>
