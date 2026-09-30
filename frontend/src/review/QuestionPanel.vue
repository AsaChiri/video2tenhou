<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useReview } from "./context";
import { cornerOf, decisionLabel, roundName, seatName } from "./format";
import AnswerEditor from "./AnswerEditor.vue";
import EvidencePlayer from "./EvidencePlayer.vue";
import EvidenceDetails from "./EvidenceDetails.vue";
import CameraContext from "./CameraContext.vue";
import type { HandData, HandEntry, ReviewItem, Turn, Decode } from "../types";
interface Loaded {
  item: ReviewItem & { t: number };
  entry: HandEntry;
  turn?: Turn;
  decode: Decode;
}
const review = useReview(),
  loaded = ref<Loaded | null>(null),
  loading = ref(false);
watch(
  review.selection,
  async (selection, _, cleanup) => {
    let current = true;
    cleanup(() => (current = false));
    loaded.value = null;
    loading.value = false;
    if (!selection) return;
    loading.value = true;
    try {
      const result = await review.api<HandData>(`hand/${selection.item.hand}`);
      if (!current) return;
      const item: ReviewItem =
        selection.choice === null
          ? { ...selection.item }
          : {
              ...selection.decision,
              hand: selection.item.hand,
              idx: selection.item.idx,
              kind: selection.decision.field || selection.decision.kind || "",
              tile:
                selection.decision.field === "draw" &&
                typeof selection.decision.value === "string"
                  ? selection.decision.value
                  : undefined,
              tiles:
                selection.decision.field === "haipai" &&
                Array.isArray(selection.decision.value)
                  ? selection.decision.value
                  : undefined,
            };
      const turn = result.decode?.turns?.find(
        (turn) => turn.seat === item.seat && turn.j === item.j,
      );
      item.t ??=
        turn?.t ??
        result.entry.t_last ??
        result.entry.play_window?.[1] ??
        result.entry.t_end;
      if (item.kind === "haipai")
        item.t = selection.decision.t ?? result.entry.t_start;
      loaded.value = {
        item: { ...item, t: item.t ?? 0 },
        entry: result.entry,
        decode: result.decode || {},
        turn,
      };
    } catch (error) {
      if (current)
        review.error.value =
          error instanceof Error ? error.message : String(error);
    } finally {
      if (current) loading.value = false;
    }
  },
  { immediate: true },
);
const span = computed(() => {
  if (!loaded.value) return [0, 0];
  const { item, entry, turn, decode } = loaded.value,
    t = item.t ?? 0;
  if (item.kind === "draw")
    return [
      Math.max(entry.t_start || 0, (turn?.t_prev ?? t - 15) - 6),
      t + (t >= (decode.t_last ?? 0) - 1 ? 30 : 4),
    ];
  if (item.kind === "discard") return [t - 8, t + 6];
  if (item.kind === "result" || item.kind === "ura" || item.kind === "dora")
    return item.guess
      ? [t - 5, t + 25]
      : [(entry.t_last ?? t) - 3, (entry.t_last ?? t) + 30];
  return [Math.max(entry.t_start || 0, t - 10), t + 10];
});
</script>
<template>
  <p v-if="loading" role="status">Loading evidence…</p>
  <p v-else-if="!loaded">Evidence could not be loaded.</p>
  <template v-else
    ><p class="question-context">
      Hanchan {{ loaded.entry.game + 1 }} · {{ roundName(loaded.entry) }}
    </p>
    <div class="workbench">
      <EvidencePlayer
        :key="`${loaded.item.hand}:${loaded.item.idx}:${review.selection.value?.choice}`"
        :start="span[0]"
        :end="span[1]"
      />
      <div class="answer-panel">
        <h2>
          {{
            loaded.item.seat
              ? seatName(loaded.entry, loaded.item.seat) + " · "
              : ""
          }}{{ decisionLabel(loaded.item) }}
        </h2>
        <AnswerEditor
          :key="`${loaded.item.hand}:${loaded.item.idx}:${review.selection.value?.choice}`"
          :item="loaded.item"
          :entry="loaded.entry"
          :decode="loaded.decode"
        />
      </div>
    </div>
    <div class="review-details">
      <details>
        <summary>Additional evidence</summary>
        <template v-if="loaded.item.seat"
          ><EvidenceDetails
            :region="`hand:${cornerOf(loaded.entry, loaded.item.seat)}`"
            :start="span[0]"
            :end="span[1]"
            label="Hand camera" /><EvidenceDetails
            :region="`pond:${cornerOf(loaded.entry, loaded.item.seat)}`"
            :at="loaded.item.t + 2"
            label="Pond after the discard" /><EvidenceDetails
            :region="`meld:${cornerOf(loaded.entry, loaded.item.seat)}`"
            :at="loaded.item.t + 2"
            label="Meld camera" /><CameraContext
            :hand="loaded.item.hand"
            :seat="loaded.item.seat"
            :at="loaded.item.t" /></template
        ><EvidenceDetails
          v-for="offset in [6, 14, 22]"
          :key="offset"
          region="overhead"
          :at="(loaded.entry.t_last ?? loaded.item.t) + offset"
          :label="`Overhead at end + ${offset}s`"
        />
      </details></div
  ></template>
</template>
