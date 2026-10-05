<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useReview } from "./context";
import { cornerOf, decisionLabel, roundName, seatName, time } from "./format";
import { handEnd, questionWindow } from "./evidence";
import { errorText } from "../shared/useAction";
import AnswerEditor from "./AnswerEditor.vue";
import EvidencePlayer from "./EvidencePlayer.vue";
import EvidenceDetails from "./EvidenceDetails.vue";
import CameraContext from "./CameraContext.vue";
import type {
  Decision,
  HandData,
  HandEntry,
  ReviewItem,
  Decode,
} from "../types";
interface Loaded {
  key: string;
  item: ReviewItem & { t: number };
  entry: HandEntry;
  decode: Decode;
  span: [number, number];
}
const review = useReview(),
  loaded = ref<Loaded | null>(null),
  loading = ref(false);

/** The question as one item: a grouped choice carries its own field and value. */
function question({ item, choice }: Decision): ReviewItem {
  if (!choice) return { ...item };
  const kind = choice.field || choice.kind || "";
  return {
    ...choice,
    id: choice.id || item.id,
    hand: item.hand,
    kind,
    tile:
      kind === "draw" && typeof choice.value === "string"
        ? choice.value
        : undefined,
    tiles:
      kind === "haipai" && Array.isArray(choice.value)
        ? choice.value
        : undefined,
  };
}
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
      const item = question(selection);
      const decode = result.decode || {};
      const turn = decode.turns?.find(
        (turn) => turn.seat === item.seat && turn.j === item.j,
      );
      const t =
        item.kind === "haipai"
          ? (item.t ?? result.entry.t_start)
          : (item.t ?? turn?.t ?? handEnd(decode, result.entry));
      const placed = { ...item, t };
      loaded.value = {
        key: selection.key,
        item: placed,
        entry: result.entry,
        decode,
        span: questionWindow(placed, result.entry, decode, turn),
      };
    } catch (error) {
      if (current) review.error.value = errorText(error);
    } finally {
      if (current) loading.value = false;
    }
  },
  { immediate: true },
);
const end = computed(() =>
  loaded.value ? handEnd(loaded.value.decode, loaded.value.entry) : 0,
);
</script>
<template>
  <p v-if="loading" role="status">Loading evidence…</p>
  <p v-else-if="!loaded">Evidence could not be loaded.</p>
  <template v-else
    ><p class="question-context">
      Hand {{ loaded.entry.hand + 1 }} · Hanchan {{ loaded.entry.game + 1 }} ·
      {{ roundName(loaded.entry) }}
    </p>
    <div class="workbench">
      <EvidencePlayer
        :key="loaded.key"
        :start="loaded.span[0]"
        :end="loaded.span[1]"
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
          :key="loaded.key"
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
            :start="loaded.span[0]"
            :end="loaded.span[1]"
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
          :at="end + offset"
          :label="`Overhead at ${time(end + offset)}`"
        />
      </details></div
  ></template>
</template>
