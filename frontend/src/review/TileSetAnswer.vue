<script setup lang="ts">
import { ref } from "vue";
import type { Decode, FactBody } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import TileListEditor from "./TileListEditor.vue";
import TilePalette from "./TilePalette.vue";
import EvidenceDetails from "./EvidenceDetails.vue";
import MeldEditor from "./MeldEditor.vue";
const props = defineProps({
  item: { type: Object, required: true },
  decode: { type: Object as PropType<Decode>, required: true },
  kind: { type: String, required: true },
  busy: Boolean,
  answer: {
    type: Function as PropType<
      (body: FactBody) => Promise<boolean | undefined>
    >,
    required: true,
  },
  lost: {
    type: Function as PropType<() => Promise<boolean | undefined>>,
    required: true,
  },
});
const review = useReview();
const tiles = ref([...(props.item.tiles || [])]);
const confirmed = ref(false);
const error = ref("");
const indicators = ref(
  (props.decode.indicators || []).map((indicator) => indicator.tile),
);
function saveTiles() {
  if (!tiles.value.length) {
    error.value = "Enter the tiles first.";
    return;
  }
  if (props.kind === "result") {
    if (!confirmed.value) {
      error.value =
        "Confirm that you compared these tiles with the pictures first.";
      return;
    }
    const melds = (props.decode.calls || []).filter(
      (call) => call.seat === props.item.seat && call.type !== "kakan",
    ).length;
    if (tiles.value.length !== 13 - 3 * melds) {
      error.value = `The concealed hand without the winning tile has ${13 - 3 * melds} tiles here.`;
      return;
    }
  }
  props.answer({
    kind: props.kind === "result" ? "final_hand" : props.kind,
    seat: props.item.seat,
    t: props.item.t,
    tiles: tiles.value,
  });
}
function winning(tile: string) {
  const result = props.decode.result || {},
    turns = props.decode.turns || [];
  if (result.outcome === "tsumo") {
    const mine = turns.filter((turn) => turn.seat === props.item.seat);
    return props.answer({
      kind: "draw",
      seat: props.item.seat,
      j: mine.length ? Math.max(...mine.map((turn) => turn.j)) + 1 : 0,
      tile,
    });
  }
  const last = turns.filter((turn) => turn.seat === result.loser).at(-1);
  if (!last) {
    error.value = "The losing seat has no discard in this reconstruction.";
    return;
  }
  props.answer({ kind: "discard", seat: result.loser, t: last.t, tile });
}
async function indicator(index: number, tile: string) {
  const next = [...indicators.value];
  next[index] = tile;
  const saved = await props.answer({ kind: "dora", tiles: next });
  if (saved) indicators.value = next;
}
</script>
<template>
  <p v-if="error" role="alert" class="warn">{{ error }}</p>
  <div>
    <p>
      {{
        item.text ||
        (kind === "haipai"
          ? "Confirm or correct the starting tiles."
          : kind === "ura"
            ? "Enter the ura indicators revealed at the end of the hand."
            : "Enter the indicator tiles.")
      }}
    </p>
    <div v-if="kind === 'result' && item.han && item.site" class="card">
      <p>
        Site: {{ item.site.join("/") }} han/fu · reconstruction:
        {{ item.han }}/{{ item.fu }} ·
        {{ item.same_payment ? "same payment" : "different payments" }}
      </p>
      <button
        :disabled="busy"
        @click="
          answer({
            kind: 'site_wrong',
            seat: item.seat,
            t: item.t,
            han: item.han,
            fu: item.fu,
            site: item.site,
          })
        "
      >
        The site's han/fu is wrong: keep {{ item.han }}/{{ item.fu }}
      </button>
    </div>
    <p v-if="kind === 'result'">
      Leave out the winning tile, whether ron or tsumo.
    </p>
    <TileListEditor
      v-model="tiles"
      :disabled="busy"
      @update:model-value="review.dirty.value = true"
    />
    <label v-if="kind === 'result'"
      ><input v-model="confirmed" type="checkbox" /> I compared these tiles with
      the pictures; they are the player's final concealed hand.</label
    >
    <div class="actions">
      <button class="primary" :disabled="busy" @click="saveTiles">
        {{ kind === "result" ? "Save final hand" : "Save" }}</button
      ><button
        v-if="kind === 'haipai' || (kind === 'dora' && item.guess)"
        :disabled="busy"
        @click="lost"
      >
        Can't tell from the video
      </button>
    </div>
    <template v-if="kind === 'result'">
      <details>
        <summary>Check or correct the winning tile</summary>
        <p>
          Read as {{ item.win_tile || "unknown" }}. Other score candidates:
          {{ item.candidates?.join(" ") }}
        </p>
        <TilePalette :guess="item.win_tile" :disabled="busy" @pick="winning" />
      </details>
      <details>
        <summary>Check or correct dora indicators</summary>
        <div v-for="(value, index) in [...indicators, null]" :key="index">
          <p>
            {{
              value
                ? `Indicator ${index + 1}: ${value}`
                : "Add a missed indicator"
            }}
          </p>
          <EvidenceDetails
            v-if="decode.indicators?.[index]?.region"
            :region="decode.indicators[index].region"
            :at="decode.indicators[index].t_first + 2"
            :box="decode.indicators[index].box"
            label="Indicator as read"
          /><TilePalette
            :guess="value || undefined"
            :disabled="busy"
            @pick="indicator(index, $event)"
          />
        </div>
      </details>
      <details v-if="decode.calls?.some((call) => call.seat === item.seat)">
        <summary>Check or correct melds</summary>
        <MeldEditor
          v-for="(call, index) in decode.calls.filter(
            (call) => call.seat === item.seat,
          )"
          :key="index"
          :hand="item.hand"
          :seat="call.seat"
          :at="call.t_first"
          :type="call.type"
          :tiles="call.tiles"
          :source="call.source"
        />
      </details>
    </template>
  </div>
</template>
