<script setup lang="ts">
import { computed, ref } from "vue";
import type { Decode, FactBody } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import { cornerOf, seatName, time } from "./format";
import EvidenceDetails from "./EvidenceDetails.vue";
import TilePalette from "./TilePalette.vue";
import TileListEditor from "./TileListEditor.vue";
import MeldEditor from "./MeldEditor.vue";
const props = defineProps({
  item: { type: Object, required: true },
  entry: { type: Object, required: true },
  decode: { type: Object as PropType<Decode>, required: true },
});
const review = useReview();
const editedHands = ref<Record<string, string[]>>({});
const candidates = computed(() =>
  [
    ...(props.item.text || "").matchAll(/([ESWN]) turn (\d+) (\S+) at (\d+)s/g),
  ].map(([, seat, , tile, t]) => ({ seat, tile, t: Number(t) })),
);
async function save(fact: FactBody) {
  try {
    await review.save({ hand: props.item.hand, ...fact });
  } catch (error) {
    review.error.value = error instanceof Error ? error.message : String(error);
  }
}
</script>
<template>
  <p>The hand cannot be reconstructed as read: the rules are violated.</p>
  <div v-for="(over, index) in item.over || []" :key="index" class="card">
    <p>
      The tile set contains {{ over.limit ?? 4 }} copies of {{ over.tile }}, but
      the record uses {{ over.count }}:
    </p>
    <div v-for="(source, i) in over.sources" :key="i" class="card">
      <template v-if="source.kind === 'haipai'">
        <p>
          {{ seatName(entry, source.seat) }}'s starting hand contains
          {{ source.tile }}.
        </p>
        <EvidenceDetails
          :region="`hand:${source.corner}`"
          :start="source.t"
          :end="source.t + 15"
          label="Starting hand"
        />
        <TileListEditor
          :model-value="editedHands[`${index}:${i}`] || source.tiles"
          @update:model-value="editedHands[`${index}:${i}`] = $event"
        />
        <button
          @click="
            save({
              kind: 'haipai',
              seat: source.seat,
              tiles: editedHands[`${index}:${i}`] || source.tiles,
            })
          "
        >
          Save starting hand
        </button>
      </template>
      <template v-else-if="source.kind === 'draw'">
        <p>
          {{ seatName(entry, source.seat) }} drew {{ source.tile }} at
          {{ time(source.t) }}.
        </p>
        <EvidenceDetails
          :region="`hand:${source.corner}`"
          :start="source.t - 15"
          :end="source.t + 4"
          label="Drawn tile"
        />
        <TilePalette
          :guess="source.tile"
          @pick="
            save({
              kind: 'draw',
              seat: source.seat,
              j: source.j,
              t: source.t,
              tile: $event,
            })
          "
        />
      </template>
      <template v-else-if="source.kind === 'replacement_draw'">
        <p>
          {{ seatName(entry, source.seat) }}'s replacement draw at
          {{ time(source.t) }} is {{ source.tile }}.
        </p>
        <EvidenceDetails
          :region="`hand:${source.corner}`"
          :start="source.t - 15"
          :end="source.t + 4"
          label="Replacement draw after the kan"
        />
      </template>
      <template v-else-if="source.kind === 'discard'">
        <p>
          {{ seatName(entry, source.seat) }}'s discard at
          {{ time(source.t) }} was read as {{ source.tile }}.
        </p>
        <template v-if="source.pos?.[0] < 0"
          ><p>
            Called away immediately; this tile never lay in the pond. Correct
            the called tile in its meld.
          </p>
          <EvidenceDetails
            v-if="source.call"
            :region="`meld:${source.call.corner}`"
            :at="source.call.t + 2"
            label="Called tile in meld" /><EvidenceDetails
            :region="`pond:${source.corner}`"
            :start="source.t - 4"
            :end="source.t + 3"
            label="Pond around the discard"
        /></template>
        <template v-else
          ><EvidenceDetails
            :region="`pond:${source.corner}`"
            :at="source.t_pic ?? source.t + 2"
            :box="source.box"
            label="Discard in the pond" /><TilePalette
            :guess="source.tile"
            @pick="
              save({
                kind: 'discard',
                seat: source.seat,
                t: source.t,
                tile: $event,
              })
            "
        /></template>
      </template>
      <template v-else-if="source.kind === 'indicator'"
        ><p>
          {{ source.field === "ura" ? "Ura" : "Dora" }} indicator:
          {{ source.tile }}
        </p>
        <EvidenceDetails
          region="overhead"
          :at="decode.t_last ?? entry.t_end"
          label="Indicator near the end of the hand" /><TilePalette
          :guess="source.tile"
          @pick="
            save({
              kind: source.field === 'ura' ? 'ura' : 'dora',
              tiles: (source.field === 'ura'
                ? decode.ura || []
                : decode.dora || [source.tile]
              ).map((tile, index) =>
                (
                  source.index != null
                    ? index === source.index
                    : tile === source.tile
                )
                  ? $event
                  : tile,
              ),
            })
          "
      /></template>
      <template v-else
        ><p>
          {{ seatName(entry, source.seat) }}'s meld at {{ time(source.t) }}
        </p>
        <EvidenceDetails
          :region="`meld:${source.corner}`"
          :at="source.t + 2"
          label="Meld camera" /><MeldEditor
          :hand="item.hand"
          :seat="source.seat"
          :at="source.t"
          :type="source.type"
          :tiles="source.tiles"
          :source="source.source"
      /></template>
    </div>
  </div>
  <div v-for="(candidate, index) in candidates" :key="index">
    <p>
      {{ seatName(entry, candidate.seat) }} discards at {{ time(candidate.t) }}
    </p>
    <EvidenceDetails
      :region="`pond:${cornerOf(entry, candidate.seat)}`"
      :at="candidate.t + 2"
      label="Pond after the discard"
    /><TilePalette
      :guess="candidate.tile"
      @pick="
        save({
          kind: 'discard',
          seat: candidate.seat,
          t: candidate.t,
          tile: $event,
        })
      "
    />
  </div>
  <template v-if="item.fact_seat"
    ><p>
      Check the saved
      {{ item.fact_kind === "haipai" ? "starting" : "final" }} hand for
      {{ seatName(entry, item.fact_seat) }} against the discards and calls.
      Incorrect saved facts can be deleted below.
    </p>
    <EvidenceDetails
      :region="`hand:${cornerOf(entry, item.fact_seat)}`"
      :start="(decode.t_last ?? entry.t_end) - 3"
      :end="(decode.t_last ?? entry.t_end) + 30"
      label="Final hand reveal" /><MeldEditor
      v-for="(call, index) in (decode.calls || []).filter(
        (call) => call.seat === item.fact_seat,
      )"
      :key="index"
      :hand="item.hand"
      :seat="call.seat"
      :at="call.t_first"
      :type="call.type"
      :tiles="call.tiles"
      :source="call.source" />
    <details
      v-for="turn in (decode.turns || []).filter(
        (turn) => turn.seat === item.fact_seat && turn.discard,
      )"
      :key="turn.i"
    >
      <summary>{{ time(turn.t) }} {{ turn.discard }}</summary>
      <EvidenceDetails
        :region="`pond:${cornerOf(entry, turn.seat)}`"
        :at="turn.t + 2"
        :box="turn.discard_box"
        label="Pond after the discard"
      /><TilePalette
        :guess="turn.discard"
        @pick="
          save({ kind: 'discard', seat: turn.seat, t: turn.t, tile: $event })
        "
      /></details
  ></template>
  <details>
    <summary>Program's diagnosis</summary>
    <p>{{ item.text }}</p>
  </details>
</template>
