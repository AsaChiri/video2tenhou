<script setup lang="ts">
import { ref } from "vue";
import type { Decode, FactBody, HandEntry, ReviewItem } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import { cornerOf, seatName, time } from "./format";
import { handEnd } from "./evidence";
import { errorText } from "../shared/useAction";
import EvidenceDetails from "./EvidenceDetails.vue";
import TilePalette from "./TilePalette.vue";
import TileListEditor from "./TileListEditor.vue";
import MeldEditor from "./MeldEditor.vue";
const props = defineProps({
  item: { type: Object as PropType<ReviewItem>, required: true },
  entry: { type: Object as PropType<HandEntry>, required: true },
  decode: { type: Object as PropType<Decode>, required: true },
});
const review = useReview();
const editedHands = ref<Record<string, string[]>>({});
async function save(fact: FactBody) {
  try {
    await review.save({ ...fact, hand: props.item.hand });
  } catch (error) {
    review.error.value = errorText(error);
  }
}
</script>
<template>
  <p>{{ item.text }}</p>
  <ul v-if="item.violations?.length">
    <li v-for="(violation, index) in item.violations" :key="index">
      {{ violation.seat ? `${seatName(entry, violation.seat)} ` : ""
      }}{{ violation.text
      }}{{ violation.t != null ? ` (${time(violation.t)})` : "" }}
    </li>
  </ul>
  <div v-for="(over, index) in item.over || []" :key="index" class="card">
    <p>
      The tile set has {{ over.limit ?? 4 }} of {{ over.tile }}; this hand uses
      {{ over.count }}:
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
        <template v-if="(source.pos?.[0] ?? 0) < 0"
          ><p>
            It was called at once and never lay in the pond: correct the called
            tile in its meld.
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
          :at="handEnd(decode, entry)"
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
  <template v-if="item.fact_seat"
    ><p>
      Saved {{ item.fact_kind === "haipai" ? "starting" : "final" }} hand of
      {{ seatName(entry, item.fact_seat) }}: compare it with the discards and
      calls below, then correct it or delete it in Advanced review.
    </p>
    <EvidenceDetails
      :region="`hand:${cornerOf(entry, item.fact_seat)}`"
      :start="handEnd(decode, entry) - 3"
      :end="handEnd(decode, entry) + 30"
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
</template>
