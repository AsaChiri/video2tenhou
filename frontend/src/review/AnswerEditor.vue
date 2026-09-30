<script setup lang="ts">
import { computed, ref } from "vue";
import type { Decode, FactBody } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import { cornerOf, seats, seatName, time } from "./format";
import TilePalette from "./TilePalette.vue";
import TileSetAnswer from "./TileSetAnswer.vue";
import EvidenceDetails from "./EvidenceDetails.vue";
import MeldEditor from "./MeldEditor.vue";
import ConflictEditor from "./ConflictEditor.vue";
const props = defineProps({
  item: { type: Object, required: true },
  entry: { type: Object, required: true },
  decode: { type: Object as PropType<Decode>, default: () => ({}) },
});
const review = useReview(),
  busy = ref(false),
  error = ref("");
const missingSeat = ref(props.item.seat || "E"),
  missingTime = ref(props.item.t || 0);
const kind = computed(() =>
  props.item.kind === "conflict" &&
  ["meld", "result", "riichi"].includes(props.item.culprit)
    ? props.item.culprit === "meld"
      ? "call"
      : props.item.culprit
    : props.item.kind,
);
const fromSeats = computed(() =>
  props.item.source
    ? [
        seats[
          (seats.indexOf(props.item.seat) +
            ({ kamicha: 3, toimen: 2, shimocha: 1 } as Record<string, number>)[
              props.item.source
            ]) %
            4
        ],
      ]
    : seats.filter((seat) => seat !== props.item.seat),
);
async function answer(body: FactBody) {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    await review.save({ hand: props.item.hand, ...body });
    return true;
  } catch (failure) {
    error.value = failure instanceof Error ? failure.message : String(failure);
    return false;
  } finally {
    busy.value = false;
  }
}
function tileAnswer(tile: string) {
  const item = props.item;
  return answer({
    kind: kind.value,
    seat: item.seat,
    t: item.t,
    tile,
    ...(kind.value === "draw" ? { j: item.j, t_discard: item.t } : {}),
  });
}
function lost() {
  const item = props.item;
  return answer({
    kind: "lost",
    ...(kind.value === "draw"
      ? { seat: item.seat, j: item.j, t: item.t }
      : {
          field: kind.value,
          t: item.t,
          ...(kind.value === "haipai" ? { seat: item.seat } : {}),
        }),
  });
}
</script>
<template>
  <div @input="review.dirty.value = true" @change="review.dirty.value = true">
    <p v-if="error" role="alert" class="warn">{{ error }}</p>
    <template v-if="kind === 'draw' || kind === 'discard'"
      ><p>
        {{
          kind === "draw"
            ? "What was drawn before this discard"
            : "Which tile was discarded"
        }}
        at {{ time(item.t) }}?
      </p>
      <p>Choose a tile to save your answer.</p>
      <TilePalette
        :guess="item.tile"
        :disabled="busy"
        @pick="tileAnswer"
      /><button v-if="kind === 'draw'" :disabled="busy" @click="lost">
        Can't tell from the video
      </button></template
    >
    <TileSetAnswer
      v-else-if="['haipai', 'dora', 'ura', 'result'].includes(kind)"
      :item="item"
      :decode="decode"
      :kind="kind"
      :busy="busy"
      :answer="answer"
      :lost="lost"
    />
    <template
      v-else-if="
        kind === 'call' || (kind === 'kan' && item.seat && item.t != null)
      "
      ><p>{{ item.text }}</p>
      <MeldEditor
        :hand="item.hand"
        :seat="item.seat"
        :at="item.t"
        :type="kind === 'kan' ? 'ankan' : item.type || 'pon'"
        :tiles="
          item.tiles ||
          (item.tile ? Array(kind === 'kan' ? 4 : 3).fill(item.tile) : [])
        "
        :source="item.source"
      /><EvidenceDetails
        v-for="seat in kind === 'call' ? fromSeats : []"
        :key="seat"
        :region="`pond:${cornerOf(entry, seat)}`"
        :start="item.t - 15"
        :end="item.t + 2"
        :label="`${seatName(entry, seat)}'s pond before the call`"
      /><button
        v-if="item.source"
        :disabled="busy"
        @click="answer({ kind: 'note', text: item.text })"
      >
        It is right
      </button></template
    >
    <template v-else-if="kind === 'riichi' || kind === 'kan'"
      ><p>{{ item.text }}</p>
      <h3>
        {{
          kind === "riichi"
            ? "Choose the discard that declared riichi"
            : "Choose the discard immediately after the kan"
        }}
      </h3>
      <div
        v-for="turn in (decode.turns || []).filter(
          (turn) => kind !== 'riichi' || turn.seat === item.seat,
        )"
        :key="turn.i"
      >
        <button
          :disabled="busy"
          @click="
            answer({
              kind: kind === 'riichi' ? 'riichi_turn' : 'kan_time',
              seat: item.seat,
              t: turn.t,
            })
          "
        >
          {{ seatName(entry, turn.seat) }} {{ turn.discard }}
          {{ time(turn.t) }}</button
        ><EvidenceDetails
          :region="`pond:${cornerOf(entry, turn.seat)}`"
          :start="turn.t - 6"
          :end="turn.t + 4"
          label="Watch the discard"
        />
      </div>
      <button
        v-if="kind === 'riichi'"
        :disabled="busy"
        @click="
          answer({
            kind: 'riichi',
            seats: (decode.riichi || []).filter((seat) => seat !== item.seat),
          })
        "
      >
        {{ seatName(entry, item.seat) }} did not declare riichi
      </button></template
    >
    <template v-else-if="kind === 'conflict'"
      ><ConflictEditor :item="item" :entry="entry" :decode="decode" /><button
        v-if="item.stage !== 'export'"
        :disabled="busy"
        @click="answer({ kind: 'note', text: item.text })"
      >
        Acknowledge (leave as conflict)
      </button></template
    >
    <template v-else-if="kind === 'order'"
      ><p>{{ item.text }}</p>
      <table>
        <thead>
          <tr>
            <th>Turn</th>
            <th>Seat</th>
            <th>Time</th>
            <th>Discard</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="turn in (decode.turns || []).slice(
              Math.max(0, (item.i || 0) - 3),
              (item.i || 0) + 4,
            )"
            :key="turn.i"
          >
            <td>{{ turn.i + 1 }}</td>
            <td>{{ seatName(entry, turn.seat) }}</td>
            <td>{{ time(turn.t) }}</td>
            <td>{{ turn.discard }}</td>
          </tr>
        </tbody>
      </table>
      <h3>Add a missed discard</h3>
      <label
        >Seat<select v-model="missingSeat">
          <option v-for="seat in seats" :key="seat" :value="seat">
            {{ seatName(entry, seat) }}
          </option>
        </select></label
      ><label
        >Time<input
          v-model.number="missingTime"
          type="number"
          min="0"
          step="any" /></label
      ><TilePalette
        :disabled="busy"
        @pick="
          Number.isFinite(missingTime) && missingTime >= 0
            ? answer({
                kind: 'missing_discard',
                seat: missingSeat,
                t: missingTime,
                tile: $event,
              })
            : (error = 'Enter a valid time.')
        "
      /><button @click="answer({ kind: 'note', text: item.text })">
        Acknowledge (no change)
      </button></template
    >
    <template v-else
      ><p>{{ item.text }}</p>
      <button
        :disabled="busy"
        @click="answer({ kind: 'note', text: item.text })"
      >
        OK
      </button></template
    >
    <div class="actions">
      <button :disabled="busy" @click="review.next">Skip for now</button>
    </div>
  </div>
</template>
