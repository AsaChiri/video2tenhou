<script setup lang="ts">
import { computed, ref } from "vue";
import type { Decode, FactBody, HandEntry, ReviewItem } from "../types";
import type { PropType } from "vue";
import { useReview } from "./context";
import { cornerOf, parseTime, seats, seatName, time } from "./format";
import { confirmedMeld } from "./meld";
import { useAction } from "../shared/useAction";
import TilePalette from "./TilePalette.vue";
import TileSetAnswer from "./TileSetAnswer.vue";
import EvidenceDetails from "./EvidenceDetails.vue";
import MeldEditor from "./MeldEditor.vue";
import ConflictEditor from "./ConflictEditor.vue";
const props = defineProps({
  item: {
    type: Object as PropType<ReviewItem & { t: number }>,
    required: true,
  },
  entry: { type: Object as PropType<HandEntry>, required: true },
  decode: { type: Object as PropType<Decode>, default: () => ({}) },
});
const review = useReview(),
  { busy, error, run } = useAction();
const missingSeat = ref(props.item.seat || "E"),
  missingTime = ref(time(props.item.t));
// A conflict whose culprit is one meld, result or riichi is answered there.
const kind = computed(() => {
  const { kind, culprit } = props.item;
  if (kind !== "conflict" || !culprit) return kind;
  return culprit === "meld"
    ? "call"
    : ["result", "riichi"].includes(culprit)
      ? culprit
      : kind;
});
const fromSeats = computed(() =>
  props.item.source
    ? [
        seats[
          (seats.indexOf(props.item.seat || "") +
            ({ kamicha: 3, toimen: 2, shimocha: 1 } as Record<string, number>)[
              props.item.source
            ]) %
            4
        ],
      ]
    : seats.filter((seat) => seat !== props.item.seat),
);
async function answer(body: FactBody) {
  const saved = await run(() =>
    review.save({ ...body, hand: props.item.hand }),
  );
  return saved !== undefined;
}
const dismiss = () => run(() => review.dismiss(props.item));
/** Save the shown call as a meld fact, so the update keeps it. */
const confirmMeld = () =>
  run(() => {
    const { hand, seat, t, type, tiles, source } = props.item;
    return review.save({
      ...confirmedMeld({
        seat,
        t,
        type: type || "",
        tiles: tiles || [],
        source,
      }),
      hand,
    });
  });
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
function missedDiscard(tile: string) {
  const t = parseTime(missingTime.value);
  if (t === null) {
    error.value = "Enter the time as m:ss.";
    return;
  }
  void answer({ kind: "missing_discard", seat: missingSeat.value, t, tile });
}
</script>
<template>
  <div @input="review.dirty.value = true" @change="review.dirty.value = true">
    <p v-if="error" role="alert" class="warn">{{ error }}</p>
    <template v-if="kind === 'draw' || kind === 'discard'"
      ><p>
        {{
          kind === "draw"
            ? "Which tile was drawn before the discard"
            : "Which tile was discarded"
        }}
        at {{ time(item.t) }}?
      </p>
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
      <button
        v-if="item.source && item.type && item.tiles?.length"
        :disabled="busy"
        @click="confirmMeld"
      >
        It is right
      </button>
      <MeldEditor
        :hand="item.hand"
        :seat="item.seat"
        :at="item.t"
        :type="kind === 'kan' ? 'ankan' : item.type || 'pon'"
        :tiles="
          item.tiles ||
          (item.tile ? Array(kind === 'kan' ? 4 : 3).fill(item.tile) : [])
        "
        :source="item.source || undefined" /><EvidenceDetails
        v-for="seat in kind === 'call' ? fromSeats : []"
        :key="seat"
        :region="`pond:${cornerOf(entry, seat)}`"
        :start="item.t - 15"
        :end="item.t + 2"
        :label="`${seatName(entry, seat)}'s pond before the call`"
    /></template>
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
        @click="dismiss"
      >
        Leave as conflict
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
      <h3>Add the missed discard</h3>
      <label
        >Seat<select v-model="missingSeat">
          <option v-for="seat in seats" :key="seat" :value="seat">
            {{ seatName(entry, seat) }}
          </option>
        </select></label
      ><label>Time<input v-model="missingTime" placeholder="m:ss" /></label
      ><TilePalette :disabled="busy" @pick="missedDiscard" /><button
        :disabled="busy"
        @click="dismiss"
      >
        Nothing is missing
      </button></template
    >
    <template v-else
      ><p>{{ item.text }}</p>
      <button :disabled="busy" @click="dismiss">Dismiss</button></template
    >
    <div class="actions">
      <button :disabled="busy" @click="review.next">Skip for now</button>
    </div>
  </div>
</template>
