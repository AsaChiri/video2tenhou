<script setup lang="ts">
import { ref, watch } from "vue";
import type { HandData, Turn } from "../types";
import { useReview } from "./context";
import {
  cornerOf,
  handStatusLabel,
  roundName,
  seats,
  seatName,
  time,
} from "./format";
import { drawWindow } from "./evidence";
import { errorText } from "../shared/useAction";
import TileFace from "./TileFace.vue";
import TileListEditor from "./TileListEditor.vue";
import EvidencePlayer from "./EvidencePlayer.vue";
import EvidenceDetails from "./EvidenceDetails.vue";
import CameraContext from "./CameraContext.vue";
import FactsPanel from "./FactsPanel.vue";
const props = defineProps<{ hand?: number | null }>();
const emit = defineEmits(["select"]);
const review = useReview(),
  data = ref<HandData | null>(null),
  selectedTurn = ref<Turn | null>(null),
  editing = ref<string | null>(null),
  tiles = ref<string[]>([]);
watch(
  [() => props.hand, review.loads],
  async ([hand], previous, cleanup) => {
    let current = true;
    cleanup(() => (current = false));
    if (hand !== previous?.[0]) {
      data.value = null;
      selectedTurn.value = null;
      editing.value = null;
    }
    if (hand == null) return;
    try {
      const result = await review.api<HandData>(`hand/${hand}`);
      if (current) data.value = result;
    } catch (error) {
      if (current) review.error.value = errorText(error);
    }
  },
  { immediate: true },
);
function needsReview(turn: Turn) {
  return review.items.value.some(
    (item) =>
      item.hand === props.hand &&
      ((item.kind === "draw" && item.seat === turn.seat && item.j === turn.j) ||
        (item.kind === "uncertain_tiles" &&
          item.choices?.some(
            (choice) =>
              choice.field === "draw" &&
              choice.seat === turn.seat &&
              choice.j === turn.j,
          ))),
  );
}
async function save() {
  if (props.hand == null || editing.value == null) return;
  try {
    await review.save({
      kind: "haipai",
      hand: props.hand,
      seat: editing.value,
      tiles: tiles.value,
    });
    editing.value = null;
  } catch (error) {
    review.error.value = errorText(error);
  }
}
</script>
<template>
  <section v-if="hand == null">
    <table>
      <thead>
        <tr>
          <th>Hand</th>
          <th>Hanchan</th>
          <th>Round</th>
          <th>Video time</th>
          <th>Status</th>
          <th>Turns</th>
          <th>Winning hand vs site</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="entry in review.hands.value" :key="entry.hand">
          <td>{{ entry.hand + 1 }}</td>
          <td>{{ entry.game + 1 }}</td>
          <td>
            <button @click="emit('select', entry.hand)">
              {{ roundName(entry) }}
            </button>
          </td>
          <td>{{ time(entry.t_start) }}–{{ time(entry.t_end) }}</td>
          <td>{{ handStatusLabel(entry.status, entry.pending) }}</td>
          <td>{{ entry.turns }}</td>
          <td>
            {{ entry.score == null ? "" : entry.score ? "Matches" : "Differs" }}
          </td>
        </tr>
      </tbody>
    </table>
  </section>
  <section v-else>
    <button @click="emit('select', null)">All hands</button>
    <p v-if="!data" role="status">Loading hand…</p>
    <template v-else
      ><h2>
        Hand {{ hand + 1 }} · Hanchan {{ data.entry.game + 1 }} ·
        {{ roundName(data.entry) }}
      </h2>
      <p v-if="!data.decode">
        {{
          review.pending.value.includes(hand)
            ? "This hand is waiting for an update."
            : "This hand has not been analyzed yet."
        }}
      </p>
      <template v-else
        ><ul v-if="data.decode.notes?.length" class="notes">
          <li v-for="note in data.decode.notes" :key="note">{{ note }}</li>
        </ul>
        <p>
          Dealer {{ seatName(data.entry, data.decode.dealer) }} · Dora
          {{ data.decode.dora?.join(" ") }} · {{ data.decode.result?.outcome }}
          {{
            data.decode.result?.winner
              ? seatName(data.entry, data.decode.result.winner)
              : ""
          }}
        </p>
        <div class="actions">
          <button
            v-if="!review.pending.value.length"
            :disabled="review.busy.value"
            @click="review.rebuild([hand])"
          >
            Rebuild hand {{ hand + 1 }}
          </button>
        </div>
        <div v-if="selectedTurn" class="card">
          <h3>
            Turn {{ selectedTurn.i + 1 }}:
            {{ seatName(data.entry, selectedTurn.seat) }} discards
            {{ selectedTurn.discard }} at {{ time(selectedTurn.t) }}
          </h3>
          <EvidencePlayer
            :key="selectedTurn.i"
            :start="drawWindow(selectedTurn, data.entry, data.decode)[0]"
            :end="drawWindow(selectedTurn, data.entry, data.decode)[1]"
          />
          <details class="review-details">
            <summary>Additional evidence</summary>
            <EvidenceDetails
              :region="`hand:${cornerOf(data.entry, selectedTurn.seat)}`"
              :start="selectedTurn.t - 15"
              :end="selectedTurn.t + 4"
              label="Hand camera"
            /><EvidenceDetails
              :region="`pond:${cornerOf(data.entry, selectedTurn.seat)}`"
              :at="selectedTurn.t + 2"
              label="Pond after discard"
            /><CameraContext
              :hand="hand"
              :seat="selectedTurn.seat"
              :at="selectedTurn.t"
            />
          </details>
        </div>
        <h3>Starting hands</h3>
        <div v-for="seat in seats" :key="seat">
          <b>{{ seatName(data.entry, seat) }}</b
          ><span
            v-for="(tile, index) in data.decode.haipai?.[seat] || []"
            :key="index"
            class="tile"
            ><TileFace :tile="tile" /></span
          ><button
            @click="
              editing = seat;
              tiles = [...(data.decode.haipai?.[seat] || [])];
            "
          >
            Edit</button
          ><template v-if="editing === seat"
            ><TileListEditor
              v-model="tiles"
              @update:model-value="review.dirty.value = true"
            /><button @click="save">Save starting hand</button></template
          >
        </div>
        <h3>Turns</h3>
        <table>
          <thead>
            <tr>
              <th>Turn</th>
              <th>Seat</th>
              <th>Time</th>
              <th>Draw</th>
              <th>Review</th>
              <th>Discard</th>
              <th>Call</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="turn in data.decode.turns" :key="turn.i">
              <td>
                <button @click="selectedTurn = turn">{{ turn.i + 1 }}</button>
              </td>
              <td>{{ seatName(data.entry, turn.seat) }}</td>
              <td>{{ time(turn.t) }}</td>
              <td>{{ turn.draw }}</td>
              <td>{{ needsReview(turn) ? "Needs review" : "—" }}</td>
              <td>
                {{ turn.discard }} {{ turn.riichi ? "RIICHI" : "" }}
                {{ turn.tsumogiri ? "(tsumogiri)" : "" }}
              </td>
              <td>
                {{
                  turn.call
                    ? `${turn.call.type} ${turn.call.tiles.join(" ")} by ${seatName(data.entry, turn.call.seat)}`
                    : ""
                }}
              </td>
            </tr>
          </tbody>
        </table>
        <details>
          <summary>All decisions</summary>
          <table>
            <thead>
              <tr>
                <th>Choice</th>
                <th>Seat</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(choice, index) in data.decode.confidence"
                :key="index"
              >
                <td>
                  {{ choice.field }}
                  {{
                    choice.turn != null && choice.turn >= 0
                      ? choice.turn + 1
                      : ""
                  }}
                </td>
                <td>{{ seatName(data.entry, choice.seat) }}</td>
                <td>
                  {{
                    Array.isArray(choice.value)
                      ? choice.value.join(" ")
                      : choice.value
                  }}
                </td>
              </tr>
            </tbody>
          </table>
        </details>
        <FactsPanel
          :hand="hand"
          :entry="data.entry"
          :ignored="data.ignored"
        /> </template
    ></template>
  </section>
</template>
