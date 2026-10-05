<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { api } from "../shared/api";
import { errorText } from "../shared/useAction";
import type { Project, Results } from "../types";
// `updating`: this project's hands are being updated with saved answers.
const props = defineProps<{ project: Project; updating?: boolean }>();
const emit = defineEmits(["error"]);
const data = ref<Results | null>(null),
  active = ref(0),
  copied = ref(""),
  loading = ref(false);
const game = computed(
  () =>
    data.value?.games.find((game) => game.index === active.value) ||
    data.value?.games[0],
);
const pending = computed(() =>
  data.value?.pending_games.includes(game.value?.index ?? -1),
);
watch(
  () =>
    JSON.stringify([
      props.project.id,
      props.project.results_revision,
      props.updating,
    ]),
  async (_, __, onCleanup) => {
    let current = true;
    onCleanup(() => (current = false));
    loading.value = true;
    try {
      const result = await api<Results>(
        `/api/projects/${props.project.id}/results`,
      );
      if (current) data.value = result;
    } catch (error) {
      if (current) emit("error", errorText(error));
    } finally {
      if (current) loading.value = false;
    }
  },
  { immediate: true },
);
async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    copied.value = "Copied";
  } catch {
    copied.value = "Copy unavailable";
  }
}
</script>
<template>
  <p v-if="loading && !data" class="empty" role="status">Loading results…</p>
  <p v-else-if="!game" class="empty">No game logs yet.</p>
  <div v-else class="results-layout">
    <aside class="games" aria-label="Games">
      <button
        v-for="entry in data?.games || []"
        :key="entry.index"
        class="game-button"
        :class="{ active: entry.index === game.index }"
        @click="active = entry.index"
      >
        Hanchan {{ entry.index + 1
        }}<small>{{ entry.hands.length }} hands</small>
      </button>
    </aside>
    <section>
      <div class="result-head">
        <div>
          <h2>Hanchan {{ game.index + 1 }}</h2>
          <p class="players">{{ game.names.join(" · ") }}</p>
        </div>
        <div v-if="!pending" class="actions">
          <a
            class="button primary"
            :href="game.viewer_url"
            target="_blank"
            rel="noopener"
            >Open replay</a
          ><a class="button" :href="game.download" download>Download JSON</a>
        </div>
      </div>
      <p v-if="pending" role="status">
        {{
          updating
            ? "Updating this hanchan with your answers…"
            : "Your latest answers are not in this record yet. Open Review to apply them."
        }}
      </p>
      <template v-else>
        <div class="game-actions">
          <span>Hands</span><span role="status">{{ copied }}</span
          ><button
            @click="copy(game.hands.map((hand) => hand.editor_url).join('\n'))"
          >
            Copy all hand links
          </button>
        </div>
        <table class="hand-table">
          <thead>
            <tr>
              <th>Round</th>
              <th>Honba</th>
              <th>Hand link</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="hand in game.hands" :key="hand.index">
              <td>
                <a :href="hand.editor_url" target="_blank" rel="noopener">{{
                  hand.round
                }}</a>
              </td>
              <td>{{ hand.honba }}</td>
              <td>
                <button class="small" @click="copy(hand.editor_url)">
                  Copy link
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </template>
      <details class="result-details">
        <summary>Reports</summary>
        <p>
          <a
            v-for="file in project.artifacts.filter(
              (file) =>
                file === 'report.md' ||
                file === 'review.json' ||
                file.endsWith('.confidence.json'),
            )"
            :key="file"
            :href="`/exports/${project.id}/${file}`"
            download
            >{{ file }}</a
          >
        </p>
      </details>
    </section>
  </div>
</template>
