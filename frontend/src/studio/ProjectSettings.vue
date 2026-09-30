<script setup lang="ts">
import { ref } from "vue";
import { api } from "../shared/api";
import type { Project } from "../types";
import { gameIds } from "./validation";
const props = defineProps<{ project: Project }>();
const emit = defineEmits([
  "updated",
  "error",
  "calibrate",
  "analyze",
  "prepare",
]);
const games = ref(props.project.games.join(", ")),
  layout = ref(props.project.layout),
  saved = ref(""),
  busy = ref(false);
async function save() {
  if (busy.value) return;
  busy.value = true;
  try {
    emit(
      "updated",
      await api(`/api/projects/${props.project.id}/settings`, {
        games: gameIds(games.value),
        layout: layout.value.trim(),
      }),
    );
    saved.value = "Saved";
  } catch (error) {
    emit("error", error instanceof Error ? error.message : String(error));
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <div class="panel settings">
    <section class="settings-section">
      <h2>Recording settings</h2>
      <p v-if="project.start || project.end != null" class="hint">
        Selected range: {{ project.start || 0 }}s to
        {{ project.end == null ? "end of recording" : project.end + "s" }}.
        Review times start at zero. Add a recording to use a different range.
      </p>
      <form @submit.prevent="save">
        <div class="field">
          <label for="settings-games">Scoremj games</label
          ><input id="settings-games" v-model="games" required />
        </div>
        <div class="field">
          <label for="settings-layout">Layout</label
          ><input id="settings-layout" v-model="layout" />
        </div>
        <button class="primary" :disabled="project.job.running || busy">
          Save settings</button
        ><span role="status">{{ saved }}</span>
      </form>
    </section>
    <section class="settings-section">
      <h3>Processing</h3>
      <div class="actions">
        <button :disabled="project.job.running" @click="emit('prepare')">
          Prepare recording
        </button>
        <button
          :disabled="
            !(project.can_calibrate ?? project.has_fit) || project.job.running
          "
          @click="emit('calibrate')"
        >
          Calibration
        </button>
        <button
          :disabled="!project.has_fit || project.job.running"
          @click="emit('analyze')"
        >
          Analyze again
        </button>
      </div>
    </section>
  </div>
</template>
