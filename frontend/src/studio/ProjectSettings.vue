<script setup lang="ts">
import { ref } from "vue";
import { api } from "../shared/api";
import { useAction } from "../shared/useAction";
import { time } from "../review/format";
import type { Project } from "../types";
import { gameIds } from "./validation";
// `locked`: a job is running somewhere in the workspace.
const props = defineProps<{ project: Project; locked?: boolean }>();
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
  { busy, run } = useAction((message) => emit("error", message));
const save = () =>
  run(async () => {
    emit(
      "updated",
      await api(`/api/projects/${props.project.id}/settings`, {
        games: gameIds(games.value),
        layout: layout.value.trim(),
      }),
    );
    saved.value = "Saved";
  });
</script>
<template>
  <div class="panel settings">
    <section class="settings-section">
      <h2>Recording settings</h2>
      <p v-if="project.start || project.end != null" class="hint">
        Selected range: {{ time(project.start) }} to
        {{ project.end == null ? "end of recording" : time(project.end) }}.
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
        <button class="primary" :disabled="locked || busy">Save settings</button
        ><span role="status">{{ saved }}</span>
      </form>
    </section>
    <section class="settings-section">
      <h3>Processing</h3>
      <div class="actions">
        <button :disabled="locked || project.checking" @click="emit('prepare')">
          Prepare recording
        </button>
        <button
          :disabled="!project.can_calibrate || locked"
          @click="emit('calibrate')"
        >
          Calibration
        </button>
        <button :disabled="!project.has_fit || locked" @click="emit('analyze')">
          Analyze again
        </button>
      </div>
    </section>
  </div>
</template>
