<script setup lang="ts">
import { ref } from "vue";
import { api, upload } from "../shared/api";
import { useAction } from "../shared/useAction";
import { gameIds, timeRange } from "./validation";

const emit = defineEmits(["created", "error"]);
const kind = ref("local"),
  name = ref(""),
  file = ref<File | null>(null),
  url = ref(""),
  games = ref(""),
  layout = ref("");
const start = ref(""),
  end = ref(""),
  progress = ref<number | null>(null),
  { busy, run } = useAction((message) => emit("error", message));
function submit() {
  emit("error", "");
  return run(async () => {
    try {
      // Capture the form before copying a large file; changing tabs cannot
      // change this import.
      const body = {
        ...(name.value.trim() ? { display_name: name.value.trim() } : {}),
        kind: kind.value,
        games: gameIds(games.value),
        layout: layout.value.trim(),
        ...timeRange(start.value.trim(), end.value.trim()),
      };
      let source = url.value.trim();
      if (body.kind === "local") {
        if (!file.value) throw new Error("Choose a video file.");
        progress.value = 0;
        source = await upload(file.value, (value) => (progress.value = value));
      }
      if (!source) throw new Error("Enter a remote video URL.");
      emit("created", await api("/api/projects", { ...body, source }));
    } finally {
      progress.value = null;
    }
  });
}
</script>

<template>
  <section class="panel intake">
    <h1>New project</h1>
    <form @submit.prevent="submit">
      <div class="field">
        <label for="project-name">Project name (optional)</label>
        <input
          id="project-name"
          v-model="name"
          maxlength="120"
          placeholder="Use the recording name"
        />
      </div>
      <div class="source-tabs">
        <button
          v-for="source in ['local', 'url']"
          :key="source"
          type="button"
          :aria-pressed="kind === source"
          @click="kind = source"
        >
          {{ source === "local" ? "Video file" : "Video URL" }}
        </button>
      </div>
      <div v-show="kind === 'local'" class="field">
        <label for="upload-source">Video file</label>
        <input
          id="upload-source"
          type="file"
          accept=".mp4,.mkv,.mov,.webm,.avi,.m4v"
          :disabled="kind !== 'local'"
          @change="
            file = ($event.target as HTMLInputElement).files?.[0] ?? null
          "
        />
        <p class="hint">The recording is copied into your local workspace.</p>
      </div>
      <div v-show="kind === 'url'" class="field">
        <label for="url-source">Video URL</label
        ><input
          id="url-source"
          v-model="url"
          type="url"
          :disabled="kind !== 'url'"
          placeholder="https://…"
        />
      </div>
      <div class="time-range">
        <div class="field">
          <label for="start-time">Start time (optional)</label
          ><input
            id="start-time"
            v-model="start"
            placeholder="Beginning of recording"
          />
        </div>
        <div class="field">
          <label for="end-time">End time (optional)</label
          ><input id="end-time" v-model="end" placeholder="End of recording" />
        </div>
      </div>
      <p class="hint time-range-help">
        Use HH:MM:SS, MM:SS or seconds. Leave blank for the full recording.
        Include complete games; review times start at zero in the selected
        range.
      </p>
      <div class="field">
        <label for="games">Scoremj games</label
        ><input
          id="games"
          v-model="games"
          required
          placeholder="21938, 21939"
        />
        <p class="hint">Game IDs or links, in broadcast order.</p>
      </div>
      <button class="primary" :disabled="busy">Prepare recording</button>
      <p v-if="progress !== null" role="status">
        Copying recording… {{ Math.round(progress) }}%
      </p>
      <progress
        v-if="progress !== null"
        class="progress"
        :value="progress"
        max="100"
        aria-label="Uploading recording"
      />
      <details>
        <summary>Layout</summary>
        <label for="layout">Layout file</label
        ><input id="layout" v-model="layout" placeholder="PML (default)" />
      </details>
    </form>
  </section>
</template>
