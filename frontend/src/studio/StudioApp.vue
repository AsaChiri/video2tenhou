<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, provide, ref } from "vue";
import type { Project, ProjectJob, Workspace } from "../types";
import { api } from "../shared/api";
import { errorText, useAction } from "../shared/useAction";
import { jobKey, useJobStatus } from "../shared/useJob";
import { time } from "../review/format";
import RecordingForm from "./RecordingForm.vue";
import ProjectSettings from "./ProjectSettings.vue";
import ProjectLibrary from "./ProjectLibrary.vue";
import ProcessingLog from "./ProcessingLog.vue";
import ResultsPanel from "./ResultsPanel.vue";
import ReviewWorkspace from "../review/ReviewWorkspace.vue";
import "../review/review.css";

const CHECK_AGAIN = 1000;
const projects = ref<Project[]>([]),
  selected = ref<string | null>(null),
  view = ref("analyze"),
  error = ref(""),
  missing = ref<string[]>([]);
const calibrationDirty = ref(false),
  reviewVersion = ref(0);
const loading = ref(true),
  loaded = ref(false),
  loadError = ref(""),
  showIntake = ref(false);
const status = useJobStatus(selected);
provide(jobKey, status);
const { busy, run } = useAction((message) => (error.value = message));
let revision = 0,
  checkTimer: ReturnType<typeof setTimeout> | undefined;
const project = computed(() =>
  projects.value.find((project) => project.id === selected.value),
);
/** The selected project's preparation or analysis, live while it runs. */
const projectJob = computed<ProjectJob | null>(() => {
  const live = status.job.value;
  if (
    live?.running &&
    live.project === selected.value &&
    (live.kind === "prepare" || live.kind === "analyze")
  )
    return { action: live.kind, ...live };
  return project.value?.job ?? null;
});
const processing = computed(() => !!projectJob.value?.running);
const hasResults = computed(() =>
  project.value?.artifacts.some((file) => /^g\d+\.json$/.test(file)),
);
const tabs = computed(() =>
  processing.value
    ? [
        ["analyze", "Analysis"],
        ...(hasResults.value ? [["results", "Results"]] : []),
        ["settings", "Settings"],
      ]
    : [
        ...(hasResults.value
          ? [
              ["review", "Review"],
              ["results", "Results"],
            ]
          : [["analyze", "Analyze"]]),
        ["settings", "Settings"],
      ],
);
const showReview = computed(
  () =>
    project.value &&
    (view.value === "review" ||
      view.value === "calibration" ||
      (view.value === "analyze" && project.value.has_fit && !processing.value)),
);
const elapsed = computed(() => {
  const job = projectJob.value;
  if (!job?.started) return "";
  return time((job.finished || status.now.value / 1000) - job.started);
});
function blocked() {
  if (!calibrationDirty.value) return false;
  error.value = "Save or discard your calibration changes before continuing.";
  return true;
}
function open(id: string | null) {
  if (blocked()) return;
  selected.value = id || null;
  showIntake.value = false;
  view.value = hasResults.value ? "review" : "analyze";
  if (location.hash.slice(1) !== (id || "")) location.hash = id || "";
  error.value = "";
  if (!id && loaded.value) void refresh();
}
function newProject() {
  if (blocked()) return;
  open(null);
  showIntake.value = true;
}
function openSettings(id: string) {
  if (blocked()) return;
  open(id);
  view.value = "settings";
}
function hashChanged() {
  const id = location.hash.slice(1);
  if (id === (selected.value || "")) return;
  if (blocked()) {
    history.replaceState(
      null,
      "",
      `${location.pathname}${location.search}${selected.value ? "#" + selected.value : ""}`,
    );
    return;
  }
  if (!id || projects.value.some((entry) => entry.id === id)) open(id || null);
  else {
    open(null);
    error.value =
      "This project is no longer available. Choose a project below.";
  }
}
onMounted(() => window.addEventListener("hashchange", hashChanged));
onBeforeUnmount(() => {
  window.removeEventListener("hashchange", hashChanged);
  clearTimeout(checkTimer);
});
function changeView(next: string) {
  if (!blocked()) view.value = next;
}
function update(updated: Project) {
  revision++;
  projects.value = projects.value.map((project) =>
    project.id === updated.id ? updated : project,
  );
}
async function action(kind: string) {
  if (blocked() || !project.value) return;
  const id = selected.value;
  error.value = "";
  await run(async () => {
    try {
      update(await api<Project>(`/api/projects/${id}/${kind}`, {}));
      if (selected.value === id) view.value = "analyze";
    } finally {
      void status.refresh();
    }
  });
  if (selected.value !== id) error.value = "";
}
async function created(project: Project) {
  revision++;
  projects.value = [
    ...projects.value.filter((entry) => entry.id !== project.id),
    project,
  ];
  open(project.id);
  if (!project.has_fit && !project.checking) await action("prepare");
}
function deleted(id: string) {
  revision++;
  projects.value = projects.value.filter((entry) => entry.id !== id);
  if (selected.value === id) open(null);
}
function saved(project: Project) {
  update(project);
  if (project.id === selected.value) reviewVersion.value++;
}
let first = true;
async function refresh() {
  const requestedRevision = revision;
  clearTimeout(checkTimer);
  try {
    const data = await api<Workspace>("/api/workspace");
    if (requestedRevision !== revision) return;
    projects.value = data.projects;
    loaded.value = true;
    loadError.value = "";
    missing.value = data.setup.ready ? [] : data.setup.missing;
    if (first) {
      first = false;
      const id = location.hash.slice(1);
      if (projects.value.some((project) => project.id === id)) open(id);
      else if (id) {
        open(null);
        error.value =
          "This project is no longer available. Choose a project below.";
      }
    }
    if (selected.value && !project.value) {
      calibrationDirty.value = false;
      open(null);
      error.value = "This project was removed. Choose another project below.";
    }
    if (project.value && !calibrationDirty.value) {
      if (view.value === "analyze" && hasResults.value && !processing.value)
        view.value = "review";
      if (!hasResults.value && ["review", "results"].includes(view.value))
        view.value = "analyze";
    }
    if (data.projects.some((project) => project.checking))
      checkTimer = setTimeout(refresh, CHECK_AGAIN);
  } catch (failure) {
    loadError.value = errorText(failure);
  } finally {
    loading.value = false;
  }
}
status.onFinished(() => void refresh());
onMounted(refresh);
</script>
<template>
  <header>
    <div class="topbar">
      <a class="brand" href="/" @click.prevent="open(null)">video2tenhou</a>
      <div v-if="selected || showIntake" class="project-switch">
        <button v-if="selected || showIntake" @click="open(null)">
          All projects
        </button>
        <button v-if="selected" @click="newProject">New project</button>
      </div>
    </div>
  </header>
  <main>
    <div v-if="missing.length" class="notice">
      <strong>Missing setup files</strong>
      <ul>
        <li v-for="file in missing" :key="file">{{ file }}</li>
      </ul>
    </div>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <div v-if="loadError" class="notice error" role="alert">
      Could not load projects: {{ loadError }}
      <button @click="refresh">Retry</button>
    </div>
    <ProjectLibrary
      v-if="!selected && !showIntake"
      :projects="projects"
      :loading="loading"
      :unavailable="!loaded && !!loadError"
      :job="status.job.value"
      @open="open"
      @settings="openSettings"
      @create="newProject"
      @updated="update"
      @deleted="deleted"
    />
    <RecordingForm
      v-show="!selected && showIntake"
      @created="created"
      @error="error = $event"
    />
    <section v-if="project">
      <h1 class="current-project-name">{{ project.display_name }}</h1>
      <nav class="project-tabs" role="tablist" aria-label="Project">
        <button
          v-for="[key, label] in tabs"
          :key="key"
          role="tab"
          :aria-selected="view === key"
          @click="changeView(key)"
        >
          {{ label }}
        </button>
      </nav>
      <div
        v-if="projectJob?.error && !processing"
        class="notice error"
        role="alert"
      >
        {{ projectJob.error }}
        <button
          v-if="project.can_calibrate && view !== 'settings'"
          @click="changeView('settings')"
        >
          Open settings
        </button>
      </div>
      <ProjectSettings
        v-if="view === 'settings'"
        :key="project.id"
        :project="project"
        :locked="!!status.job.value?.running"
        @updated="saved"
        @error="error = $event"
        @calibrate="changeView('calibration')"
        @analyze="action('analyze')"
        @prepare="action('prepare')"
      />
      <ResultsPanel
        v-if="view === 'results'"
        :key="project.id"
        :project="project"
        :updating="
          !!status.job.value?.running &&
          status.job.value.project === project.id &&
          status.job.value.kind === 'rebuild'
        "
        @error="error = $event"
      />
      <button v-if="view === 'calibration'" @click="changeView('settings')">
        Back to settings
      </button>
      <section v-if="view === 'analyze'" class="processing">
        <p v-if="project.checking" role="status">Checking the recording…</p>
        <template v-else-if="processing"
          ><h2>{{ projectJob?.stage || "Analyzing recording" }}</h2>
          <span>Elapsed {{ elapsed }}</span
          ><progress class="progress" aria-label="Processing"
        /></template>
        <template v-else
          ><p v-if="project.stale_exports" class="notice">
            Recording inputs changed.
            {{
              project.has_fit
                ? "Analyze the recording"
                : "Prepare the recording again"
            }}
            to refresh the results.
          </p>
          <h2>
            {{ project.has_fit ? "Check calibration" : "Prepare recording" }}
          </h2>
          <button
            class="primary"
            :disabled="busy || !!status.job.value?.running"
            @click="action(project.has_fit ? 'analyze' : 'prepare')"
          >
            {{ project.has_fit ? "Analyze recording" : "Prepare recording" }}
          </button></template
        >
      </section>
      <ProcessingLog
        v-if="['settings', 'analyze'].includes(view) && projectJob?.started"
        :key="project.id"
        :project-id="project.id"
        :job="projectJob"
        :lines="status.job.value?.log_lines ?? 0"
      />
      <ReviewWorkspace
        v-if="
          project.has_fit || (project.can_calibrate && view === 'calibration')
        "
        v-show="showReview"
        :key="`${selected}:${reviewVersion}`"
        :active="showReview"
        :project-id="project.id"
        :calibration="['analyze', 'calibration'].includes(view)"
        @dirty="calibrationDirty = $event"
        @results="changeView('results')"
      />
    </section>
  </main>
</template>
