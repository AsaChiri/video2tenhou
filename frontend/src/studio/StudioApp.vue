<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from "vue";
import type { Project, Workspace } from "../types";
import { api } from "../shared/api";
import { usePolling } from "../shared/usePolling";
import RecordingForm from "./RecordingForm.vue";
import ProjectSettings from "./ProjectSettings.vue";
import ProjectLibrary from "./ProjectLibrary.vue";
import ProcessingLog from "./ProcessingLog.vue";
import ResultsPanel from "./ResultsPanel.vue";
import ReviewWorkspace from "../review/ReviewWorkspace.vue";
import "../review/review.css";

const projects = ref<Project[]>([]),
  selected = ref<string | null>(null),
  view = ref("analyze"),
  error = ref(""),
  missing = ref<string[]>([]),
  busy = ref(false);
const calibrationDirty = ref(false),
  reviewVersion = ref(0);
const loading = ref(true),
  loaded = ref(false),
  loadError = ref(""),
  showIntake = ref(false);
let revision = 0;
const project = computed(() =>
  projects.value.find((project) => project.id === selected.value),
);
const hasResults = computed(() =>
  project.value?.artifacts.some((file) => /^g\d+\.json$/.test(file)),
);
const tabs = computed(() =>
  project.value?.job.running
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
      (view.value === "analyze" &&
        project.value.has_fit &&
        !project.value.job.running)),
);
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
onBeforeUnmount(() => window.removeEventListener("hashchange", hashChanged));
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
  if (blocked() || busy.value || !project.value) return;
  const id = selected.value;
  busy.value = true;
  error.value = "";
  try {
    update(await api<Project>(`/api/projects/${id}/${kind}`, {}));
    if (selected.value === id) view.value = "analyze";
  } catch (failure) {
    if (selected.value === id)
      error.value =
        failure instanceof Error ? failure.message : String(failure);
  } finally {
    busy.value = false;
  }
}
async function created(project: Project) {
  revision++;
  projects.value = [
    ...projects.value.filter((entry) => entry.id !== project.id),
    project,
  ];
  open(project.id);
  if (!project.has_fit) await action("prepare");
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
      if (
        view.value === "analyze" &&
        hasResults.value &&
        !project.value.job.running
      )
        view.value = "review";
      if (!hasResults.value && ["review", "results"].includes(view.value))
        view.value = "analyze";
    }
  } catch (failure) {
    loadError.value =
      failure instanceof Error ? failure.message : String(failure);
  } finally {
    loading.value = false;
  }
}
usePolling(refresh, 2500);
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
      <h1 class="current-project-name">
        {{ project.display_name || project.name }}
      </h1>
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
      <div v-if="project.job.error" class="notice error" role="alert">
        {{ project.job.error }}
        <button
          v-if="project.can_calibrate && !project.job.running"
          @click="changeView('settings')"
        >
          Open settings
        </button>
      </div>
      <ProjectSettings
        v-if="view === 'settings'"
        :key="project.id"
        :project="project"
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
        @error="error = $event"
      />
      <button v-if="view === 'calibration'" @click="changeView('settings')">
        Back to settings
      </button>
      <section v-if="view === 'analyze'" class="processing">
        <template v-if="project.job.running"
          ><h2>{{ project.job.stage || "Analyzing recording" }}</h2>
          <span
            >{{
              Math.floor(
                ((project.job.finished || Date.now() / 1000) -
                  (project.job.started || 0)) /
                  60,
              )
            }}
            min elapsed</span
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
            :disabled="busy"
            @click="action(project.has_fit ? 'analyze' : 'prepare')"
          >
            {{ project.has_fit ? "Analyze recording" : "Prepare recording" }}
          </button></template
        >
      </section>
      <template v-for="entry in projects" :key="entry.id"
        ><ProcessingLog
          v-show="
            entry.id === selected &&
            ['settings', 'analyze'].includes(view) &&
            entry.job.log?.length
          "
          :job="entry.job"
      /></template>
      <ReviewWorkspace
        v-if="
          project.has_fit || (project.can_calibrate && view === 'calibration')
        "
        v-show="showReview"
        :active="showReview"
        :key="`${selected}:${reviewVersion}`"
        :project-id="project.id"
        :calibration="['analyze', 'calibration'].includes(view)"
        @dirty="calibrationDirty = $event"
        @results="changeView('results')"
      />
    </section>
  </main>
</template>
