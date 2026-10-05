<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import type { Project, WorkspaceJob } from "../types";
import { api } from "../shared/api";
import { useAction } from "../shared/useAction";

const props = defineProps<{
  projects: Project[];
  loading: boolean;
  unavailable?: boolean;
  job?: WorkspaceJob | null;
}>();
const emit = defineEmits<{
  open: [id: string];
  settings: [id: string];
  create: [];
  updated: [project: Project];
  deleted: [id: string];
}>();
const query = ref("");
const editing = ref<string | null>(null);
const removing = ref<string | null>(null);
const name = ref("");
const { busy, error, run } = useAction();
const nameInput = ref<HTMLInputElement[] | null>(null);
const running = (project: Project) =>
  !!props.job?.running && props.job.project === project.id;
const visible = computed(() => {
  const search = query.value.trim().toLocaleLowerCase();
  return [...props.projects]
    .sort((a, b) => (b.created ?? 0) - (a.created ?? 0))
    .filter((project) =>
      [project.display_name || project.name, project.source, ...project.games]
        .join(" ")
        .toLocaleLowerCase()
        .includes(search),
    );
});
function title(project: Project) {
  return project.display_name || project.name;
}
function status(project: Project) {
  if (running(project)) return props.job?.stage || "Processing";
  if (project.checking) return "Checking recording";
  if (project.job.error) return "Needs attention";
  if (project.stale_exports) return "Analysis needed";
  if (project.artifacts.some((file) => /^g\d+\.json$/.test(file)))
    return project.open_items
      ? `${project.open_items} review items`
      : "Results ready";
  return project.has_fit ? "Ready to analyze" : "Preparation needed";
}
async function edit(project: Project) {
  removing.value = null;
  editing.value = project.id;
  name.value = title(project);
  error.value = "";
  await nextTick();
  nameInput.value?.[0]?.focus();
  nameInput.value?.[0]?.select();
}
function confirmDelete(project: Project) {
  editing.value = null;
  removing.value = project.id;
  error.value = "";
}
const save = (project: Project) =>
  run(async () => {
    emit(
      "updated",
      await api<Project>(`/api/projects/${project.id}/rename`, {
        display_name: name.value.trim(),
      }),
    );
    editing.value = null;
  });
const remove = (project: Project) =>
  run(async () => {
    await api(`/api/projects/${project.id}/delete`, {});
    emit("deleted", project.id);
    removing.value = null;
  });
</script>

<template>
  <section
    class="project-library"
    aria-labelledby="projects-heading"
    :aria-busy="loading"
  >
    <div class="library-heading">
      <h1 id="projects-heading">Projects</h1>
      <button class="primary" @click="emit('create')">New project</button>
    </div>
    <p v-if="loading" role="status">Loading projects…</p>
    <p v-else-if="unavailable">
      Your project list is unavailable. Use Retry above to load it again.
    </p>
    <template v-else>
      <div v-if="projects.length" class="library-tools">
        <div>
          <label for="project-search">Find a project</label>
          <input
            id="project-search"
            v-model="query"
            type="search"
            placeholder="Search by name, recording or game ID"
          />
        </div>
        <span class="muted" role="status"
          >{{ visible.length }} of {{ projects.length }} projects</span
        >
      </div>
      <div v-if="!projects.length" class="library-empty">
        <h2>Your recordings start here</h2>
        <p>
          Create a project from a video file or URL. Your saved projects will
          appear here when you return.
        </p>
      </div>
      <p v-else-if="!visible.length" class="empty">
        No projects match “{{ query }}”. Try another name or game ID.
      </p>
      <ul v-else class="project-list">
        <li v-for="project in visible" :key="project.id" class="project-row">
          <div class="project-summary">
            <h2>
              <a
                :href="`#${project.id}`"
                @click.prevent="emit('open', project.id)"
                >{{ title(project) }}</a
              >
            </h2>
            <p class="project-source" :title="project.source">
              {{ project.source }}
            </p>
            <p class="hint">
              Games {{ project.games.join(", ")
              }}<template v-if="project.created">
                · Added
                {{
                  new Date(project.created * 1000).toLocaleDateString()
                }}</template
              >
            </p>
          </div>
          <span
            class="project-status"
            :class="{ 'status-error': project.job.error }"
            >{{ status(project) }}</span
          >
          <div class="actions project-actions">
            <button :disabled="busy" @click="emit('open', project.id)">
              Open project
            </button>
            <button :disabled="busy" @click="emit('settings', project.id)">
              Settings
            </button>
            <button :disabled="busy" @click="edit(project)">Rename</button>
            <button
              class="danger"
              :disabled="busy || running(project)"
              :title="
                running(project)
                  ? 'Wait for this project’s job to finish.'
                  : undefined
              "
              @click="confirmDelete(project)"
            >
              Delete
            </button>
          </div>
          <form
            v-if="editing === project.id"
            class="project-edit"
            @submit.prevent="save(project)"
          >
            <label :for="`name-${project.id}`">Project name</label>
            <div class="actions">
              <input
                :id="`name-${project.id}`"
                ref="nameInput"
                v-model="name"
                required
                maxlength="120"
                :disabled="busy"
              />
              <button class="primary" :disabled="busy || !name.trim()">
                {{ busy ? "Saving…" : "Save name" }}
              </button>
              <button
                type="button"
                :disabled="busy"
                @click="
                  editing = null;
                  error = '';
                "
              >
                Cancel
              </button>
            </div>
          </form>
          <div
            v-if="removing === project.id"
            class="project-edit delete-confirm"
            role="group"
            :aria-label="`Delete ${title(project)}`"
          >
            <h3>Delete “{{ title(project) }}”?</h3>
            <p>
              This removes the project from your library. Video files, saved
              review data and outputs stay on disk.
            </p>
            <div class="actions">
              <button
                class="danger"
                :disabled="busy || running(project)"
                @click="remove(project)"
              >
                {{ busy ? "Deleting…" : "Delete project" }}
              </button>
              <button
                :disabled="busy"
                @click="
                  removing = null;
                  error = '';
                "
              >
                Cancel
              </button>
            </div>
          </div>
          <p
            v-if="error && (editing === project.id || removing === project.id)"
            class="notice error project-edit"
            role="alert"
          >
            {{ error }}
          </p>
        </li>
      </ul>
    </template>
  </section>
</template>
