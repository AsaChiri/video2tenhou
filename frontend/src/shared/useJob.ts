import { inject, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { InjectionKey, Ref } from "vue";
import { api } from "./api";
import type { JobStatus, WorkspaceJob } from "../types";

export const RUNNING_DELAY = 1000;
export const IDLE_DELAY = 5000;

export interface JobState {
  /** The workspace's current or most recent job. */
  job: Ref<WorkspaceJob | null>;
  /** Changes when the open project's review data changes on disk. */
  revision: Ref<string | null>;
  /** Time of the latest status, for elapsed-time displays. */
  now: Ref<number>;
  refresh: () => Promise<void>;
  /** Adopt the status a job request returned, without waiting for a poll. */
  show: (status: JobStatus) => void;
  /** Call `callback` when a job seen running finishes (until unmount). */
  onFinished: (callback: (job: WorkspaceJob) => void) => void;
}
export const jobKey: InjectionKey<JobState> = Symbol("job");

const jobId = (job: WorkspaceJob) =>
  `${job.kind}:${job.project}:${job.started}`;

/** The studio's single status poll: fast while a job runs, slow otherwise. */
export function useJobStatus(project: Ref<string | null>): JobState {
  const job = ref<WorkspaceJob | null>(null),
    revision = ref<string | null>(null),
    now = ref(Date.now());
  const listeners = new Set<(job: WorkspaceJob) => void>(),
    reported = new Set<string>();
  let timer: ReturnType<typeof setTimeout> | undefined,
    generation = 0,
    first = true,
    disposed = false;
  // Every job that finishes after the first status is reported once, even one
  // that started and ended between two polls.
  function show(status: JobStatus) {
    job.value = status.job;
    revision.value = status.revision;
    now.value = Date.now();
    const current = status.job,
      initial = first;
    first = false;
    if (!current || current.running || reported.has(jobId(current))) return;
    reported.add(jobId(current));
    if (!initial) for (const listener of [...listeners]) listener(current);
  }
  function schedule(run: number) {
    if (!disposed && run === generation)
      timer = setTimeout(
        () => tick(run),
        job.value?.running ? RUNNING_DELAY : IDLE_DELAY,
      );
  }
  async function tick(run: number) {
    const asked = project.value;
    try {
      const status = await api<JobStatus>(
        asked ? `/api/job?project=${encodeURIComponent(asked)}` : "/api/job",
      );
      if (!disposed && asked === project.value) show(status);
    } catch {
      /* A transient failure keeps the last status; the next poll retries. */
    } finally {
      schedule(run);
    }
  }
  function restart() {
    clearTimeout(timer);
    return tick(++generation);
  }
  watch(project, () => {
    revision.value = null;
    void restart();
  });
  onMounted(restart);
  onBeforeUnmount(() => {
    disposed = true;
    clearTimeout(timer);
  });
  function onFinished(callback: (job: WorkspaceJob) => void) {
    listeners.add(callback);
    onBeforeUnmount(() => listeners.delete(callback));
  }
  return {
    job,
    revision,
    now,
    refresh: restart,
    show: (status) => {
      show(status);
      clearTimeout(timer);
      schedule(++generation);
    },
    onFinished,
  };
}

export function useJob(): JobState {
  const state = inject(jobKey);
  if (!state) throw new Error("Job status requires the studio application.");
  return state;
}
