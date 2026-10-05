import { computed, onBeforeUnmount, ref, watch } from "vue";
import { api as request } from "../shared/api";
import { errorText } from "../shared/useAction";
import type { JobState } from "../shared/useJob";
import { reviewDecisions } from "./decisions";
import type {
  Decision,
  Fact,
  FactBody,
  Facts,
  HandRow,
  JobStatus,
  ReviewItem,
} from "../types";

/**
 * Review data of one project. A hand with new or deleted answers is pending: its
 * questions wait until the server has updated it, and the update then asks again
 * only what is still open. Job progress comes from the studio's status poll.
 */
export function useReviewState(projectId: string, status: JobState) {
  const url = (path: string) => `/review/${projectId}/api/${path}`;
  const api = <T>(path: string, body?: unknown) => request<T>(url(path), body);
  const hands = ref<HandRow[]>([]),
    items = ref<ReviewItem[]>([]),
    facts = ref<Facts>({}),
    selection = ref<Decision | null>(null);
  const guided = ref(false),
    saving = ref(false),
    requesting = ref(false),
    skipped = ref(new Set<string>()),
    answered = ref(new Set<number>());
  const dirty = ref(false),
    players = ref(0),
    error = ref(""),
    message = ref(""),
    updates = ref(false),
    loads = ref(0);
  // `loaded` is the revision the data reflects; `ownChange` marks a revision
  // change caused by this tab's own answer; `queued` allows one automatic update.
  let loaded: string | null = null,
    ownChange = false,
    queued = true,
    mutation = 0,
    disposed = false;
  onBeforeUnmount(() => (disposed = true));
  const job = computed(() =>
    status.job.value?.project === projectId ? status.job.value : null,
  );
  const updating = computed(
    () => !!job.value?.running && job.value.kind === "rebuild",
  );
  const busy = computed(() => !!status.job.value?.running);
  const pending = computed(() =>
    [
      ...new Set([
        ...hands.value.filter((hand) => hand.pending).map((hand) => hand.hand),
        ...answered.value,
        ...(updating.value ? job.value?.hands || [] : []),
      ]),
    ].sort((a, b) => a - b),
  );
  const decisions = computed(() => reviewDecisions(items.value));
  const open = computed(() => {
    const waiting = new Set(pending.value);
    return decisions.value.filter((row) => !waiting.has(row.item.hand));
  });
  const questions = computed(() =>
    open.value.filter((row) => !skipped.value.has(row.key)),
  );
  const deferred = computed(() =>
    open.value.filter((row) => skipped.value.has(row.key)),
  );
  const waiting = computed(() => decisions.value.length - open.value.length);

  function select(decision: Decision | null) {
    selection.value = decision;
    dirty.value = false;
  }
  function keepSelection() {
    const key = selection.value?.key;
    if (!questions.value.some((row) => row.key === key))
      select(questions.value[0] || null);
  }
  function next() {
    if (saving.value) return;
    if (selection.value)
      skipped.value = new Set([...skipped.value, selection.value.key]);
    select(questions.value[0] || null);
  }
  function revisit() {
    skipped.value = new Set();
    select(questions.value[0] || null);
  }
  async function load() {
    const at = mutation;
    const [newHands, newItems, newFacts] = await Promise.all([
      api<HandRow[]>("hands"),
      api<ReviewItem[]>("items"),
      api<Fact[]>("facts"),
    ]);
    if (disposed || at !== mutation) return false;
    hands.value = newHands;
    items.value = newItems;
    facts.value = Object.groupBy(newFacts, (fact) => fact.hand);
    answered.value = new Set();
    loads.value++;
    return true;
  }
  async function refresh(preserve = false) {
    loaded = status.revision.value;
    if (!(await load())) return;
    if (preserve) keepSelection();
    else select(questions.value[0] || null);
    updates.value = false;
  }
  watch(status.revision, (value) => {
    if (!value || value === loaded) return;
    if (loaded === null || ownChange) {
      loaded = value;
      ownChange = false;
    } else if (!saving.value && !updating.value) {
      if (dirty.value || players.value) updates.value = true;
      else void refresh(true);
    }
  });
  async function post(fact: FactBody) {
    mutation++;
    const row = await api<Fact>("facts", fact);
    mutation++;
    ownChange = true;
    (facts.value[row.hand] ??= []).push(row);
    if (row.kind !== "dismiss") {
      answered.value = new Set([...answered.value, row.hand]);
      queued = true;
    }
    return row;
  }
  async function save(fact: FactBody & { hand: number }) {
    if (saving.value) return undefined;
    if (guided.value && pending.value.includes(fact.hand))
      throw new Error(
        "This hand is being updated. Answer a question from another hand.",
      );
    saving.value = true;
    try {
      const row = await post(fact);
      dirty.value = false;
      message.value = guided.value ? "" : "Answer saved.";
      if (guided.value) select(questions.value[0] || null);
      return row;
    } finally {
      saving.value = false;
      maybeStart();
    }
  }
  /** Close a question that needs no tile answer; this never updates the hand. */
  async function dismiss(item: ReviewItem) {
    if (saving.value) return;
    saving.value = true;
    try {
      await post({ kind: "dismiss", hand: item.hand, item: item.id });
      items.value = items.value.filter(
        (row) => row.hand !== item.hand || row.id !== item.id,
      );
      select(questions.value[0] || null);
    } finally {
      saving.value = false;
    }
  }
  async function remove(fact: Fact) {
    mutation++;
    await api("facts/delete", { ts: fact.ts });
    mutation++;
    ownChange = true;
    facts.value[fact.hand] = (facts.value[fact.hand] || []).filter(
      (row) => row.ts !== fact.ts,
    );
    if (fact.kind === "dismiss") await refresh(true);
    else {
      answered.value = new Set([...answered.value, fact.hand]);
      queued = true;
    }
  }
  function maybeStart() {
    if (
      !disposed &&
      guided.value &&
      queued &&
      pending.value.length &&
      !busy.value &&
      !requesting.value &&
      !error.value
    )
      void rebuild();
  }
  /** Update pending hands, every hand, or the listed hands from saved answers. */
  async function rebuild(hands: "pending" | "all" | number[] = "pending") {
    if (busy.value || requesting.value) return;
    if (dirty.value && !guided.value) {
      message.value = "Save the current answer before applying changes.";
      return;
    }
    requesting.value = true;
    error.value = "";
    queued = false;
    try {
      const started = await request<JobStatus>(url("rebuild"), { hands });
      status.show(started);
      if (!started.job?.running) await refresh(true);
    } catch (failure) {
      error.value = errorText(failure);
    } finally {
      requesting.value = false;
    }
  }
  status.onFinished((finished) => {
    if (finished.project !== projectId || finished.kind !== "rebuild") return;
    if (finished.error) error.value = finished.error;
    if (!guided.value && (dirty.value || players.value)) updates.value = true;
    else void refresh(true).then(maybeStart);
  });
  return {
    projectId,
    url,
    api,
    hands,
    items,
    facts,
    guided,
    saving,
    requesting,
    selection,
    dirty,
    players,
    error,
    message,
    updates,
    loads,
    job,
    updating,
    busy,
    pending,
    waiting,
    decisions,
    questions,
    deferred,
    select,
    next,
    revisit,
    refresh,
    save,
    dismiss,
    remove,
    rebuild,
    maybeStart,
  };
}
