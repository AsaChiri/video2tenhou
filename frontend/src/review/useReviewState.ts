import { computed, onBeforeUnmount, ref } from "vue";
import { api as request } from "../shared/api";
import { reviewDecisions } from "./decisions";
import type { RequestOptions } from "../shared/api";
import type {
  HandEntry,
  ReviewItem,
  Facts,
  Fact,
  Decision,
  Job,
} from "../types";

export function useReviewState(base: string) {
  const hands = ref<HandEntry[]>([]),
    items = ref<ReviewItem[]>([]),
    facts = ref<Facts>({}),
    selection = ref<Decision | null>(null);
  const guided = ref(false),
    saving = ref(false),
    deferred = ref(new Set<string>());
  const dirty = ref(false),
    players = ref(0),
    error = ref(""),
    message = ref(""),
    updates = ref(false),
    revision = ref("");
  const job = ref<Job>({ running: false, pending: [] }),
    localPending = ref<number[]>([]),
    requesting = ref(false);
  const url = (path: string) => `${base}/api/${path}`;
  const api = <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(url(path), body, options);
  const decisions = computed(() =>
    reviewDecisions(items.value, hands.value, facts.value),
  );
  const key = ({ item, decision }: Decision) =>
    JSON.stringify([
      item.hand,
      decision.field || decision.kind,
      decision.seat,
      decision.j,
      decision.t,
      decision.text,
    ]);
  const questions = computed(() =>
    decisions.value.filter(
      (row) =>
        !deferred.value.has(key(row)) && !blocked.value.has(row.item.hand),
    ),
  );
  const skipped = computed(() =>
    decisions.value.filter(
      (row) =>
        deferred.value.has(key(row)) && !blocked.value.has(row.item.hand),
    ),
  );
  const updating = computed(
    () => saving.value || requesting.value || job.value.running,
  );
  const pending = computed(() =>
    [
      ...new Set([
        ...(job.value.pending || []),
        ...localPending.value,
        ...hands.value
          .filter((hand) => hand.pending_rebuild || (hand.facts_newer ?? 0) > 0)
          .map((hand) => hand.hand),
      ]),
    ].sort((a, b) => a - b),
  );
  const blocked = computed(
    () =>
      new Set([
        ...pending.value,
        ...(job.value.running
          ? job.value.hands || hands.value.map((hand) => hand.hand)
          : []),
      ]),
  );
  let disposed = false,
    mutation = 0,
    timer: ReturnType<typeof setTimeout> | undefined,
    polling = false;
  onBeforeUnmount(() => {
    disposed = true;
    clearTimeout(timer);
  });
  function select(decision: Decision | null) {
    selection.value = decision;
    dirty.value = false;
  }
  function keepSelection() {
    if (
      !selection.value ||
      !questions.value.some((row) => key(row) === key(selection.value!))
    )
      select(questions.value[0] || null);
  }
  function next() {
    if (saving.value) return;
    if (selection.value) deferred.value.add(key(selection.value));
    select(questions.value[0] || null);
  }
  function revisit() {
    deferred.value.clear();
    select(questions.value[0] || null);
  }
  async function load() {
    const version = mutation;
    const [newHands, newItems, newFacts, status] = await Promise.all([
      api<HandEntry[]>("hands"),
      api<ReviewItem[]>("items"),
      api<Fact[]>("facts"),
      api<Job>("decode_pending", undefined, { jobStatus: true }),
    ]);
    if (disposed || version !== mutation) return false;
    hands.value = newHands;
    items.value = newItems;
    facts.value = Object.groupBy(newFacts, (fact) => fact.hand);
    job.value = status;
    if (status.running && !requesting.value && !timer)
      timer = setTimeout(() => rebuild("decode_pending", true), 2000);
    localPending.value = [];
    if (status.error) error.value = status.error;
    return true;
  }
  async function refresh(preserve = false) {
    const loaded = await load();
    if (disposed) return;
    if (!loaded) return;
    if (preserve) keepSelection();
    else select(questions.value[0] || null);
    updates.value = false;
    revision.value = JSON.stringify(await api("revision"));
  }
  async function poll() {
    if (polling || updating.value) return;
    polling = true;
    try {
      const value = JSON.stringify(await api("revision"));
      if (disposed || updating.value) return;
      if (revision.value && value !== revision.value) {
        if (dirty.value || players.value) updates.value = true;
        else await refresh();
      }
      if (!dirty.value && !players.value) revision.value = value;
    } catch {
      /* Retry transient file writes without dropping the current answer. */
    } finally {
      polling = false;
    }
  }
  async function save(fact: Fact) {
    if (saving.value || (guided.value && blocked.value.has(fact.hand)))
      throw new Error(
        "This hand is already waiting for an update. Answer another hand first.",
      );
    saving.value = true;
    mutation++;
    try {
      const current = selection.value;
      const result = await api<Fact>("facts", fact);
      mutation++;
      (facts.value[fact.hand] ??= []).push(result);
      localPending.value.push(fact.hand);
      dirty.value = false;
      message.value = guided.value ? "" : "Answer saved.";
      if (guided.value) {
        if (fact.kind === "lost" && current) deferred.value.add(key(current));
        select(questions.value[0] || null);
      }
      return result;
    } finally {
      saving.value = false;
      startQueued();
    }
  }
  async function remove(fact: Fact) {
    mutation++;
    await api("facts/delete", { ts: fact.ts });
    mutation++;
    facts.value[fact.hand] = (facts.value[fact.hand] || []).filter(
      (row) => row.ts !== fact.ts,
    );
    localPending.value.push(fact.hand);
  }
  function startQueued() {
    if (
      !disposed &&
      guided.value &&
      pending.value.length &&
      !updating.value &&
      !error.value
    )
      void rebuild();
  }
  async function rebuild(path = "decode_pending", resume = false) {
    if (requesting.value) return;
    if (!resume && job.value.running) return;
    clearTimeout(timer);
    timer = undefined;
    if (!resume && dirty.value && !guided.value) {
      message.value = "Save the current answer before rebuilding changes.";
      return;
    }
    requesting.value = true;
    error.value = "";
    try {
      const status = await api<Job>(path, resume ? undefined : {}, {
        jobStatus: true,
      });
      if (disposed) return;
      job.value = {
        ...status,
        pending: status.pending || pending.value,
        hands:
          status.hands ||
          (path === "decode_all"
            ? hands.value.map((hand) => hand.hand)
            : path.startsWith("decode/")
              ? [Number(path.slice(7))]
              : status.pending || pending.value),
      };
      if (status.running) {
        if (guided.value) keepSelection();
        timer = setTimeout(() => rebuild(path, true), 2000);
      } else {
        if (status.error) error.value = status.error;
        if (!guided.value && (dirty.value || players.value))
          updates.value = true;
        else await refresh(true);
      }
    } catch (failure) {
      error.value =
        failure instanceof Error ? failure.message : String(failure);
      job.value.running = false;
    } finally {
      requesting.value = false;
      startQueued();
    }
  }
  return {
    url,
    api,
    hands,
    items,
    facts,
    guided,
    saving,
    selection,
    dirty,
    players,
    error,
    message,
    updates,
    revision,
    job,
    pending,
    requesting,
    updating,
    decisions,
    skipped,
    questions,
    select,
    next,
    revisit,
    load,
    refresh,
    poll,
    save,
    remove,
    rebuild,
  };
}
