import { defineComponent, nextTick, ref } from "vue";
import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import { api } from "../src/shared/api";
import { IDLE_DELAY, RUNNING_DELAY, useJobStatus } from "../src/shared/useJob";
import type { JobState } from "../src/shared/useJob";
import type { JobStatus, WorkspaceJob } from "../src/types";

vi.mock("../src/shared/api", () => ({ api: vi.fn() }));

function deferred() {
  let resolve!: (status: JobStatus) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<JobStatus>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

const job = (extra: Partial<WorkspaceJob> = {}): WorkspaceJob => ({
  kind: "rebuild",
  project: "A",
  stage: "Updating hands",
  hands: [0],
  running: true,
  started: 2,
  finished: null,
  error: null,
  log_lines: 0,
  ...extra,
});

function setup() {
  const requests: ReturnType<typeof deferred>[] = [],
    project = ref<string | null>("A"),
    finished = vi.fn();
  vi.mocked(api).mockImplementation(() => {
    const request = deferred();
    requests.push(request);
    return request.promise;
  });
  let state: JobState | undefined;
  const wrapper = mount(
    defineComponent({
      setup() {
        state = useJobStatus(project);
        state.onFinished(finished);
        return () => null;
      },
    }),
  );
  if (!state) throw new Error("Job state was not initialized.");
  return { state, requests, project, finished, wrapper };
}

beforeEach(() => {
  vi.mocked(api).mockReset();
  vi.useFakeTimers();
  vi.setSystemTime(0);
});

it.each(["older first", "newer first"])(
  "ignores a superseded refresh response resolving %s",
  async (order) => {
    const { state, requests, finished } = setup();
    const refresh = state.refresh(),
      current = { job: job(), revision: "new" },
      stale = {
        job: job({ started: 1, running: false, finished: 2 }),
        revision: "old",
      };
    expect(api).toHaveBeenCalledTimes(2);
    if (order === "older first") {
      requests[0]!.resolve(stale);
      await flushPromises();
      expect(state.job.value).toBeNull();
      expect(state.revision.value).toBeNull();
      expect(vi.getTimerCount()).toBe(0);
    }
    vi.setSystemTime(100);
    requests[1]!.resolve(current);
    await refresh;
    if (order === "newer first") {
      vi.setSystemTime(200);
      requests[0]!.resolve(stale);
      await flushPromises();
    }
    expect(state.job.value).toEqual(current.job);
    expect(state.revision.value).toBe("new");
    expect(state.now.value).toBe(100);
    expect(finished).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(1);
    await vi.advanceTimersByTimeAsync(RUNNING_DELAY);
    expect(api).toHaveBeenCalledTimes(3);
  },
);

it.each([null, job({ started: 1, running: false, finished: 2 })])(
  "keeps a directly shown job when an older poll returns %j",
  async (staleJob) => {
    const { state, requests, finished } = setup();
    const current = { job: job(), revision: "new" };
    vi.setSystemTime(100);
    state.show(current);
    vi.setSystemTime(200);
    requests[0]!.resolve({ job: staleJob, revision: "old" });
    await flushPromises();
    expect(state.job.value).toEqual(current.job);
    expect(state.revision.value).toBe("new");
    expect(state.now.value).toBe(100);
    expect(finished).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(1);

    await vi.advanceTimersByTimeAsync(RUNNING_DELAY - 1);
    expect(api).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(api).toHaveBeenCalledTimes(2);
    const completed = job({ running: false, finished: 3 });
    requests[1]!.resolve({ job: completed, revision: "complete" });
    await flushPromises();
    expect(state.revision.value).toBe("complete");
    expect(finished).toHaveBeenCalledExactlyOnceWith(completed);
    await vi.advanceTimersByTimeAsync(IDLE_DELAY);
    expect(api).toHaveBeenCalledTimes(3);
    requests[2]!.resolve({ job: completed, revision: "complete" });
    await flushPromises();
    expect(finished).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1);
  },
);

it("ignores an earlier visit's poll after switching projects and returning", async () => {
  const { state, requests, project, finished } = setup();
  project.value = "B";
  await nextTick();
  project.value = "A";
  await nextTick();
  expect(vi.mocked(api).mock.calls).toEqual([
    ["/api/job?project=A"],
    ["/api/job?project=B"],
    ["/api/job?project=A"],
  ]);
  const current = { job: job(), revision: "returned" };
  requests[2]!.resolve(current);
  await flushPromises();
  requests[1]!.resolve({ job: null, revision: "B" });
  requests[0]!.resolve({ job: null, revision: "first visit" });
  await flushPromises();
  expect(state.job.value).toEqual(current.job);
  expect(state.revision.value).toBe("returned");
  expect(finished).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(1);
});

it("retries a current failure without restarting an obsolete poll", async () => {
  const { state, requests, finished } = setup();
  const current = { job: job(), revision: "new" };
  state.show(current);
  requests[0]!.reject(new Error("Old request failed"));
  await flushPromises();
  expect(vi.getTimerCount()).toBe(1);
  await vi.advanceTimersByTimeAsync(RUNNING_DELAY);
  requests[1]!.reject(new Error("Temporary failure"));
  await flushPromises();
  expect(state.job.value).toEqual(current.job);
  expect(state.revision.value).toBe("new");
  expect(finished).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(1);
  await vi.advanceTimersByTimeAsync(RUNNING_DELAY);
  expect(api).toHaveBeenCalledTimes(3);
  requests[2]!.resolve({ job: current.job, revision: "recovered" });
  await flushPromises();
  expect(state.revision.value).toBe("recovered");
});

it("ignores a response after unmount without scheduling another poll", async () => {
  const { state, requests, finished, wrapper } = setup();
  const current = { job: job(), revision: "new" };
  state.show(current);
  wrapper.unmount();
  requests[0]!.resolve({
    job: job({ running: false, finished: 3 }),
    revision: "late",
  });
  await flushPromises();
  expect(state.job.value).toEqual(current.job);
  expect(state.revision.value).toBe("new");
  expect(finished).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0);
});
