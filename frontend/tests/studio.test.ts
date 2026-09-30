import { mount, flushPromises } from "@vue/test-utils";
import type { VueWrapper } from "@vue/test-utils";
import type { Project } from "../src/types";
import { defineComponent } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
import { api, upload } from "../src/shared/api";
import StudioApp from "../src/studio/StudioApp.vue";
import RecordingForm from "../src/studio/RecordingForm.vue";
import ProjectSettings from "../src/studio/ProjectSettings.vue";
import ProjectLibrary from "../src/studio/ProjectLibrary.vue";
import ProcessingLog from "../src/studio/ProcessingLog.vue";
import ResultsPanel from "../src/studio/ResultsPanel.vue";
vi.mock("../src/shared/api", () => ({ api: vi.fn(), upload: vi.fn() }));
const project = (id: string, extra: Partial<Project> = {}): Project => ({
  id,
  name: id,
  games: [12],
  source: "video.mp4",
  start: "",
  end: "",
  open_items: 0,
  layout: "pml",
  has_fit: true,
  artifacts: ["g0.json"],
  job: { running: false, log: [], started: 1 },
  ...extra,
});
const results = (id: string) => ({
  games: [
    {
      index: 0,
      names: [id],
      hands: [
        {
          index: 0,
          round: "East 1",
          honba: 0,
          editor_url: "https://example.com/hand",
        },
      ],
      viewer_url: "https://example.com",
      download: `/exports/${id}/g0.json`,
    },
  ],
  pending_games: [],
});
function button(wrapper: Pick<VueWrapper, "findAll">, text: string) {
  const found = wrapper
    .findAll("button")
    .find((button) => button.text() === text);
  if (!found) throw new Error(`Missing button: ${text}`);
  return found;
}
beforeEach(() => {
  vi.mocked(api).mockReset();
  vi.mocked(upload).mockReset();
  location.hash = "";
});
async function studio(projects: Project[]) {
  vi.mocked(api).mockImplementation(async (path) => {
    if (path === "/api/workspace") return { projects, setup: { ready: true } };
    if (path.endsWith("/results")) return results(path);
    throw Error(path);
  });
  const wrapper = mount(StudioApp, {
    global: {
      stubs: {
        ReviewWorkspace: defineComponent({
          name: "ReviewWorkspace",
          props: ["projectId", "calibration", "active"],
          emits: ["dirty"],
          template: '<div class="review-stub">Review {{ projectId }}</div>',
        }),
      },
    },
  });
  await flushPromises();
  return wrapper;
}
it("lists existing projects on arrival, searches them and opens settings directly", async () => {
  const wrapper = await studio([
    project("A", { display_name: "League final", games: [123] }),
    project("B", { display_name: "Practice", games: [456] }),
  ]);
  expect(wrapper.findAll(".project-row")).toHaveLength(2);
  expect(wrapper.getComponent(RecordingForm).isVisible()).toBe(false);
  await wrapper.get("#project-search").setValue("123");
  expect(wrapper.findAll(".project-row")).toHaveLength(1);
  expect(wrapper.get(".project-row").text()).toContain("League final");
  await button(wrapper, "Settings").trigger("click");
  expect(wrapper.getComponent(ProjectSettings).props("project").id).toBe("A");
  await button(wrapper, "All projects").trigger("click");
  await button(wrapper, "New project").trigger("click");
  expect(wrapper.getComponent(RecordingForm).attributes("style")).not.toContain(
    "display: none",
  );
});
it("renames a project and requires confirmation before removing it", async () => {
  const wrapper = await studio([project("A")]);
  await button(wrapper, "Rename").trigger("click");
  await wrapper.get("#name-A").setValue("Final table");
  vi.mocked(api).mockResolvedValue(
    project("A", { display_name: "Final table" }),
  );
  await wrapper.get(".project-edit").trigger("submit");
  await flushPromises();
  expect(api).toHaveBeenCalledWith("/api/projects/A/rename", {
    display_name: "Final table",
  });
  expect(wrapper.get(".project-summary h2").text()).toBe("Final table");
  await button(wrapper, "Delete").trigger("click");
  expect(api).not.toHaveBeenCalledWith("/api/projects/A/delete", {});
  await button(wrapper, "Cancel").trigger("click");
  expect(wrapper.findAll(".project-row")).toHaveLength(1);
  await button(wrapper, "Delete").trigger("click");
  vi.mocked(api).mockResolvedValue({ deleted: "A" });
  await button(wrapper, "Delete project").trigger("click");
  await flushPromises();
  expect(api).toHaveBeenCalledWith("/api/projects/A/delete", {});
  expect(wrapper.findAll(".project-row")).toHaveLength(0);
  expect(wrapper.text()).toContain("Your recordings start here");
});
it("retains project and input when a mutation fails, and blocks deleting active jobs", async () => {
  const wrapper = mount(ProjectLibrary, {
    props: { projects: [project("A")], loading: false },
  });
  await button(wrapper, "Rename").trigger("click");
  await wrapper.get("#name-A").setValue("Unsaved name");
  vi.mocked(api).mockRejectedValue(new Error("Disk unavailable"));
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(wrapper.get('[role="alert"]').text()).toContain("Disk unavailable");
  expect(wrapper.get<HTMLInputElement>("#name-A").element.value).toBe(
    "Unsaved name",
  );
  expect(wrapper.emitted("updated")).toBeUndefined();
  await button(wrapper, "Cancel").trigger("click");
  await button(wrapper, "Delete").trigger("click");
  await button(wrapper, "Delete project").trigger("click");
  await flushPromises();
  expect(wrapper.emitted("deleted")).toBeUndefined();
  expect(wrapper.findAll(".project-row")).toHaveLength(1);
  await wrapper.setProps({
    projects: [project("A", { review_running: true })],
  });
  expect(
    button(wrapper, "Delete project").attributes("disabled"),
  ).toBeDefined();
});
it("opens saved project links and follows browser navigation", async () => {
  history.replaceState(null, "", "#A");
  const wrapper = await studio([project("A"), project("B")]);
  expect(wrapper.get(".current-project-name").text()).toBe("A");
  history.replaceState(null, "", "#B");
  window.dispatchEvent(new Event("hashchange"));
  await flushPromises();
  expect(wrapper.get(".current-project-name").text()).toBe("B");
  history.replaceState(null, "", "/");
  window.dispatchEvent(new Event("hashchange"));
  await flushPromises();
  expect(wrapper.findComponent(ProjectLibrary).exists()).toBe(true);
});
it("shows a retryable load failure without claiming the project library is empty", async () => {
  vi.mocked(api).mockRejectedValueOnce(new Error("Server unavailable"));
  const wrapper = mount(StudioApp);
  await flushPromises();
  expect(wrapper.text()).toContain("Could not load projects");
  expect(wrapper.text()).not.toContain("Your recordings start here");
  vi.mocked(api).mockResolvedValue({
    projects: [project("A")],
    setup: { ready: true },
  });
  await button(wrapper, "Retry").trigger("click");
  await flushPromises();
  expect(wrapper.findAll(".project-row")).toHaveLength(1);
  expect(wrapper.text()).not.toContain("Could not load projects");
});
it("does not let an older poll resurrect a deleted project", async () => {
  vi.useFakeTimers();
  const wrapper = await studio([project("A")]);
  let finish: ((value: unknown) => void) | undefined;
  vi.mocked(api).mockImplementation((path) =>
    path === "/api/workspace"
      ? new Promise((resolve) => (finish = resolve))
      : Promise.resolve({ deleted: "A" }),
  );
  await vi.advanceTimersByTimeAsync(2500);
  await button(wrapper, "Delete").trigger("click");
  await button(wrapper, "Delete project").trigger("click");
  await flushPromises();
  if (!finish) throw Error("Poll did not start");
  finish({ projects: [project("A")], setup: { ready: true } });
  await flushPromises();
  expect(wrapper.findAll(".project-row")).toHaveLength(0);
});
it("retains settings input and the review component through unrelated polls", async () => {
  vi.useFakeTimers();
  const wrapper = await studio([project("A")]);
  await wrapper.get('.project-summary a[href="#A"]').trigger("click");
  await flushPromises();
  const review = wrapper.get(".review-stub").element;
  await button(wrapper, "Settings").trigger("click");
  await wrapper.get("#settings-games").setValue("99, 100");
  await vi.advanceTimersByTimeAsync(2500);
  expect(wrapper.get<HTMLInputElement>("#settings-games").element.value).toBe(
    "99, 100",
  );
  await button(wrapper, "Review").trigger("click");
  expect(wrapper.get(".review-stub").element).toBe(review);
});
it("blocks navigation and analysis while calibration is dirty", async () => {
  const wrapper = await studio([project("A", { artifacts: [] }), project("B")]);
  await wrapper.get('.project-summary a[href="#A"]').trigger("click");
  await flushPromises();
  wrapper.findComponent({ name: "ReviewWorkspace" }).vm.$emit("dirty", true);
  await flushPromises();
  await button(wrapper, "Analyze recording").trigger("click");
  await button(wrapper, "All projects").trigger("click");
  expect(wrapper.get(".review-stub").text()).toBe("Review A");
  expect(wrapper.text()).toContain("Save or discard");
  expect(
    vi.mocked(api).mock.calls.some(([path]) => path.endsWith("/analyze")),
  ).toBe(false);
});
it("opens calibration in settings after a failed prepare and retries only on request", async () => {
  const failed = project("A", {
    has_fit: false,
    can_calibrate: true,
    artifacts: [],
    job: { running: false, error: "No stable play windows found." },
  });
  const wrapper = await studio([failed]);
  await wrapper.get('.project-summary a[href="#A"]').trigger("click");
  await button(wrapper, "Settings").trigger("click");
  expect(button(wrapper, "Calibration").attributes("disabled")).toBeUndefined();
  await button(wrapper, "Calibration").trigger("click");
  expect(wrapper.get(".review-stub").isVisible()).toBe(true);
  expect(
    vi.mocked(api).mock.calls.some(([path]) => path.endsWith("/prepare")),
  ).toBe(false);
  await button(wrapper, "Back to settings").trigger("click");
  vi.mocked(api).mockResolvedValue({ ...failed, job: { running: true } });
  await button(
    wrapper.getComponent(ProjectSettings),
    "Prepare recording",
  ).trigger("click");
  await flushPromises();
  expect(api).toHaveBeenCalledWith("/api/projects/A/prepare", {});
});
it("keeps calibration unavailable while a recording is missing or processing", async () => {
  const wrapper = mount(ProjectSettings, {
    props: { project: project("A", { can_calibrate: false, has_fit: false }) },
  });
  expect(button(wrapper, "Calibration").attributes("disabled")).toBeDefined();
  await wrapper.setProps({
    project: project("A", { can_calibrate: true, job: { running: true } }),
  });
  expect(button(wrapper, "Calibration").attributes("disabled")).toBeDefined();
});
it("does not let a delayed action replace another selected recording", async () => {
  const wrapper = await studio([project("A", { artifacts: [] }), project("B")]);
  await wrapper.get('.project-summary a[href="#A"]').trigger("click");
  let finish: ((value: unknown) => void) | undefined;
  vi.mocked(api).mockImplementation((path) =>
    path.endsWith("/analyze")
      ? new Promise((resolve) => (finish = resolve))
      : Promise.resolve({ projects: [], setup: { ready: true } }),
  );
  await button(wrapper, "Analyze recording").trigger("click");
  await button(wrapper, "All projects").trigger("click");
  if (!finish) throw new Error("Request did not start.");
  await wrapper.get('.project-summary a[href="#B"]').trigger("click");
  finish(project("A", { job: { running: true, log: [] } }));
  await flushPromises();
  expect(wrapper.get(".review-stub").text()).toBe("Review B");
});
it("keeps the latest project results when an older request completes last", async () => {
  const pending: ((value: unknown) => void)[] = [];
  vi.mocked(api).mockImplementation(
    () => new Promise((resolve) => pending.push(resolve)),
  );
  const wrapper = mount(ResultsPanel, { props: { project: project("A") } });
  await wrapper.setProps({ project: project("B") });
  pending[1](results("B"));
  await flushPromises();
  pending[0](results("A"));
  await flushPromises();
  expect(wrapper.get(".players").text()).toBe("B");
});
it("does not restart a slow results request on identical poll data", async () => {
  let finish: ((value: unknown) => void) | undefined;
  vi.mocked(api).mockImplementation(
    () => new Promise((resolve) => (finish = resolve)),
  );
  const wrapper = mount(ResultsPanel, { props: { project: project("A") } });
  await wrapper.setProps({ project: project("A", { open_items: 4 }) });
  expect(api).toHaveBeenCalledTimes(1);
  if (!finish) throw new Error("Request did not start.");
  finish(results("A"));
  await flushPromises();
  expect(wrapper.text()).toContain("Hanchan 1");
});
it("withholds replay and download links for games with pending answers", async () => {
  vi.mocked(api).mockResolvedValue({ ...results("A"), pending_games: [0] });
  const wrapper = mount(ResultsPanel, { props: { project: project("A") } });
  await flushPromises();
  expect(wrapper.text()).toContain("Return to Review to update the record");
  expect(wrapper.find("a[download]").exists()).toBe(false);
});
it("preserves settings drafts and reports an API failure", async () => {
  vi.mocked(api).mockRejectedValue(new Error("Cannot save"));
  const wrapper = mount(ProjectSettings, { props: { project: project("A") } });
  await wrapper.get("#settings-games").setValue("15");
  await wrapper.setProps({
    project: project("A", {
      job: { running: false, error: "failed", log: ["failed"] },
    }),
  });
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(wrapper.get<HTMLInputElement>("#settings-games").element.value).toBe(
    "15",
  );
  expect(wrapper.emitted("error")).toEqual([["Cannot save"]]);
});
it("captures the local file and form before a slow copy", async () => {
  let finish: ((value: string) => void) | undefined;
  vi.mocked(upload).mockImplementation(
    () => new Promise((resolve) => (finish = resolve)),
  );
  vi.mocked(api).mockResolvedValue(project("A"));
  const wrapper = mount(RecordingForm);
  await wrapper.get("#games").setValue("12");
  const file = new File(["video"], "local.mp4");
  Object.defineProperty(wrapper.get("input[type=file]").element, "files", {
    value: [file],
  });
  await wrapper.get("input[type=file]").trigger("change");
  await wrapper.get("form").trigger("submit");
  await button(wrapper, "Video URL").trigger("click");
  await wrapper.get("#games").setValue("99");
  if (!finish) throw new Error("Upload did not start.");
  finish("copied.mp4");
  await flushPromises();
  expect(vi.mocked(upload).mock.calls[0][0]).toBe(file);
  expect(api).toHaveBeenCalledWith("/api/projects", {
    kind: "local",
    games: [12],

    layout: "",
    start: "",
    end: "",
    source: "copied.mp4",
  });
});
it("does not upload a stale local selection when importing a URL", async () => {
  vi.mocked(api).mockResolvedValue(project("A"));
  const wrapper = mount(RecordingForm);
  await wrapper.get("#games").setValue("12");
  await button(wrapper, "Video URL").trigger("click");
  await wrapper.get("#url-source").setValue("https://example.com/video");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(upload).not.toHaveBeenCalled();
  expect(vi.mocked(api).mock.calls[0][1]).toMatchObject({
    kind: "url",
    source: "https://example.com/video",
  });
});
it.each([
  ["local", "Choose a video file."],
  ["url", "Enter a remote video URL."],
])("keeps an empty %s form correctable", async (kind, error) => {
  const wrapper = mount(RecordingForm);
  await wrapper.get("#games").setValue("12");
  if (kind === "url") await button(wrapper, "Video URL").trigger("click");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(wrapper.emitted("error")?.at(-1)).toEqual([error]);
  expect(
    button(wrapper, "Prepare recording").attributes("disabled"),
  ).toBeUndefined();
  expect(api).not.toHaveBeenCalled();
});
it("rejects a reversed time range before copying a recording", async () => {
  const wrapper = mount(RecordingForm);
  await wrapper.get("#games").setValue("12");
  await wrapper.get("#start-time").setValue("2:00");
  await wrapper.get("#end-time").setValue("1:00");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(upload).not.toHaveBeenCalled();
  expect(wrapper.emitted("error")?.at(-1)?.[0]).toContain("after start");
});
it("updates logs as text, preserves expansion and stops following when reading older output", async () => {
  const wrapper = mount(ProcessingLog, {
    props: { job: { started: 1, log: ["first"] } },
  });
  const output = wrapper.get("pre").element;
  Object.defineProperties(output, {
    scrollHeight: { value: 1000, configurable: true },
    clientHeight: { value: 100 },
  });
  wrapper.get("details").element.open = true;
  await wrapper.get("details").trigger("toggle");
  await flushPromises();
  expect(output.scrollTop).toBe(1000);
  output.scrollTop = 200;
  await wrapper.get("pre").trigger("scroll");
  await wrapper.setProps({
    job: { started: 1, log: ["first", "<script>latest</script>"] },
  });
  await flushPromises();
  expect(output.scrollTop).toBe(200);
  expect(wrapper.get("details").element.open).toBe(true);
  expect(wrapper.find("script").exists()).toBe(false);
  await wrapper.setProps({ job: { started: 2, log: ["retry"] } });
  await flushPromises();
  expect(output.scrollTop).toBe(1000);
});
