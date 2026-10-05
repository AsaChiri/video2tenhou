import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

const project = {
  id: "a".repeat(32),
  name: "Test recording",
  display_name: "Test recording",
  kind: "local",
  source: "recording.mp4",
  games: [12],
  layout: "pml",
  start: 0,
  end: null,
  status: "complete",
  has_fit: true,
  can_calibrate: true,
  checking: false,
  stale_exports: false,
  results_revision: [],
  artifacts: ["g0.json"],
  open_items: 1,
  job: { running: false, started: 1 },
};
const entry = {
  hand: 0,
  game: 0,
  kyoku: 0,
  honba: 0,
  t_start: 10,
  t_end: 100,
  corner_wind: { TL: "E", TR: "S", BL: "W", BR: "N" },
};
const row = (hand: number, pending = false) => ({
  hand,
  game: 0,
  kyoku: hand,
  honba: 0,
  t_start: 10,
  t_end: 100,
  status: "review",
  pending,
  turns: 1,
  score: null,
});
const decode = {
  turns: [{ i: 0, j: 0, seat: "N", t: 30, t_prev: 20, discard: "1m" }],
  haipai: { N: Array(13).fill("1m") },
  dora: ["2p"],
  result: { outcome: "ryuukyoku" },
};
const draw = {
  id: "draw:N:0",
  hand: 0,
  kind: "draw",
  seat: "N",
  j: 0,
  t: 30,
  margin: 0,
};
interface Server {
  job: Record<string, unknown> | null;
  revision: string;
  hold?: boolean;
}
/** Answer the studio's API; a running update finishes on its next status poll. */
async function serve(page: Page, server: Server) {
  const facts: unknown[] = [];
  let applied = false;
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url()),
      path = url.pathname,
      method = route.request().method();
    let data;
    if (path === "/api/workspace")
      data = { setup: { ready: true, missing: [] }, projects: [project] };
    else if (path === "/api/job") {
      if (server.job?.running && server.job.seen && !server.hold) {
        server.job = { ...server.job, running: false, finished: 2 };
        applied ||= (server.job.hands as number[]).includes(0);
        server.revision += "+";
      } else if (server.job?.running) server.job.seen = true;
      data = { job: server.job, revision: server.revision };
    } else if (path.endsWith("/hands"))
      data = [row(0, facts.length > 0 && !applied)];
    else if (path.endsWith("/items")) data = applied ? [] : [draw];
    else if (path.endsWith("/facts")) {
      if (method === "POST") {
        data = { ...route.request().postDataJSON(), ts: 10 };
        facts.push(data);
      } else data = facts;
    } else if (path.endsWith("/rebuild")) {
      server.job = {
        kind: "rebuild",
        project: project.id,
        stage: "Updating hands",
        hands: [0],
        running: true,
        started: 1,
        finished: null,
        error: null,
        log_lines: 0,
        seen: true,
      };
      data = { job: server.job, revision: server.revision };
    } else if (path.endsWith("/hand/0")) data = { entry, decode, ignored: [] };
    else if (path.endsWith("/results"))
      data = {
        games: [
          {
            index: 0,
            names: ["Alice", "Bob"],
            hands: [
              {
                index: 0,
                round: "East 1",
                honba: 0,
                editor_url: "https://example.com/hand",
              },
            ],
            viewer_url: "https://example.com/replay",
            download: "/exports/game.json",
          },
        ],
        pending_games: [],
      };
    else if (path.endsWith("/calib"))
      data = {
        frame: [1000, 1000],
        fit: { overhead: { center: [500, 500], angle: 0, scale: 1 } },
        regions: {
          "hand:TL": {
            kind: "hand",
            movable: true,
            rect: [100, 100, 100, 100],
            quad: [
              [100, 100],
              [200, 100],
              [200, 200],
              [100, 200],
            ],
            roll: 3,
          },
        },
        checks: {},
      };
    else if (path.endsWith("/clip") || path.endsWith("/plate"))
      return route.fulfill({ status: 204 });
    else throw Error(`Unexpected request: ${method} ${path}`);
    await route.fulfill({ json: data });
  });
}
let server: Server;
test.beforeEach(async ({ page }) => {
  server = { job: null, revision: "initial" };
  await serve(page, server);
});
test("one application opens review, saves an answer, updates the hand and shows results", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 1" }),
  ).toBeVisible();
  expect(await page.locator("iframe").count()).toBe(0);
  expect(await page.locator("video").count()).toBe(1);
  const saved = page.waitForRequest(
    (request) =>
      request.url().endsWith("/facts") && request.method() === "POST",
  );
  const rebuild = page.waitForRequest((request) =>
    request.url().endsWith("/rebuild"),
  );
  await page.getByRole("button", { name: "Choose 0p", exact: true }).click();
  expect((await saved).postDataJSON()).toMatchObject({
    kind: "draw",
    tile: "0p",
    hand: 0,
    j: 0,
  });
  expect((await rebuild).postDataJSON()).toEqual({ hands: "pending" });
  await expect(
    page.getByRole("heading", { name: "No more questions" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Open results", exact: true }).click();
  await expect(page.getByRole("link", { name: "Open replay" })).toBeVisible();
  expect(errors).toEqual([]);
});
test("questions that need no tile are dismissed without updating the hand", async ({
  page,
}, testInfo) => {
  let rebuilds = 0;
  await page.route("**/review/**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/rebuild")) rebuilds++;
    if (path.endsWith("/items"))
      return route.fulfill({
        json: [
          {
            id: "conflict::90",
            hand: 0,
            kind: "conflict",
            t: 90,
            culprit: null,
            text: "No legal reconstruction: check the over-counted tiles.",
            over: [],
          },
        ],
      });
    return route.fallback();
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(
    page.getByText("No legal reconstruction: check the over-counted tiles."),
  ).toBeVisible();
  await expect(page.getByText("diagnosis")).toHaveCount(0);
  await page.screenshot({
    path: testInfo.outputPath("conflict-desktop.png"),
    fullPage: true,
  });
  const dismissed = page.waitForRequest(
    (request) =>
      request.url().endsWith("/facts") && request.method() === "POST",
  );
  await page
    .getByRole("button", { name: "Leave as conflict", exact: true })
    .click();
  expect((await dismissed).postDataJSON()).toEqual({
    kind: "dismiss",
    hand: 0,
    item: "conflict::90",
  });
  await expect(
    page.getByRole("heading", { name: "No more questions" }),
  ).toBeVisible();
  expect(rebuilds).toBe(0);
});
test("advanced review holds the hand inspector, its notes and answers", async ({
  page,
}) => {
  await page.route("**/api/hand/0", (route) =>
    route.fulfill({
      json: {
        entry,
        decode: { ...decode, notes: ["The live wall ends two tiles early."] },
        ignored: [],
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await page
    .getByRole("button", { name: "Advanced review", exact: true })
    .click();
  await page
    .getByRole("button", { name: "East 1, 0 honba", exact: true })
    .click();
  await expect(
    page.getByText("The live wall ends two tiles early."),
  ).toBeVisible();
  await expect(page.getByText("Certified margin")).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Turns", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "1", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Turn 1: North (BR) discards 1m at 0:30",
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Back to questions", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 1" }),
  ).toBeVisible();
});
test("a running update shows its hands and elapsed time, then returns to questions", async ({
  page,
}) => {
  server.job = {
    kind: "rebuild",
    project: project.id,
    stage: "Updating hands",
    hands: [1],
    running: true,
    started: Date.now() / 1000 - 7 * 60,
    finished: null,
    error: null,
    log_lines: 0,
  };
  let posts = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/rebuild")) posts++;
  });
  server.hold = true;
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(page.getByText("Updating hand 2…")).toBeVisible();
  await expect(page.getByText(/^7:0\d$/)).toBeVisible();
  // Questions of other hands stay available during the update.
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 1" }),
  ).toBeVisible();
  server.hold = false;
  await expect(page.getByText("Updating hand 2…")).toHaveCount(0);
  expect(posts).toBe(0);
});
test("calibration edits guard project navigation and discard releases it", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await page.getByRole("tab", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Calibration", exact: true }).click();
  await page.getByText("Overhead adjustment and crop previews").click();
  await page.getByLabel("Overhead centre X").fill("520");
  await page.getByRole("button", { name: "New project", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Save or discard");
  await page
    .getByRole("button", { name: "Discard changes", exact: true })
    .click();
  await page.getByRole("button", { name: "New project", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "New project" }),
  ).toBeVisible();
});
test("mobile intake fits the viewport and keeps its source controls usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "New project", exact: true }).click();
  await page.getByRole("button", { name: "Video URL", exact: true }).click();
  await expect(page.getByLabel("Video URL", { exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});
test("calibration stops at the image edge and blocks invalid numeric positions", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await page.getByRole("tab", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Calibration", exact: true }).click();
  const canvas = page.getByLabel("Camera regions on the table preview");
  await expect(canvas).toHaveAttribute("width", "1000");
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  if (!box) throw Error("Missing calibration canvas");
  await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.15);
  await page.mouse.down();
  await page.mouse.move(box.x - 10, box.y + box.height * 0.15);
  await page.mouse.up();
  const save = page.waitForRequest(
    (request) =>
      request.url().endsWith("/calib") && request.method() === "POST",
  );
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  expect((await save).postDataJSON().hand.TL.rect).toEqual([0, 100, 100, 100]);
  await expect(page.getByRole("status").first()).toContainText("Saved.");
  await page.getByText("Overhead adjustment and crop previews").click();
  await page.getByLabel("Overhead centre X").fill("-2");
  await expect(page.getByRole("alert")).toContainText("inside the image");
  await expect(
    page.getByRole("button", { name: "Save changes", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Check borders", exact: true }),
  ).toBeDisabled();
});
test("failed preparation shows its own message, opens calibration and retries", async ({
  page,
}) => {
  const failed = {
    ...project,
    has_fit: false,
    can_calibrate: true,
    artifacts: [],
    job: {
      action: "prepare",
      running: false,
      error: "Adjust the table borders in Calibration: hand:TL cuts tiles.",
    },
  };
  const geometry = {
    layout: "pml",
    frame: [1000, 1000],
    fit: null,
    overhead: { center: [500, 500], angle: 45, scale: 1 },
    regions: {
      "hand:TL": {
        kind: "hand",
        movable: true,
        rect: [100, 100, 100, 100],
        quad: [
          [100, 100],
          [200, 100],
          [200, 200],
          [100, 200],
        ],
      },
    },
    checks: { "hand:TL": { level: "fail" } },
  };
  let saved: { hand?: { TL: { rect: number[] } } } | undefined;
  let prepares = 0;
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/workspace")
      return route.fulfill({
        json: { projects: [failed], setup: { ready: true } },
      });
    if (path.endsWith("/calib")) {
      if (route.request().method() === "POST") {
        saved = route.request().postDataJSON();
        return route.fulfill({ json: { fit: null } });
      }
      return route.fulfill({ json: geometry });
    }
    if (path.endsWith("/prepare")) {
      prepares++;
      return route.fulfill({ json: { ...failed, job: { running: true } } });
    }
    if (path.endsWith("/frame") || path.endsWith("/plate"))
      return route.fulfill({
        contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000"><rect width="1000" height="1000" fill="#333"/></svg>',
      });
    return route.fallback();
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText(
    /Adjust the table borders in Calibration: hand:TL cuts tiles\./,
  );
  await page.getByRole("tab", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Calibration", exact: true }).click();
  const canvas = page.getByLabel("Camera regions on the table preview");
  await expect(canvas).toHaveAttribute("width", "1000");
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  if (!box) throw Error("Missing calibration canvas");
  await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.15);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.17, box.y + box.height * 0.15);
  await page.mouse.up();
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(page.getByText("Saved.", { exact: true })).toBeVisible();
  expect(saved?.hand?.TL.rect[0]).toBeGreaterThan(100);
  expect(prepares).toBe(0);
  await page
    .getByRole("button", { name: "Back to settings", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Prepare recording", exact: true })
    .click();
  await expect.poll(() => prepares).toBe(1);
  expect(errors).toEqual([]);
});
