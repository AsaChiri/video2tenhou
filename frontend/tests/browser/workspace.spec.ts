import { test, expect } from "@playwright/test";

const project = {
  id: "a".repeat(32),
  name: "Test recording",
  games: [12],
  layout: "pml",
  has_fit: true,
  has_hands: true,
  artifacts: ["g0.json"],
  open_items: 1,
  job: { running: false, started: 1, log: ["Ready"] },
};
const entry = {
  hand: 0,
  game: 0,
  kyoku: 0,
  honba: 0,
  t_start: 10,
  t_end: 100,
  decoded_at: 1,
  corner_wind: { TL: "E", TR: "S", BL: "W", BR: "N" },
};
const decode = {
  turns: [{ i: 0, j: 0, seat: "N", t: 30, t_prev: 20, discard: "1m" }],
  haipai: { N: Array(13).fill("1m") },
  dora: ["2p"],
  result: { outcome: "ryuukyoku" },
  problems: [],
};
test.beforeEach(async ({ page }) => {
  const facts: unknown[] = [];
  let applied = false;
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url()),
      path = url.pathname,
      method = route.request().method();
    let data;
    if (path === "/api/workspace")
      data = { setup: { ready: true }, projects: [project] };
    else if (path.endsWith("/hands")) data = [entry];
    else if (path.endsWith("/items"))
      data = applied
        ? []
        : [
            {
              hand: 0,
              idx: 0,
              kind: "draw",
              seat: "N",
              j: 0,
              t: 30,
              margin: 0,
            },
          ];
    else if (path.endsWith("/facts")) {
      if (method === "POST") {
        data = { ...route.request().postDataJSON(), ts: 10 };
        facts.push(data);
      } else data = facts;
    } else if (path.endsWith("/decode_pending")) {
      if (method === "POST") applied = true;
      data = { running: false, pending: facts.length && !applied ? [0] : [] };
    } else if (path.endsWith("/decode_all")) data = { running: false };
    else if (path.endsWith("/revision")) data = ["initial"];
    else if (path.endsWith("/hand/0")) data = { entry, decode };
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
});
test("one application opens review, saves an answer and shows results without an iframe", async ({
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
  await page.getByRole("button", { name: "Choose 0p", exact: true }).click();
  expect((await saved).postDataJSON()).toMatchObject({
    kind: "draw",
    tile: "0p",
    hand: 0,
    j: 0,
  });
  await expect(
    page.getByRole("heading", { name: "No more questions" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Open results", exact: true }).click();
  await expect(page.getByRole("link", { name: "Open replay" })).toBeVisible();
  expect(errors).toEqual([]);
});
test("the guided queue crosses hands, waits for reconstruction and avoids questions resolved by one answer", async ({
  page,
}, testInfo) => {
  let applied = false,
    running = false,
    finish = false;
  await page.route("**/review/**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/hands"))
      return route.fulfill({ json: [entry, { ...entry, hand: 1, kyoku: 1 }] });
    if (path.endsWith("/items"))
      return route.fulfill({
        json: [
          { hand: 0, idx: 0, kind: "draw", seat: "N", j: 0, t: 30, margin: 1 },
          ...(applied
            ? []
            : [
                {
                  hand: 1,
                  idx: 0,
                  kind: "uncertain_tiles",
                  choices: [
                    { field: "draw", seat: "E", j: 0, t: 31, margin: 0 },
                    { field: "draw", seat: "E", j: 1, t: 41, margin: 0.1 },
                  ],
                },
              ]),
        ],
      });
    if (path.endsWith("/hand/1"))
      return route.fulfill({
        json: { entry: { ...entry, hand: 1, kyoku: 1 }, decode },
      });
    if (path.endsWith("/decode_pending")) {
      if (route.request().method() === "POST") {
        running = true;
        return route.fulfill({ json: { running: true, pending: [1] } });
      }
      if (running && finish) {
        applied = true;
        running = false;
      }
      return route.fulfill({
        json: { running, hands: [1], pending: running ? [1] : [] },
      });
    }
    return route.fallback();
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "East (TL) · Draw 1" }),
  ).toBeVisible();
  await expect(page.getByText(/3 questions identified/)).toBeVisible();
  await expect(page.getByLabel("Hand", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Facts for this hand")).toHaveCount(0);
  await page.screenshot({
    path: testInfo.outputPath("guided-review-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: testInfo.outputPath("guided-review-mobile.png"),
    fullPage: true,
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "Choose 2p", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Updating answered hands in the background",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Choose 2p", exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 1" }),
  ).toBeVisible();
  finish = true;
  await expect(page.getByText(/1 question identified/)).toBeVisible();
  await page.getByRole("button", { name: "Skip for now", exact: true }).click();
  await expect(
    page.getByText("Skipped questions are still unresolved."),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "No more questions" }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Return to skipped questions" })
    .click();
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 1" }),
  ).toBeVisible();
});
test("unresolvable confidence checks do not become questions or block completion", async ({
  page,
}) => {
  const choices = [0, 1, 2].map((j) => ({
    field: "draw",
    seat: "N",
    j,
    t: 30 + j,
    margin: 0,
    value: "1m",
  }));
  await page.route("**/api/items", (route) =>
    route.fulfill({
      json: [
        { hand: 0, idx: 0, kind: "uncertain_tiles", choices: [choices[2]] },
      ],
    }),
  );
  await page.route("**/api/hand/0", (route) =>
    route.fulfill({
      json: {
        entry,
        decode: {
          ...decode,
          confidence: choices.map((choice, j) => ({
            ...choice,
            turn: j,
            margin: j === 0 ? 2 : 0,
            alternative_gap: [null, null, 0.2][j],
            runner_up: j === 1 ? null : "2p",
            state: ["resolved", "unresolvable", "ambiguous"][j],
          })),
        },
      },
    }),
  );
  const retries: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/decode/0"))
      retries.push(request.url());
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(page.getByText(/1 question identified/)).toBeVisible();
  await expect(
    page.getByText(/choices still need confidence verification/),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 3" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Choose 2p", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Search incomplete",
    }),
  ).toHaveCount(0);
  await expect(page.locator(".question-count")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Retry processing", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "No more questions", exact: true }),
  ).toBeVisible();
  expect(retries).toEqual([]);
});
test("advanced review holds the full hand and action inspector outside the question queue", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await page
    .getByRole("button", { name: "Advanced review", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Hands", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "East 1, 0 honba", exact: true })
    .click();
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
  await expect(
    page.getByRole("heading", { name: "Turns", exact: true }),
  ).toHaveCount(0);
});
test("stored processing notices stay off the question queue and appear as a hand note", async ({
  page,
}) => {
  await page.route("**/api/items", (route) =>
    route.fulfill({
      json: [{ hand: 0, idx: 0, kind: "solver_incomplete" }],
    }),
  );
  await page.route("**/api/hand/0", (route) =>
    route.fulfill({
      json: {
        entry,
        decode: {
          ...decode,
          notes: ["Some automatic checks reached their time limit."],
        },
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Search incomplete" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "No more questions" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Retry processing" }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Advanced review", exact: true })
    .click();
  await page
    .getByRole("button", { name: "East 1, 0 honba", exact: true })
    .click();
  await page.getByLabel("Processing notes").click();
  await expect(
    page.getByText("Some automatic checks reached their time limit."),
  ).toBeVisible();
});
test("a running update shows real progress and elapsed time, then returns to questions", async ({
  page,
}) => {
  let finished = false;
  let posts = 0;
  await page.route("**/api/decode_pending", async (route) => {
    if (route.request().method() === "POST") posts++;
    await route.fulfill({
      json: {
        running: !finished,
        pending: finished ? [] : [0, 2, 8],
        started: Date.now() / 1000 - 7 * 60,
        hands_done: finished ? 3 : 1,
        hands_total: 3,
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(page.getByText("1 of 3 hands reconstructed")).toBeVisible();
  await expect(page.getByText(/Elapsed 7:/)).toBeVisible();
  await expect(
    page.getByRole("progressbar", { name: "Hands reconstructed" }),
  ).toHaveAttribute("value", "1");
  finished = true;
  await expect(
    page.getByRole("heading", { name: "North (BR) · Draw 1" }),
  ).toBeVisible();
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
  await expect(page.getByRole("status")).toContainText("Saved.");
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

test("failed preparation opens table calibration in settings and retries after saving", async ({
  page,
}) => {
  const failed = {
    ...project,
    has_fit: false,
    can_calibrate: true,
    artifacts: [],
    job: { running: false, error: "No stable play windows found." },
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
        return route.fulfill({ json: { saved: "pml.json" } });
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
  await expect(
    page.getByText("Saved. Return to Settings", { exact: false }),
  ).toBeVisible();
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
