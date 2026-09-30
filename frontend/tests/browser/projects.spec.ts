import { expect, test } from "@playwright/test";

test("project library supports creation, opening, editing, renaming and confirmed deletion", async ({
  page,
}, testInfo) => {
  const project = {
    id: "a".repeat(32),
    name: "week_11",
    display_name: "PML · Week 11",
    source: "D:/recordings/PML_Week_11.mp4",
    games: [21938, 21939],
    layout: "pml",
    start: 0,
    end: null,
    created: 1780000000,
    artifacts: [],
    has_fit: false,
    can_calibrate: false,
    open_items: 0,
    job: { running: false },
  };
  let projects = [project];
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/workspace")
      return route.fulfill({ json: { projects, setup: { ready: true } } });
    if (path.endsWith("/rename")) {
      project.display_name = route.request().postDataJSON().display_name;
      return route.fulfill({ json: project });
    }
    if (path.endsWith("/settings")) {
      project.games = route.request().postDataJSON().games;
      return route.fulfill({ json: project });
    }
    if (path.endsWith("/delete")) {
      projects = [];
      return route.fulfill({ json: { deleted: project.id } });
    }
    if (path === "/api/projects") {
      const body = route.request().postDataJSON();
      Object.assign(project, body, { id: "b".repeat(32) });
      projects = [project];
      return route.fulfill({ status: 201, json: project });
    }
    if (path.endsWith("/prepare")) return route.fulfill({ json: project });
    throw Error(`Unexpected request: ${path}`);
  });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Projects", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "PML · Week 11" })).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("projects-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: testInfo.outputPath("projects-mobile.png"),
    fullPage: true,
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.getByLabel("Find a project").fill("missing");
  await expect(page.getByText(/No projects match/)).toBeVisible();
  await page.getByLabel("Find a project").fill("21938");
  await page.getByRole("button", { name: "Open project", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`#${project.id}$`));
  await page.reload();
  await expect(page.locator(".current-project-name")).toHaveText(
    "PML · Week 11",
  );
  await page.getByRole("button", { name: "All projects" }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("textbox", { name: "Scoremj games" }).fill("21940");
  await page.getByRole("button", { name: "Save settings" }).click();
  await expect(page.getByText("Saved", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "All projects" }).click();
  await page.getByRole("button", { name: "Rename", exact: true }).click();
  await page.getByLabel("Project name", { exact: true }).fill("Final table");
  await page.getByRole("button", { name: "Save name" }).click();
  await expect(page.getByRole("link", { name: "Final table" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("link", { name: "Final table" })).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("link", { name: "Final table" })).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page
    .getByRole("button", { name: "Delete project", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your recordings start here" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New project", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "New project" }),
  ).toBeVisible();
  await page.getByLabel("Project name (optional)").fill("Next recording");
  await page.getByRole("button", { name: "Video URL", exact: true }).click();
  await page
    .getByLabel("Video URL", { exact: true })
    .fill("https://example.com/recording");
  await page.getByLabel("Scoremj games").fill("21941");
  await page
    .getByRole("button", { name: "Prepare recording", exact: true })
    .click();
  await expect(page.locator(".current-project-name")).toHaveText(
    "Next recording",
  );
});
