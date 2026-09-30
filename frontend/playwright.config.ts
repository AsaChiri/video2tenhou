import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  use: {
    baseURL: "http://127.0.0.1:4173",
    channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
  },
  webServer: {
    command: "node node_modules/vite/bin/vite.js preview --host 127.0.0.1",
    url: "http://127.0.0.1:4173",
  },
});
