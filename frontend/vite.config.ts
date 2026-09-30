import { defineConfig } from "vitest/config";
import type { ProxyOptions } from "vite";
import vue from "@vitejs/plugin-vue";
import { cleanBundles } from "./build-cleanup.ts";

const backend = "http://127.0.0.1:8765";
export const backendProxy: ProxyOptions = {
  target: backend,
  changeOrigin: true,
  configure(server) {
    server.on("proxyReq", (outgoing, incoming) => {
      // Translate only same-origin development requests. Foreign origins must
      // still reach the backend unchanged so its write guard rejects them.
      if (incoming.headers.origin === `http://${incoming.headers.host}`)
        outgoing.setHeader("Origin", `http://${outgoing.getHeader("host")}`);
    });
  },
};

export default defineConfig({
  plugins: [
    vue(),
    {
      name: "clean-generated-bundles",
      apply: "build",
      buildStart: cleanBundles,
    },
  ],
  build: {
    outDir: "../src/video2tenhou/tool/static",
    emptyOutDir: false, // Tile artwork and its license are maintained separately.
  },
  server: {
    proxy: Object.fromEntries(
      ["/api", "/review", "/exports", "/tiles"].map((path) => [
        path,
        backendProxy,
      ]),
    ),
  },
  preview: { proxy: {} },
  test: {
    include: ["tests/*.test.ts"],
    environment: "jsdom",
    restoreMocks: true,
    setupFiles: ["./tests/setup.ts"],
  },
});
