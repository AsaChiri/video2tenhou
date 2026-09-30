import { createServer as createHttpServer } from "node:http";
import { createServer } from "vite";
import { expect, test } from "vitest";
import { backendProxy } from "../vite.config";

test("development proxy preserves the backend's same-origin write checks", async () => {
  const backend = createHttpServer((request, response) => {
    const allowed =
      request.headers.origin === `http://${request.headers.host}` &&
      request.headers["x-video2tenhou"] === "1";
    response.writeHead(allowed ? 200 : 403);
    response.end();
  });
  await new Promise<void>((resolve) => backend.listen(0, "127.0.0.1", resolve));
  const address = backend.address();
  if (!address || typeof address === "string")
    throw new Error("No backend port");
  const server = await createServer({
    configFile: false,
    server: {
      host: "127.0.0.1",
      port: 0,
      proxy: {
        "/api": {
          ...backendProxy,
          target: `http://127.0.0.1:${address.port}`,
        },
      },
    },
  });
  try {
    await server.listen();
    const url = server.resolvedUrls?.local[0];
    if (!url) throw new Error("No development URL");
    for (const [origin, appHeader, expected] of [
      [new URL(url).origin, "1", 200],
      ["https://foreign.example", "1", 403],
      [new URL(url).origin, "0", 403],
    ] as const) {
      const response = await fetch(`${url}api/projects`, {
        method: "POST",
        headers: { Origin: origin, "X-Video2Tenhou": appHeader },
      });
      expect(response.status).toBe(expected);
    }
  } finally {
    await server.close();
    await new Promise<void>((resolve, reject) =>
      backend.close((error) => (error ? reject(error) : resolve())),
    );
  }
});
