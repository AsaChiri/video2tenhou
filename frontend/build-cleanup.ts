import { readdirSync, unlinkSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

// Only generated bundle files are removed. Tile art is never a build output.
export function cleanBundles() {
  const directory = fileURLToPath(
    new URL("../src/video2tenhou/tool/static/assets/", import.meta.url),
  );
  try {
    for (const file of readdirSync(directory)) {
      if (/^[A-Za-z0-9_-]+\.(?:js|css)$/.test(file))
        unlinkSync(join(directory, file));
    }
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
      throw error;
  }
}
