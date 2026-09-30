import { afterEach, vi } from "vitest";
import { enableAutoUnmount } from "@vue/test-utils";
enableAutoUnmount(afterEach);
HTMLMediaElement.prototype.pause = vi.fn();
HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
