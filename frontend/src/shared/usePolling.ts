import { onMounted, onBeforeUnmount } from "vue";

// Schedule after completion so a slow request can never overlap the next poll.
export function usePolling(
  callback: () => void | Promise<void>,
  delay: number,
  immediate = true,
) {
  let timer: ReturnType<typeof setTimeout> | undefined,
    disposed = false;
  async function tick() {
    try {
      await callback();
    } finally {
      if (!disposed) timer = setTimeout(tick, delay);
    }
  }
  onMounted(() => {
    if (immediate) tick();
    else timer = setTimeout(tick, delay);
  });
  onBeforeUnmount(() => {
    disposed = true;
    clearTimeout(timer);
  });
}
