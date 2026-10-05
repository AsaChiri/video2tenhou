import { ref } from "vue";

export function errorText(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}

/**
 * Run one user action at a time, exposing whether it is running and its error.
 * `report` additionally receives each failure message (e.g. to emit it).
 */
export function useAction(report?: (message: string) => void) {
  const busy = ref(false),
    error = ref("");
  async function run<T>(action: () => Promise<T>): Promise<T | undefined> {
    if (busy.value) return undefined;
    busy.value = true;
    error.value = "";
    try {
      return await action();
    } catch (failure) {
      error.value = errorText(failure);
      report?.(error.value);
      return undefined;
    } finally {
      busy.value = false;
    }
  }
  return { busy, error, run };
}
