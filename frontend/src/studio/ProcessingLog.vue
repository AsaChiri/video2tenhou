<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import type { PropType } from "vue";
import type { ProjectJob } from "../types";
import { api } from "../shared/api";
import { errorText } from "../shared/useAction";
// The developer log is fetched only while it is open.
const props = defineProps({
  projectId: { type: String, required: true },
  job: { type: Object as PropType<ProjectJob>, required: true },
  lines: { type: Number, default: 0 },
});
const output = ref<HTMLElement | null>(null),
  open = ref(false),
  follow = ref(true),
  log = ref<string[]>([]);
watch(
  () => props.job.started,
  () => {
    follow.value = true;
  },
);
watch(
  [open, () => props.lines, () => props.job.started, () => props.job.running],
  async (_, __, cleanup) => {
    let current = true;
    cleanup(() => (current = false));
    if (!open.value) return;
    try {
      const result = await api<{ log: string[] }>(
        `/api/projects/${props.projectId}/log`,
      );
      if (current) log.value = result.log;
    } catch (failure) {
      if (current) log.value = [errorText(failure)];
    }
  },
);
watch([log, open], async () => {
  await nextTick();
  if (open.value && follow.value && output.value)
    output.value.scrollTop = output.value.scrollHeight;
});
function scroll() {
  const element = output.value;
  if (!element) return;
  follow.value =
    element.scrollHeight - element.clientHeight - element.scrollTop <= 4;
}
</script>
<template>
  <details
    :open="open"
    @toggle="open = ($event.target as HTMLDetailsElement).open"
  >
    <summary>Processing log</summary>
    <pre ref="output" class="log" @scroll="scroll">{{ log.join("\n") }}</pre>
  </details>
</template>
