<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
const props = defineProps({ job: { type: Object, required: true } });
const output = ref<HTMLElement | null>(null),
  open = ref(false),
  follow = ref(true);
watch(
  () => props.job.started,
  () => {
    follow.value = true;
  },
);
watch([() => props.job.log?.join("\n"), open], async () => {
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
    <pre ref="output" class="log" @scroll="scroll">{{
      job.log?.join("\n")
    }}</pre>
  </details>
</template>
