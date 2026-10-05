<script setup lang="ts">
import { computed, onMounted, provide, ref, watch } from "vue";
import { reviewKey } from "./context";
import { useReviewState } from "./useReviewState";
import { useJob } from "../shared/useJob";
import { errorText } from "../shared/useAction";
import { time } from "./format";
import CalibrationEditor from "./CalibrationEditor.vue";
import QuestionPanel from "./QuestionPanel.vue";
import HandsPanel from "./HandsPanel.vue";
import LabelEditor from "./LabelEditor.vue";
const props = defineProps({
  projectId: { type: String, required: true },
  calibration: Boolean,
  active: { type: Boolean, default: true },
});
const emit = defineEmits(["dirty", "results"]);
const status = useJob();
const review = useReviewState(props.projectId, status);
provide(reviewKey, review);
const root = ref<HTMLElement | null>(null);
const elapsed = computed(() => {
  const job = review.job.value;
  return job?.running ? time(status.now.value / 1000 - job.started) : "";
});
const updatingHands = computed(() => {
  const numbers = review.pending.value.map((hand) => hand + 1);
  return `${numbers.length === 1 ? "hand" : "hands"} ${numbers.join(", ")}`;
});
const view = ref(props.calibration ? "calib" : "questions"),
  hand = ref<number | null>(null),
  calibrationDirty = ref(false),
  loaded = ref(false);
watch([view, () => props.active], () => {
  root.value?.querySelectorAll("video").forEach((video) => video.pause());
});
watch(view, (value) => (review.guided.value = value === "questions"), {
  immediate: true,
  flush: "sync",
});
watch(
  () => props.calibration,
  (value) => {
    view.value = value ? "calib" : "questions";
  },
);
function go(next: string) {
  if (calibrationDirty.value) {
    review.message.value =
      "Save or discard your calibration changes before continuing.";
    return;
  }
  if (review.dirty.value) {
    review.message.value =
      "Save your answer or skip the question before changing views.";
    return;
  }
  view.value = next;
  review.message.value = "";
  if (next === "questions") review.maybeStart();
}
function dirty(value: boolean) {
  calibrationDirty.value = value;
  emit("dirty", value);
}
function inspect(value: number | null) {
  hand.value = value;
  go("hands");
}
onMounted(async () => {
  try {
    await review.refresh();
    review.maybeStart();
  } catch (error) {
    review.error.value = errorText(error);
  } finally {
    loaded.value = true;
  }
});
</script>
<template>
  <div ref="root" class="review-workspace">
    <header v-if="!calibration" class="review-heading">
      <h2>
        {{ view === "questions" ? "Review questions" : "Advanced review" }}
      </h2>
      <button
        v-if="view === 'questions'"
        :disabled="review.saving.value"
        @click="inspect(null)"
      >
        Advanced review
      </button>
      <button v-else :disabled="review.saving.value" @click="go('questions')">
        Back to questions
      </button>
    </header>
    <p v-if="review.error.value" class="banner" role="alert">
      {{ review.error.value }}
      <button
        v-if="review.pending.value.length && !review.busy.value"
        @click="review.rebuild()"
      >
        Retry update
      </button>
    </p>
    <p v-if="review.message.value" role="status">{{ review.message.value }}</p>
    <p
      v-if="review.updates.value && !review.updating.value"
      class="banner update-notice"
    >
      Updated results are available.
      <button @click="review.refresh(true)">Load updates</button>
    </p>
    <template v-if="view === 'questions'">
      <p v-if="!loaded" role="status">Loading review…</p>
      <p
        v-else-if="review.updating.value || review.requesting.value"
        class="review-progress"
        role="status"
      >
        Updating {{ updatingHands }}…
        <span v-if="elapsed" class="update-progress">{{ elapsed }}</span>
      </p>
      <template v-if="loaded && review.selection.value">
        <p class="question-count">
          {{ review.questions.value.length }}
          {{ review.questions.value.length === 1 ? "question" : "questions" }}
        </p>
        <fieldset class="question-content" :disabled="review.saving.value">
          <QuestionPanel />
        </fieldset>
      </template>
      <section
        v-else-if="loaded && review.deferred.value.length"
        class="review-progress"
      >
        <h3>
          {{ review.deferred.value.length }} skipped
          {{ review.deferred.value.length === 1 ? "question" : "questions" }}
        </h3>
        <button class="primary" @click="review.revisit">
          Return to skipped questions
        </button>
      </section>
      <p v-else-if="loaded && review.waiting.value" class="review-progress">
        More questions may follow after updating {{ updatingHands }}.
      </p>
      <section
        v-else-if="loaded && !review.error.value && review.hands.value.length"
        class="review-progress"
      >
        <h3>No more questions</h3>
        <button class="primary" @click="emit('results')">Open results</button>
      </section>
      <p v-else-if="loaded && !review.error.value">
        Analyze the recording to generate review questions.
      </p>
    </template>
    <template v-else-if="view === 'hands' || view === 'label'">
      <div class="actions">
        <button v-if="view === 'label'" @click="go('hands')">All hands</button>
        <button
          v-if="review.pending.value.length || review.updating.value"
          :disabled="review.busy.value || review.requesting.value"
          @click="review.rebuild()"
        >
          {{
            review.updating.value ? "Updating hands…" : "Apply saved changes"
          }}
        </button>
        <details>
          <summary>Tools</summary>
          <button @click="go('label')">Label tiles for training</button>
          <button
            :disabled="review.busy.value || review.requesting.value"
            @click="review.rebuild('all')"
          >
            Rebuild all hands
          </button>
        </details>
      </div>
      <HandsPanel
        v-if="view === 'hands'"
        :hand="hand"
        @select="hand = $event"
      />
      <LabelEditor v-else />
    </template>
    <CalibrationEditor v-else-if="view === 'calib'" @dirty="dirty" />
  </div>
</template>
