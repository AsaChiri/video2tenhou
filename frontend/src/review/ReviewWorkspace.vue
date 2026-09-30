<script setup lang="ts">
import { computed, onMounted, provide, ref, watch } from "vue";
import type { Job } from "../types";
import { reviewKey } from "./context";
import { useReviewState } from "./useReviewState";
import { usePolling } from "../shared/usePolling";
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
const review = useReviewState(`/review/${props.projectId}`);
const questionCount = computed(() => review.decisions.value.length);
provide(reviewKey, review);
const root = ref<HTMLElement | null>(null);
const now = ref(Date.now());
const elapsed = computed(() => {
  const started = review.job.value.started;
  if (!started) return "";
  const seconds = Math.max(0, Math.floor(now.value / 1000 - started));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
});
usePolling(() => {
  now.value = Date.now();
}, 1000);
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
  if (
    next === "questions" &&
    review.pending.value.length &&
    !review.updating.value
  )
    review.rebuild();
}
function dirty(value: boolean) {
  calibrationDirty.value = value;
  emit("dirty", value);
}
function inspect(value: number | null) {
  hand.value = value;
  go("hands");
}
usePolling(review.poll, 5000, false);
onMounted(async () => {
  try {
    await review.refresh();
    if (review.job.value.running) await review.rebuild("decode_pending", true);
    else {
      const all = await review.api<Job>("decode_all", undefined, {
        jobStatus: true,
      });
      if (all.running) await review.rebuild("decode_all", true);
      else if (
        review.guided.value &&
        review.pending.value.length &&
        !review.error.value
      )
        await review.rebuild();
    }
  } catch (error) {
    review.error.value = error instanceof Error ? error.message : String(error);
  } finally {
    loaded.value = true;
  }
});
</script>
<template>
  <div ref="root" class="review-workspace">
    <header v-if="!calibration" class="review-heading">
      <div>
        <h2>
          {{
            view === "questions"
              ? "Answer uncertain questions"
              : "Advanced review"
          }}
        </h2>
        <p v-if="view === 'questions'">
          Most uncertain first. Answers update in the background; keep reviewing
          other hands while related questions are recalculated.
        </p>
        <p v-else>
          Inspect any hand and every action. Optional for normal review.
        </p>
      </div>
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
    </p>
    <p v-if="review.message.value" role="status">{{ review.message.value }}</p>
    <p
      v-if="review.updates.value && !review.updating.value"
      class="banner update-notice"
    >
      Updated results available.
      <button @click="review.refresh()">Load updates</button>
    </p>
    <template v-if="view === 'questions'">
      <section
        v-if="!loaded || review.requesting.value || review.job.value.running"
        class="review-progress"
        role="status"
      >
        <h3>
          {{
            !loaded
              ? "Loading review…"
              : "Updating answered hands in the background"
          }}
        </h3>
        <template v-if="loaded">
          <p v-if="review.job.value.running">
            You can answer questions from other hands below. Further questions
            from an answered hand wait for its updated result.
          </p>
          <p v-else>Checking the updated record…</p>
          <p
            v-if="review.job.value.hands_total || elapsed"
            class="update-progress"
          >
            <span v-if="review.job.value.hands_total"
              >{{ review.job.value.hands_done || 0 }} of
              {{ review.job.value.hands_total }} hands reconstructed</span
            >
            <span v-if="elapsed">Elapsed {{ elapsed }}</span>
          </p>
          <progress
            v-if="review.job.value.running && review.job.value.hands_total"
            aria-label="Hands reconstructed"
            :value="review.job.value.hands_done || 0"
            :max="review.job.value.hands_total"
          />
          <p
            v-if="
              review.job.value.running &&
              review.job.value.hands_total &&
              review.job.value.hands_done === review.job.value.hands_total
            "
          >
            All selected hands are reconstructed. Finishing the record and
            exports…
          </p>
        </template>
      </section>
      <section
        v-if="
          loaded &&
          review.pending.value.length &&
          !review.job.value.running &&
          !review.requesting.value &&
          review.error.value
        "
        class="review-progress"
      >
        <h3>Your saved answers still need to be applied</h3>
        <p>
          Answers are saved. Retry the update; other hands remain available
          below.
        </p>
        <button class="primary" @click="review.rebuild()">Retry update</button>
      </section>
      <template v-if="loaded && review.selection.value">
        <p v-if="questionCount" class="question-count">
          {{ questionCount }}
          {{ questionCount === 1 ? "question" : "questions" }}
          identified · answers may resolve several
        </p>
        <fieldset class="question-content" :disabled="review.saving.value">
          <QuestionPanel @inspect="inspect" />
        </fieldset>
      </template>
      <section
        v-else-if="loaded && review.skipped.value.length"
        class="review-progress"
      >
        <h3>
          {{ review.skipped.value.length }}
          {{ review.skipped.value.length === 1 ? "question" : "questions" }}
          left for later
        </h3>
        <p>Skipped questions are still unresolved.</p>
        <button class="primary" @click="review.revisit">
          Return to skipped questions
        </button>
      </section>
      <section
        v-else-if="
          loaded &&
          (review.pending.value.length ||
            review.job.value.running ||
            review.requesting.value)
        "
        class="review-progress"
      >
        <p>
          The remaining questions depend on hands being updated. They will
          appear automatically when ready.
        </p>
      </section>
      <section
        v-else-if="loaded && !review.error.value && review.hands.value.length"
        class="review-progress"
      >
        <h3>No more questions</h3>
        <p>
          No unresolved questions remain in the current reconstruction. Open the
          results to check or download the record.
        </p>
        <button class="primary" @click="emit('results')">Open results</button>
      </section>
      <p v-else-if="loaded && !review.error.value">
        Analyze the recording to generate review questions.
      </p>
    </template>
    <template v-else-if="view === 'hands' || view === 'label'">
      <div class="actions">
        <button v-if="view === 'label'" @click="go('hands')">
          Inspect hands
        </button>
        <button
          v-if="review.pending.value.length || review.updating.value"
          :disabled="review.updating.value"
          @click="review.rebuild()"
        >
          {{
            review.updating.value
              ? "Updating the record…"
              : "Apply saved changes"
          }}
        </button>
        <details>
          <summary>Tools</summary>
          <button @click="go('label')">Label tiles for training</button>
          <button
            :disabled="review.updating.value"
            @click="review.rebuild('decode_all')"
          >
            Rebuild all hands
          </button>
        </details>
      </div>
      <HandsPanel
        v-if="view === 'hands'"
        :key="review.revision.value"
        :hand="hand"
        @select="hand = $event"
      />
      <LabelEditor v-else />
    </template>
    <CalibrationEditor v-else-if="view === 'calib'" @dirty="dirty" />
  </div>
</template>
