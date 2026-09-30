import { inject } from "vue";
import type { InjectionKey } from "vue";
import type { useReviewState } from "./useReviewState";
export const reviewKey: InjectionKey<ReturnType<typeof useReviewState>> =
  Symbol("review");
export function useReview() {
  const review = inject(reviewKey);
  if (!review) throw new Error("Review components require a review workspace.");
  return review;
}
