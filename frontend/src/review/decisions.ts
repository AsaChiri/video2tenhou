import type { Decision, ReviewChoice, ReviewItem } from "../types";

/** Identify a question across reloads: its hand and the server's question id. */
export const decisionKey = (item: ReviewItem, choice: ReviewChoice | null) =>
  `${item.hand}:${(choice ?? item).id}`;

// Only certified reconstruction margins rank measured decisions. A feasible
// alternative's gap and the reader's probability are different quantities.
function decisionRank(decision: ReviewChoice) {
  if (decision.kind === "conflict") return [0, 0];
  if (decision.lost === true) return [1, 0];
  if (typeof decision.margin === "number" && Number.isFinite(decision.margin))
    return [2, decision.margin];
  return [3, 0];
}
function compareDecisions(a: ReviewChoice, b: ReviewChoice) {
  const x = decisionRank(a),
    y = decisionRank(b);
  return (
    x[0] - y[0] ||
    x[1] - y[1] ||
    Number((b.field || b.kind) === "haipai") -
      Number((a.field || a.kind) === "haipai")
  );
}

/**
 * Order every open question, most uncertain first; a grouped item contributes
 * one question per choice. Answered questions are not filtered here: a hand
 * with new answers is updated, and its next reconstruction asks again only
 * what remains open.
 */
export function reviewDecisions(items: ReviewItem[]) {
  const decisions: Decision[] = [];
  for (const item of items) {
    if (item.kind === "uncertain_tiles")
      for (const choice of item.choices || [])
        decisions.push({ key: decisionKey(item, choice), item, choice });
    else decisions.push({ key: decisionKey(item, null), item, choice: null });
  }
  // Stable sorting preserves source order when confidence is tied or unmeasured.
  return decisions.sort((a, b) =>
    compareDecisions(a.choice ?? a.item, b.choice ?? b.item),
  );
}
