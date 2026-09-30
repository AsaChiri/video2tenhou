import type {
  ReviewItem,
  ReviewChoice,
  HandEntry,
  FactBody,
  Decision,
} from "../types";

export function isAnswered(
  it: ReviewItem,
  hands: Partial<HandEntry>[],
  facts: Partial<Record<number, FactBody[]>>,
): boolean {
  // only a fact saved after the last decode of the hand answers a question: a fact the decoder already had and
  // still asks about did not settle it (the decoder notes say why)
  const cutoff = hands.find((hand) => hand.hand === it.hand)?.decoded_at || 0;
  const fs = (facts[it.hand] || []).filter((f) => (f.ts ?? Infinity) > cutoff);
  const near = (f: FactBody) =>
    f.seat === it.seat &&
    (it.j !== undefined && f.j !== undefined
      ? f.j === it.j
      : Math.abs((f.t ?? -1e9) - (it.t ?? 1e9)) < 3);
  switch (it.kind) {
    case "uncertain_tiles":
      return (it.choices || []).every((choice) =>
        choiceAnswered(it, choice, hands, facts),
      );
    case "draw":
      return fs.some(
        (f) =>
          (f.kind === "draw" || (f.kind === "lost" && !f.field)) && near(f),
      );
    case "discard":
      return fs.some((f) => f.kind === "discard" && near(f));
    case "ura":
      return fs.some((f) => f.kind === "ura" && (f.tiles || []).length);
    case "dora":
      return fs.some(
        (f) =>
          (f.kind === "dora" && (f.tiles || []).length) ||
          (f.kind === "lost" && f.field === "dora"),
      );
    case "riichi":
      return fs.some(
        (f) =>
          (f.kind === "riichi_turn" && near(f)) ||
          (f.kind === "riichi" && !f.seats?.includes(it.seat || "")),
      );
    case "result":
      return fs.some(
        (f) =>
          (f.kind === "final_hand" && f.seat === it.seat) ||
          f.kind === "site_wrong",
      );
    case "kan":
      return fs.some(
        (f) =>
          (f.kind === "meld" &&
            f.seat === it.seat &&
            Math.abs((f.t ?? -1e9) - (it.t ?? 1e9)) < 40) ||
          (f.kind === "kan_time" && (!it.seat || f.seat === it.seat)),
      );
    case "call":
      return (
        fs.some(
          (f) =>
            (f.kind === "meld" || f.kind === "meld_remove") &&
            Math.abs((f.t ?? -1e9) - (it.t ?? 1e9)) < 40,
        ) || fs.some((f) => f.kind === "note" && f.text === it.text)
      );
    case "haipai":
      return fs.some(
        (f) =>
          (f.kind === "haipai" ||
            (f.kind === "lost" && f.field === "haipai")) &&
          f.seat === it.seat,
      );
    case "conflict":
      return fs.some(
        (f) =>
          ((f.kind === "meld" || f.kind === "meld_remove") &&
            Math.abs((f.t ?? -1e9) - (it.t ?? 1e9)) < 40) ||
          (f.kind === "final_hand" && f.seat === it.seat) ||
          (f.kind === "riichi_turn" && f.seat === it.seat) ||
          f.kind === "discard" ||
          (f.kind === "note" && f.text === it.text),
      );
    case "order":
      return fs.some(
        (f) =>
          f.kind === "missing_discard" ||
          (f.kind === "note" && f.text === it.text),
      );
    default:
      return fs.some((f) => f.kind === "note" && f.text === it.text);
  }
}
export function choiceAnswered(
  group: ReviewItem,
  choice: ReviewChoice,
  hands: Partial<HandEntry>[],
  facts: Partial<Record<number, FactBody[]>>,
): boolean {
  return isAnswered(
    { ...choice, hand: group.hand, kind: choice.field || choice.kind || "" },
    hands,
    facts,
  );
}
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
function orderedChoices(item: ReviewItem) {
  // Sort a view, retaining the original choice index used by answer handlers.
  return (item.choices || [])
    .map((choice, i) => ({ choice, i }))
    .sort((a, b) => compareDecisions(a.choice, b.choice));
}
export function reviewDecisions(
  items: ReviewItem[],
  hands: Partial<HandEntry>[],
  facts: Partial<Record<number, FactBody[]>>,
  hand: number | null = null,
) {
  const decisions: Decision[] = [];
  for (const item of items) {
    if (item.kind === "solver_incomplete") continue;
    if (isAnswered(item, hands, facts) || (hand !== null && item.hand !== hand))
      continue;
    if (item.kind === "uncertain_tiles") {
      for (const { choice, i } of orderedChoices(item)) {
        if (!choiceAnswered(item, choice, hands, facts))
          decisions.push({ item, choice: i, decision: choice });
      }
    } else decisions.push({ item, choice: null, decision: item });
  }
  // Stable sorting preserves source order when confidence is tied or unmeasured.
  return decisions.sort((a, b) => compareDecisions(a.decision, b.decision));
}
