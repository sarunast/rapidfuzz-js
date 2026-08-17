import { convElement, isUnmatchableElement } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import {
  groupFor,
  weightedComponents,
  weightedProfile,
  weightedQueryGroups,
  zeroMassSimilarity,
  type CompiledElementWeights,
} from '../ngram/weightedProfile.js'
import { weightedTverskyScore } from '../ngram/weightedTverskyScore.js'
import { tverskyScore } from './score.js'

/**
 * One element occurrence that both sequences hold, and what it contributed.
 *
 * `first` and `second` are the values the caller passed — for a string operand,
 * the character at that code point. They can differ while still matching:
 * equality is decided on canonical elements, where `'a'` and `97` are one
 * element and `'😀'` is one element rather than two code units.
 *
 * The trap is that `sharedMass` and the two residuals are **informational**.
 * Summing them across matches is not how `totals` is produced and will not
 * reproduce it bit for bit — see {@link TverskyEvidenceTotals}.
 */
export interface TverskyEvidenceMatch {
  /** The element as the first sequence held it, before canonicalization. */
  readonly first: unknown
  /** The element as the second sequence held it, before canonicalization. */
  readonly second: unknown
  /** Where it sits in the first sequence, counting code points for a string. */
  readonly firstIndex: number
  /** Where it sits in the second sequence, counting code points for a string. */
  readonly secondIndex: number
  /** How alike the two elements are, `0..1`. Always `1` while matching is exact. */
  readonly similarity: number
  /** The first occurrence's effective weight, on the scorer's normalized scale. */
  readonly firstWeight: number
  /** The second occurrence's effective weight. Equal to `firstWeight` while matching is exact. */
  readonly secondWeight: number
  /** What the pair contributed to the overlap: the effective weight, exactly. */
  readonly sharedMass: number
  /** What the first occurrence still owes. Always `0` while matching is exact. */
  readonly firstUnmatchedMass: number
  /** What the second occurrence still owes. Always `0` while matching is exact. */
  readonly secondUnmatchedMass: number
}

/**
 * One element occurrence that only one of the two sequences holds.
 *
 * Occurrences are paired in input order — the earliest unmatched occurrence of
 * an element on one side takes the earliest unmatched occurrence on the other —
 * so `['react', 'react']` against `['react']` leaves the *second* `react`
 * unmatched, not the first.
 */
export interface TverskyUnmatchedElement {
  /** The element as its sequence held it, before canonicalization. */
  readonly element: unknown
  /** Where it sits in its sequence, counting code points for a string. */
  readonly index: number
  /** Its effective weight, on the scorer's normalized scale. */
  readonly weight: number
  /** What it costs its side: the effective weight, exactly. */
  readonly unmatchedMass: number
}

/**
 * The three Tversky components, plus each side's total.
 *
 * **No total is obtained by subtraction**, because `mass − shared` lets a
 * rounded mass absorb the very occurrence the penalty is made of. Each side's
 * mass is folded on its own, and the three components come from the same
 * grouped arithmetic the scorer uses; where the shared mass is provably zero —
 * a side carrying no weight at all can share nothing, since a weight belongs to
 * an element rather than to a side — each unmatched mass is exactly that side's
 * own folded mass.
 *
 * The consequence is that exact algebraic identities between them are not
 * guaranteed: `firstMass` may differ from `sharedMass + firstUnmatchedMass` in
 * the last bit once weights span a wide range. Only these totals are
 * authoritative — the per-occurrence masses on {@link TverskyEvidenceMatch} are
 * for reading, not for re-deriving.
 *
 * Masses are **relative scorer masses**, not the numbers handed to
 * `elementWeights`: a scorer rescales its whole weight table by a power of two
 * when it must, which changes no score because Tversky is invariant to one
 * positive factor. Practical weights such as `5`, `1` and `0.1` come back
 * unchanged. Where no weighting applies — no `elementWeights`, or a uniform
 * positive one, which prices nothing and compiles away — every weight is `1`
 * and every mass is a count.
 */
export interface TverskyEvidenceTotals {
  /** Everything the first sequence brings, matched or not. */
  readonly firstMass: number
  /** Everything the second sequence brings, matched or not. */
  readonly secondMass: number
  /** What both hold in common — the numerator of the Tversky ratio. */
  readonly sharedMass: number
  /** What only the first holds, the part `alpha` prices. */
  readonly firstUnmatchedMass: number
  /** What only the second holds, the part `beta` prices. */
  readonly secondUnmatchedMass: number
}

/**
 * Why one pair scored what it did: the score, the three Tversky components,
 * and every element occurrence behind them.
 *
 * ```ts
 * const company = createScorer(similarity, {
 *   gramSize: 1,
 *   alpha: 1,
 *   beta: 0.1,
 *   elementWeights: new Map([['swisscom', 5], ['ag', 0.1]]),
 * })
 *
 * const evidence = company.explain(['swisscom', 'ag'], ['swisscom'])
 * evidence.score // 0.9803921568627452
 * evidence.matches.length // 1
 * evidence.unmatchedFirst // [{ element: 'ag', index: 1, weight: 0.1, unmatchedMass: 0.1 }]
 * ```
 *
 * `score` is on the scorer's own scale, so a distance scorer reports
 * `1 − similarity` there while `similarity` stays the underlying `0..1` overlap
 * the masses describe.
 *
 * An element weighing `0` appears nowhere: it contributed no overlap and no
 * penalty, and a zero row is noise. That has a consequence worth knowing before
 * it looks like a bug — with every element ignored, both masses are `0`, all
 * three arrays are empty, and the score comes from the **zero-mass rule**
 * instead: `1` when the two are equal as multisets, `0` otherwise, and always
 * `0` when either holds a `NaN`. So empty evidence beside a score of `0` means
 * "nothing was priced, and the two were not identical", which
 * `firstMass === 0 && secondMass === 0` identifies.
 */
export interface TverskyEvidence {
  /** The score on this scorer's scale — a distance scorer reports `1 − similarity`. */
  readonly score: number
  /** The underlying overlap, `0..1`, whichever direction the scorer runs in. */
  readonly similarity: number
  /** The three components and each side's total. */
  readonly totals: TverskyEvidenceTotals
  /** Paired occurrences, ascending by `firstIndex`. */
  readonly matches: readonly TverskyEvidenceMatch[]
  /** Occurrences only the first sequence holds, ascending by index. */
  readonly unmatchedFirst: readonly TverskyUnmatchedElement[]
  /** Occurrences only the second sequence holds, ascending by index. */
  readonly unmatchedSecond: readonly TverskyUnmatchedElement[]
}

interface Occurrence {
  readonly raw: unknown
  readonly canonical: unknown
  readonly index: number
  readonly weight: number
}

/**
 * One entry per element, raw value kept beside the canonical one that decides
 * equality. A string is walked by code point, which is what `convSequence`
 * counts, so an astral character is one occurrence at one index rather than two
 * halves — and the character itself is reported rather than its code point.
 */
function occurrencesOf(
  sequence: Sequence,
  weights: CompiledElementWeights | null,
): Occurrence[] {
  const occurrences: Occurrence[] = []
  let index = 0
  if (typeof sequence === 'string') {
    for (const raw of sequence) {
      occurrences.push(occurrenceOf(raw, convElement(raw), index, weights))
      index++
    }
    return occurrences
  }
  for (; index < sequence.length; index++) {
    const raw = sequence[index]
    occurrences.push(occurrenceOf(raw, convElement(raw), index, weights))
  }
  return occurrences
}

function occurrenceOf(
  raw: unknown,
  canonical: unknown,
  index: number,
  weights: CompiledElementWeights | null,
): Occurrence {
  return {
    raw,
    canonical,
    index,
    weight: weights === null ? 1 : weights.groupWeights[groupFor(weights, canonical)],
  }
}

function unmatchedOf(occurrence: Occurrence): TverskyUnmatchedElement {
  return {
    element: occurrence.raw,
    index: occurrence.index,
    weight: occurrence.weight,
    unmatchedMass: occurrence.weight,
  }
}

function matchOf(first: Occurrence, second: Occurrence): TverskyEvidenceMatch {
  return {
    first: first.raw,
    second: second.raw,
    firstIndex: first.index,
    secondIndex: second.index,
    similarity: 1,
    firstWeight: first.weight,
    secondWeight: second.weight,
    sharedMass: first.weight,
    firstUnmatchedMass: 0,
    secondUnmatchedMass: 0,
  }
}

interface Pairing {
  readonly matches: TverskyEvidenceMatch[]
  readonly unmatchedFirst: TverskyUnmatchedElement[]
  readonly unmatchedSecond: TverskyUnmatchedElement[]
}

/**
 * Pairs occurrences of the same canonical element in input order: the earliest
 * unmatched one on each side, so a repeat that has no partner is always the
 * last of its run.
 *
 * Weightless and unmatchable occurrences never enter the buckets — the first
 * because it contributes nothing either way, the second because `NaN` matches
 * nothing while a `Map` keyed by it would say otherwise.
 */
function pairOccurrences(first: Occurrence[], second: Occurrence[]): Pairing {
  // A cursor per bucket rather than `shift()`, which would make one long run of
  // a repeated element quadratic.
  const available = new Map<unknown, { items: Occurrence[]; next: number }>()
  for (const occurrence of second) {
    if (occurrence.weight === 0 || isUnmatchableElement(occurrence.canonical)) continue
    const bucket = available.get(occurrence.canonical)
    if (bucket === undefined) {
      available.set(occurrence.canonical, { items: [occurrence], next: 0 })
    } else bucket.items.push(occurrence)
  }
  const matches: TverskyEvidenceMatch[] = []
  const unmatchedFirst: TverskyUnmatchedElement[] = []
  const taken = new Uint8Array(second.length)
  for (const occurrence of first) {
    if (occurrence.weight === 0) continue
    const bucket = isUnmatchableElement(occurrence.canonical)
      ? undefined
      : available.get(occurrence.canonical)
    if (bucket === undefined || bucket.next === bucket.items.length) {
      unmatchedFirst.push(unmatchedOf(occurrence))
      continue
    }
    const partner = bucket.items[bucket.next++]
    taken[partner.index] = 1
    matches.push(matchOf(occurrence, partner))
  }
  const unmatchedSecond: TverskyUnmatchedElement[] = []
  for (const occurrence of second) {
    if (occurrence.weight === 0 || taken[occurrence.index] === 1) continue
    unmatchedSecond.push(unmatchedOf(occurrence))
  }
  return { matches, unmatchedFirst, unmatchedSecond }
}

/** The ascending fold of one side's own groups, in `weightedComponents`' order. */
function weightedMass(
  groupIds: Uint32Array,
  groupTotals: Uint32Array,
  weights: CompiledElementWeights,
): number {
  let mass = 0
  for (let at = 0; at < groupIds.length; at++) {
    mass += weights.groupWeights[groupIds[at]] * groupTotals[at]
  }
  return mass
}

interface Components {
  readonly totals: TverskyEvidenceTotals
  readonly similarity: number
}

/**
 * The authoritative arithmetic, taken from the scorer's own engines rather than
 * re-derived from the occurrence walk: `weightedComponents` folds each group's
 * overlap over an exact integer count, which is what keeps every path — direct,
 * prepared, indexed and this one — agreeing bit for bit.
 *
 * A side with no positive group can share nothing, because a weight belongs to
 * an element rather than to a side: an element it holds weighs the same in the
 * other sequence, so a zero there is a zero everywhere. That makes the
 * degenerate branch exact rather than approximate — the shared mass really is
 * `0`, and each side really does owe all of its own.
 */
function weightedTotals(
  first: Occurrence[],
  second: Occurrence[],
  weights: CompiledElementWeights,
  alpha: number,
  beta: number,
): Components {
  const query = weightedQueryGroups(canonicalElements(first), weights)
  const choice = weightedProfile(canonicalElements(second), weights)
  const firstMass = weightedMass(query.groupIds, query.groupTotals, weights)
  const secondMass = weightedMass(choice.groupIds, choice.groupTotals, weights)
  if (query.groupIds.length === 0 || choice.groupIds.length === 0) {
    return {
      totals: {
        firstMass,
        secondMass,
        sharedMass: 0,
        firstUnmatchedMass: firstMass,
        secondUnmatchedMass: secondMass,
      },
      similarity: zeroMassSimilarity(query, choice),
    }
  }
  const parts = new Float64Array(3)
  weightedComponents(query, choice, weights, parts)
  return {
    totals: {
      firstMass,
      secondMass,
      sharedMass: parts[0],
      firstUnmatchedMass: parts[1],
      secondUnmatchedMass: parts[2],
    },
    similarity: weightedTverskyScore(parts[0], parts[1], parts[2], alpha, beta),
  }
}

function canonicalElements(occurrences: Occurrence[]): unknown[] {
  const elements = new Array<unknown>(occurrences.length)
  for (let index = 0; index < occurrences.length; index++) {
    elements[index] = occurrences[index].canonical
  }
  return elements
}

function plainTotals(
  first: Occurrence[],
  second: Occurrence[],
  pairing: Pairing,
  alpha: number,
  beta: number,
): Components {
  const totals: TverskyEvidenceTotals = {
    firstMass: first.length,
    secondMass: second.length,
    sharedMass: pairing.matches.length,
    firstUnmatchedMass: pairing.unmatchedFirst.length,
    secondUnmatchedMass: pairing.unmatchedSecond.length,
  }
  // At `gramSize: 1` a side has no grams exactly when it is empty, so the
  // equality the scorer falls back on can only be empty against empty.
  if (first.length === 0 || second.length === 0) {
    return { totals, similarity: first.length === second.length ? 1 : 0 }
  }
  return {
    totals,
    similarity: tverskyScore(
      pairing.matches.length,
      first.length,
      second.length,
      alpha,
      beta,
    ),
  }
}

/**
 * Builds the cold explainer a `gramSize: 1` Tversky scorer carries.
 *
 * It recomputes the pair from scratch and retains nothing between calls, which
 * is the whole trade: search decides *which* candidate, and this answers *why*
 * for the handful search already chose.
 */
export function tverskyExplainer(
  direction: 'distance' | 'similarity',
  weights: CompiledElementWeights | null,
  alpha: number,
  beta: number,
): (first: Sequence, second: Sequence) => TverskyEvidence {
  return (first, second) => {
    const firstOccurrences = occurrencesOf(first, weights)
    const secondOccurrences = occurrencesOf(second, weights)
    const pairing = pairOccurrences(firstOccurrences, secondOccurrences)
    const components =
      weights === null
        ? plainTotals(firstOccurrences, secondOccurrences, pairing, alpha, beta)
        : weightedTotals(firstOccurrences, secondOccurrences, weights, alpha, beta)
    const similarity = components.similarity
    return {
      score: direction === 'distance' ? 1 - similarity : similarity,
      similarity,
      totals: components.totals,
      matches: pairing.matches,
      unmatchedFirst: pairing.unmatchedFirst,
      unmatchedSecond: pairing.unmatchedSecond,
    }
  }
}
