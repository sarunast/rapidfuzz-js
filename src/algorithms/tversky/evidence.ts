import { isUnmatchableElement } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import {
  weightedComponents,
  weightedProfile,
  weightedQueryGroups,
  zeroMassSimilarity,
  type CompiledElementWeights,
} from '../ngram/weightedProfile.js'
import { weightedTverskyScore } from '../ngram/weightedTverskyScore.js'
import type { CompiledElementSimilarity } from './elementSimilarity.js'
import {
  canonicalElements,
  elementTableOf,
  occurrencesOf,
  type ElementTable,
  type Occurrence,
} from './occurrences.js'
import { tverskyScore } from './score.js'
import { softComponentsOf, softTablesOf, type SoftComponents } from './soft.js'

/**
 * A paired occurrence, and what the pair contributed.
 *
 * Two occurrences are paired either because they are the **same element** — the
 * ordinary case, and the only one without `elementSimilarity` — or because an
 * element scorer found them alike enough to share part of their mass. `exact`
 * says which, and it is the only reliable way to ask: an element scorer may
 * return `1` for two elements that are not equal.
 *
 * `first` and `second` are the values the caller passed — for a string operand,
 * the character at that code point. They can differ even on an exact pair:
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
  /**
   * Whether the two are the same element, rather than two elements an element
   * scorer found alike. Always `true` without `elementSimilarity`.
   */
  readonly exact: boolean
  /** How alike the two elements are, `0..1`. Always `1` on an exact pair. */
  readonly similarity: number
  /** The first occurrence's effective weight, on the scorer's normalized scale. */
  readonly firstWeight: number
  /** The second occurrence's effective weight. Equal to `firstWeight` on an exact pair. */
  readonly secondWeight: number
  /** What the pair contributed to the overlap: `min(firstWeight, secondWeight) × similarity`. */
  readonly sharedMass: number
  /** What the first occurrence still owes. Always `0` on an exact pair. */
  readonly firstUnmatchedMass: number
  /** What the second occurrence still owes. Always `0` on an exact pair. */
  readonly secondUnmatchedMass: number
}

/**
 * One element occurrence left with no partner at all.
 *
 * Occurrences are paired in input order — the earliest unmatched occurrence of
 * an element on one side takes the earliest unmatched occurrence on the other —
 * so `['react', 'react']` against `['react']` leaves the *second* `react`
 * unmatched, not the first.
 *
 * With `elementSimilarity`, "no partner" is stricter than "the other side does
 * not hold it". An occurrence an element scorer paired across is in `matches`
 * with a positive `firstUnmatchedMass` or `secondUnmatchedMass`, even though the
 * other sequence holds no such element — so a partly matched occurrence is never
 * here.
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
 * arithmetic the scorer uses; where the shared mass is provably zero — a side
 * carrying no weight at all can share nothing, since a weight belongs to an
 * element rather than to a side — each unmatched mass is exactly that side's own
 * folded mass.
 *
 * The consequence is that exact algebraic identities between them are not
 * guaranteed: `firstMass` may differ from `sharedMass + firstUnmatchedMass` in
 * the last bit once weights span a wide range. Only these totals are
 * authoritative — the per-occurrence masses on {@link TverskyEvidenceMatch} are
 * for reading, not for re-deriving.
 *
 * `elementSimilarity` widens that gap, and adds a second one. Once any pair is
 * matched across, the three components are folded per element and per matched
 * pair rather than per weight group, while `firstMass` and `secondMass` keep
 * their group fold — so the two orders disagree in the last bit even before any
 * fuzzy mass is added. A configuration whose element scorer pairs nothing is
 * therefore bit-identical to the exact scorer, and the first pair it does match
 * can move the score by a last-bit step that the added mass alone does not
 * explain.
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
  /** First-sequence occurrences left with no partner at all, ascending by index. */
  readonly unmatchedFirst: readonly TverskyUnmatchedElement[]
  /** Second-sequence occurrences left with no partner at all, ascending by index. */
  readonly unmatchedSecond: readonly TverskyUnmatchedElement[]
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
    exact: true,
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
 * The unmatched occurrences behind each element table entry, in ascending
 * index, with a cursor — plus the ones no entry can hold.
 *
 * Keyed by entry index rather than by canonical value so that reading a queue
 * for an edge needs no lookup that could miss. `loose` is where the occurrences
 * the element table excludes end up: an unmatchable element, or one weighing
 * nothing. They can never be paired, and stay unmatched throughout.
 */
interface UnmatchedQueues {
  readonly queues: readonly { items: TverskyUnmatchedElement[]; next: number }[]
  readonly loose: TverskyUnmatchedElement[]
}

function queuesOf(
  unmatched: TverskyUnmatchedElement[],
  occurrences: Occurrence[],
  table: ElementTable,
): UnmatchedQueues {
  const queues = table.entries.map(() => {
    const items: TverskyUnmatchedElement[] = []
    return { items, next: 0 }
  })
  const loose: TverskyUnmatchedElement[] = []
  for (const one of unmatched) {
    const at = table.indexOf.get(occurrences[one.index].canonical)
    if (at === undefined) loose.push(one)
    else queues[at].items.push(one)
  }
  return { queues, loose }
}

function remaining(held: UnmatchedQueues): TverskyUnmatchedElement[] {
  const rest = [...held.loose]
  for (const queue of held.queues) {
    for (let at = queue.next; at < queue.items.length; at++) rest.push(queue.items[at])
  }
  return rest.sort((one, other) => one.index - other.index)
}

/**
 * Expands the solved element pairs back into occurrence rows.
 *
 * The masses are **not** recomputed here — every total came from the fold in
 * `soft.ts`, which works per element pair. Row expansion only decides which
 * occurrences to name, so it cannot move a number. Which ones it names follows
 * the rule the exact pairing already set: the earliest occurrences were
 * reserved by exact matching, so the ones left over are the later ones.
 */
function softPairing(
  pairing: Pairing,
  first: Occurrence[],
  second: Occurrence[],
  tables: ExplainedTables,
  components: SoftComponents,
): Pairing {
  const firstQueues = queuesOf(pairing.unmatchedFirst, first, tables.first)
  const secondQueues = queuesOf(pairing.unmatchedSecond, second, tables.second)
  const matches = [...pairing.matches]
  for (let at = 0; at < components.edges.length; at++) {
    const edge = components.edges[at]
    const rows = firstQueues.queues[edge.first]
    const columns = secondQueues.queues[edge.second]
    for (let taken = 0; taken < components.units[at]; taken++) {
      const row = rows.items[rows.next++]
      const column = columns.items[columns.next++]
      matches.push({
        first: first[row.index].raw,
        second: second[column.index].raw,
        firstIndex: row.index,
        secondIndex: column.index,
        exact: false,
        similarity: edge.similarity,
        firstWeight: row.weight,
        secondWeight: column.weight,
        sharedMass: edge.profit,
        firstUnmatchedMass: row.weight - edge.profit,
        secondUnmatchedMass: column.weight - edge.profit,
      })
    }
  }
  return {
    matches: matches.sort((one, other) => one.firstIndex - other.firstIndex),
    unmatchedFirst: remaining(firstQueues),
    unmatchedSecond: remaining(secondQueues),
  }
}

/**
 * Builds the cold explainer a `gramSize: 1` Tversky scorer carries.
 *
 * It recomputes the pair from scratch and retains nothing between calls, which
 * is the whole trade: search decides *which* candidate, and this answers *why*
 * for the handful search already chose.
 *
 * A soft configuration runs the exact explanation first and keeps it untouched
 * whenever the fuzzy phase adds nothing, which is what makes an unreachable
 * threshold bit-identical to no `elementSimilarity` at all.
 */
export function tverskyExplainer(
  direction: 'distance' | 'similarity',
  weights: CompiledElementWeights | null,
  alpha: number,
  beta: number,
  soft: CompiledElementSimilarity | null = null,
): (first: Sequence, second: Sequence) => TverskyEvidence {
  return (first, second) => {
    const firstOccurrences = occurrencesOf(first, weights)
    const secondOccurrences = occurrencesOf(second, weights)
    const exactPairing = pairOccurrences(firstOccurrences, secondOccurrences)
    const exact =
      weights === null
        ? plainTotals(firstOccurrences, secondOccurrences, exactPairing, alpha, beta)
        : weightedTotals(firstOccurrences, secondOccurrences, weights, alpha, beta)
    const softened =
      soft === null
        ? null
        : softenEvidence(firstOccurrences, secondOccurrences, exact, soft, alpha, beta)
    const components = softened === null ? exact : softened.components
    const pairing =
      softened === null
        ? exactPairing
        : softPairing(
            exactPairing,
            firstOccurrences,
            secondOccurrences,
            softened.tables,
            softened.matching,
          )
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

/**
 * Both sides indexed, which scoring needs of the query alone: an explanation
 * walks the second side's leftovers back to their occurrences too.
 */
interface ExplainedTables {
  readonly first: ElementTable
  readonly second: ElementTable
}

function softenEvidence(
  first: Occurrence[],
  second: Occurrence[],
  exact: Components,
  soft: CompiledElementSimilarity,
  alpha: number,
  beta: number,
): { components: Components; tables: ExplainedTables; matching: SoftComponents } | null {
  const indexed: ExplainedTables = {
    first: elementTableOf(first),
    second: elementTableOf(second),
  }
  const tables = softTablesOf(indexed.first, indexed.second)
  const matching = softComponentsOf(tables, soft, exact.totals.sharedMass, null)
  if (matching === null) return null
  return {
    tables: indexed,
    matching,
    components: {
      totals: {
        firstMass: exact.totals.firstMass,
        secondMass: exact.totals.secondMass,
        sharedMass: matching.shared,
        firstUnmatchedMass: matching.firstOnly,
        secondUnmatchedMass: matching.secondOnly,
      },
      similarity: weightedTverskyScore(
        matching.shared,
        matching.firstOnly,
        matching.secondOnly,
        alpha,
        beta,
      ),
    },
  }
}
