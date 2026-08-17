import { convElement, isUnmatchableElement } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { groupFor, type CompiledElementWeights } from '../ngram/weightedProfile.js'

/** One element of a sequence, kept beside the canonical value that decides equality. */
export interface Occurrence {
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
export function occurrencesOf(
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

/** The canonical values alone, in occurrence order, for the profile builders. */
export function canonicalElements(occurrences: readonly Occurrence[]): unknown[] {
  const elements = new Array<unknown>(occurrences.length)
  for (let index = 0; index < occurrences.length; index++) {
    elements[index] = occurrences[index].canonical
  }
  return elements
}

/**
 * What an element scorer may be handed, or `null` where it may not.
 *
 * Only a string qualifies, for two independent reasons. `convElement` maps every
 * single code point — astral included — to a number, so a one-character token is
 * not a string here to begin with. And an array-valued token would stay the
 * caller's own object: `snapshotSequence` copies the outer sequence only, so a
 * scorer reading the token's *contents* would see later mutations through a
 * prepared choice. A string cannot be mutated, so the hazard does not arise.
 */
export function fuzzyOperand(canonical: unknown): string | null {
  return typeof canonical === 'string' ? canonical : null
}

/** One distinct element of a sequence, with how many occurrences carry it. */
export interface ElementEntry {
  readonly canonical: unknown
  readonly operand: string | null
  readonly count: number
  readonly weight: number
}

/**
 * The distinct elements a pair can match on, in first-occurrence order, beside
 * the mass that can never be matched at all.
 *
 * Two properties are load-bearing for soft matching. `entries` applies exactly
 * the exclusions the exact pairing applies — weightless and unmatchable
 * occurrences — so the exact and fuzzy phases cannot disagree about who is
 * eligible. And `unmatchableMass` is not optional: both sides' totals already
 * count a `NaN` occurrence, so a residual fold that walked only `entries` would
 * silently drop that penalty.
 */
export interface ElementTable {
  readonly entries: readonly ElementEntry[]
  readonly indexOf: ReadonlyMap<unknown, number>
  readonly unmatchableMass: number
}

interface CountingEntry extends Omit<ElementEntry, 'count'> {
  count: number
}

export function elementTableOf(occurrences: readonly Occurrence[]): ElementTable {
  const entries: CountingEntry[] = []
  const indexOf = new Map<unknown, number>()
  let unmatchableMass = 0
  for (const occurrence of occurrences) {
    if (occurrence.weight === 0) continue
    if (isUnmatchableElement(occurrence.canonical)) {
      unmatchableMass += occurrence.weight
      continue
    }
    const at = indexOf.get(occurrence.canonical)
    if (at === undefined) {
      indexOf.set(occurrence.canonical, entries.length)
      entries.push({
        canonical: occurrence.canonical,
        operand: fuzzyOperand(occurrence.canonical),
        count: 1,
        weight: occurrence.weight,
      })
      continue
    }
    entries[at].count++
  }
  return { entries, indexOf, unmatchableMass }
}
