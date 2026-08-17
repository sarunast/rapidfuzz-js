import { convElement, isUnmatchableElement, MAX_SEQUENCE_LENGTH } from '#core/sequence.js'

import { WEIGHTED_MASS_LIMIT } from './weightedTverskyScore.js'

const WEIGHT_LIMIT = WEIGHTED_MASS_LIMIT / MAX_SEQUENCE_LENGTH

interface WeightMapLike {
  entries(): Iterable<readonly [unknown, unknown]>
  get(key: unknown): unknown
}

/**
 * Element weights compiled into ascending weight groups.
 *
 * Group `0` is reserved for weight `0` whether or not any element has it, so
 * every fold can start at `1` and a configuration of `{1, 3, 5}` cannot end up
 * treating `1` as the ignored group. `groupOf` holds canonical elements, and an
 * element it does not name belongs to `defaultGroup`.
 *
 * `uniformPositive` says the table prices nothing: one positive group, held by
 * the default too, is a constant factor over all three components and cancels
 * from the ratio. It is decided here, once, rather than asked of the table later
 * — the answer needs a walk of every entry, and a scorer is handed its compiled
 * table on every call, so asking measured 27x the cost of a direct score over a
 * 20,000-entry vocabulary holding one ignored element.
 */
export class CompiledElementWeights {
  constructor(
    readonly groupWeights: Float64Array,
    readonly groupOf: ReadonlyMap<unknown, number>,
    readonly defaultGroup: number,
    readonly uniformPositive: boolean,
  ) {}
}

/** One sequence's exact element counts and its occurrences per weight group. */
export class WeightedProfile {
  constructor(
    readonly counts: Map<unknown, number>,
    readonly groupIds: Uint32Array,
    readonly groupTotals: Uint32Array,
    readonly hasUnmatchable: boolean,
  ) {}
}

/**
 * A prepared query's groups, each holding the distinct elements that carry its
 * weight. `zeroElements` is the ignored group, kept because an all-zero-weight
 * query still has to prove multiset equality, while `groupIds` stays positive so
 * that an empty one means exactly "no weighted mass".
 */
export class WeightedQueryGroups {
  constructor(
    readonly groupIds: Uint32Array,
    readonly groupStart: Uint32Array,
    readonly groupTotals: Uint32Array,
    readonly elements: unknown[],
    readonly counts: Uint32Array,
    readonly zeroElements: unknown[],
    readonly zeroCounts: Uint32Array,
    readonly hasUnmatchable: boolean,
  ) {}
}

function isWeightMapLike(value: unknown): value is WeightMapLike {
  return (
    typeof value === 'object' &&
    value !== null &&
    'entries' in value &&
    typeof value.entries === 'function' &&
    'get' in value &&
    typeof value.get === 'function'
  )
}

function validElementWeight(value: unknown, name: string): number {
  if (typeof value !== 'number') throw new TypeError(`${name} must be a number`)
  if (!(Number.isFinite(value) && value >= 0)) {
    throw new RangeError(`${name} has to be a finite non-negative number`)
  }
  return value
}

/**
 * Brings a weight under the limit a longest-sequence mass can hold, by a power
 * of two so that no mantissa changes. Tversky is invariant to scaling every
 * weight by one positive constant, so this changes no score — and a span too
 * wide to survive it is refused rather than silently flushing a weight the
 * caller called important down to nothing.
 */
function scaleWeight(weight: number, scale: number): number {
  if (scale === 1) return weight
  const scaled = weight / scale
  if (weight !== 0 && (scaled === 0 || scaled * scale !== weight)) {
    throw new RangeError(
      'element weights span a range too wide to represent; scale them yourself',
    )
  }
  return scaled
}

export function compileElementWeights(
  rawWeights: unknown,
  rawDefault: unknown,
): CompiledElementWeights {
  const defaultWeight =
    rawDefault === undefined ? 1 : validElementWeight(rawDefault, 'defaultElementWeight')
  const raw = new Map<unknown, number>()
  if (rawWeights !== undefined) {
    if (!isWeightMapLike(rawWeights)) {
      throw new TypeError('elementWeights must be a map from elements to weights')
    }
    for (const entry of rawWeights.entries()) {
      const element = convElement(entry[0])
      const weight = validElementWeight(entry[1], 'an element weight')
      const previous = raw.get(element)
      if (previous !== undefined && previous !== weight) {
        throw new RangeError(
          'elementWeights gives one element two weights: a single-character string and its code point are the same element',
        )
      }
      raw.set(element, weight)
    }
  }
  let maxWeight = defaultWeight
  for (const weight of raw.values()) if (weight > maxWeight) maxWeight = weight
  let scale = 1
  while (maxWeight / scale > WEIGHT_LIMIT) scale *= 2
  const scaled = new Map<unknown, number>()
  const distinct = new Set<number>()
  for (const [element, weight] of raw) {
    const value = scaleWeight(weight, scale)
    scaled.set(element, value)
    if (value !== 0) distinct.add(value)
  }
  const scaledDefault = scaleWeight(defaultWeight, scale)
  if (scaledDefault !== 0) distinct.add(scaledDefault)
  const ascending = [...distinct].sort((left, right) => left - right)
  const groupWeights = new Float64Array(ascending.length + 1)
  const groupIndex = new Map<number, number>()
  for (let at = 0; at < ascending.length; at++) {
    groupWeights[at + 1] = ascending[at]
    groupIndex.set(ascending[at], at + 1)
  }
  const groupOf = new Map<unknown, number>()
  let ignoresAnElement = false
  for (const [element, value] of scaled) {
    const group = groupIndex.get(value)
    if (group === undefined) ignoresAnElement = true
    groupOf.set(element, group === undefined ? 0 : group)
  }
  const defaultIndex = groupIndex.get(scaledDefault)
  return new CompiledElementWeights(
    groupWeights,
    groupOf,
    defaultIndex === undefined ? 0 : defaultIndex,
    ascending.length === 1 && defaultIndex === 1 && !ignoresAnElement,
  )
}

export function groupFor(weights: CompiledElementWeights, element: unknown): number {
  const group = weights.groupOf.get(element)
  return group === undefined ? weights.defaultGroup : group
}

export function weightedProfile(
  elements: ArrayLike<unknown>,
  weights: CompiledElementWeights,
): WeightedProfile {
  const counts = new Map<unknown, number>()
  const totals = new Map<number, number>()
  let hasUnmatchable = false
  for (let index = 0; index < elements.length; index++) {
    const element = elements[index]
    const group = groupFor(weights, element)
    if (group !== 0) totals.set(group, (totals.get(group) ?? 0) + 1)
    if (isUnmatchableElement(element)) {
      hasUnmatchable = true
      continue
    }
    counts.set(element, (counts.get(element) ?? 0) + 1)
  }
  const present = [...totals].sort((left, right) => left[0] - right[0])
  const groupIds = new Uint32Array(present.length)
  const groupTotals = new Uint32Array(present.length)
  for (let at = 0; at < present.length; at++) {
    groupIds[at] = present[at][0]
    groupTotals[at] = present[at][1]
  }
  return new WeightedProfile(counts, groupIds, groupTotals, hasUnmatchable)
}

export function preparedWeightedProfile(value: unknown): WeightedProfile {
  if (!(value instanceof WeightedProfile)) {
    throw new TypeError('invalid prepared weighted profile')
  }
  return value
}

export function weightedQueryGroups(
  elements: ArrayLike<unknown>,
  weights: CompiledElementWeights,
): WeightedQueryGroups {
  const profile = weightedProfile(elements, weights)
  const distinctElements: unknown[] = []
  const distinctCounts: number[] = []
  const distinctGroups: number[] = []
  const zeroElements: unknown[] = []
  const zeroCounts: number[] = []
  for (const [element, count] of profile.counts) {
    const group = groupFor(weights, element)
    if (group === 0) {
      zeroElements.push(element)
      zeroCounts.push(count)
      continue
    }
    distinctElements.push(element)
    distinctCounts.push(count)
    distinctGroups.push(group)
  }
  const order = distinctGroups.map((_, index) => index)
  order.sort((left, right) => distinctGroups[left] - distinctGroups[right])
  const groupIds = profile.groupIds
  const flatElements = new Array<unknown>(order.length)
  const flatCounts = new Uint32Array(order.length)
  for (let at = 0; at < order.length; at++) {
    flatElements[at] = distinctElements[order[at]]
    flatCounts[at] = distinctCounts[order[at]]
  }
  const groupStart = new Uint32Array(groupIds.length + 1)
  let cursor = 0
  for (let at = 0; at < groupIds.length; at++) {
    groupStart[at] = cursor
    while (cursor < order.length && distinctGroups[order[cursor]] === groupIds[at]) {
      cursor++
    }
  }
  groupStart[groupIds.length] = cursor
  return new WeightedQueryGroups(
    groupIds,
    groupStart,
    profile.groupTotals,
    flatElements,
    flatCounts,
    zeroElements,
    Uint32Array.from(zeroCounts),
    profile.hasUnmatchable,
  )
}

/**
 * Writes `shared`, `firstOnly` and `secondOnly` into `parts`, folding the
 * ascending merge of the two sides' weight groups.
 *
 * Ascending and independent, both load-bearing: the per-group counts are exact
 * integers whatever order the postings arrive in, so one canonical fold makes
 * every path — direct, prepared, indexed — agree bit for bit, keeps a
 * permutation of either side scoring the same, and keeps each penalty a sum of
 * real unmatched occurrences rather than the residue of two rounded masses.
 */
export function weightedComponents(
  query: WeightedQueryGroups,
  choice: WeightedProfile,
  weights: CompiledElementWeights,
  parts: Float64Array,
): void {
  const groupWeights = weights.groupWeights
  const queryIds = query.groupIds
  const choiceIds = choice.groupIds
  const queryStart = query.groupStart
  const queryElements = query.elements
  const queryCounts = query.counts
  const choiceCounts = choice.counts
  let shared = 0
  let firstOnly = 0
  let secondOnly = 0
  let qi = 0
  let ci = 0
  while (qi < queryIds.length || ci < choiceIds.length) {
    const inChoice = ci < choiceIds.length
    if (qi < queryIds.length && (!inChoice || queryIds[qi] <= choiceIds[ci])) {
      const group = queryIds[qi]
      const weight = groupWeights[group]
      let sharedCount = 0
      for (let at = queryStart[qi]; at < queryStart[qi + 1]; at++) {
        const choiceCount = choiceCounts.get(queryElements[at])
        if (choiceCount === undefined) continue
        const queryCount = queryCounts[at]
        sharedCount += queryCount < choiceCount ? queryCount : choiceCount
      }
      shared += weight * sharedCount
      firstOnly += weight * (query.groupTotals[qi] - sharedCount)
      if (inChoice && choiceIds[ci] === group) {
        secondOnly += weight * (choice.groupTotals[ci] - sharedCount)
        ci++
      }
      qi++
      continue
    }
    secondOnly += groupWeights[choiceIds[ci]] * choice.groupTotals[ci]
    ci++
  }
  parts[0] = shared
  parts[1] = firstOnly
  parts[2] = secondOnly
}

/**
 * Both sides carry no weighted mass: equal only as multisets, and never when
 * either holds an element that matches nothing.
 */
export function zeroMassSimilarity(
  query: WeightedQueryGroups,
  choice: WeightedProfile,
): number {
  if (query.groupIds.length !== 0 || choice.groupIds.length !== 0) return 0
  if (query.hasUnmatchable || choice.hasUnmatchable) return 0
  const zeroElements = query.zeroElements
  if (choice.counts.size !== zeroElements.length) return 0
  for (let at = 0; at < zeroElements.length; at++) {
    if (choice.counts.get(zeroElements[at]) !== query.zeroCounts[at]) return 0
  }
  return 1
}
