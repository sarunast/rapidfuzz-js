import type { SelectedChoices } from '#core/scoring/choiceIndex.js'
import { elementsEqual } from '#core/sequence.js'

import type { Postings, SealedIndex } from './builder.js'

export function outranks(
  score: number,
  id: number,
  otherScore: number,
  otherId: number,
): boolean {
  return score > otherScore || (score === otherScore && id < otherId)
}

export function zeroesQualify(threshold: number | null): boolean {
  return threshold === null || threshold <= 0
}

/**
 * The largest result capacity that is always kept for reuse — 786,432 bytes
 * across the two arrays.
 *
 * A capacity above this is still reused while later queries need more than it.
 * The first query needing at most this many slots replaces the oversized pair
 * with exactly the capacity that query asked for, so a `limit: null` search over
 * a large collection stops holding 12 bytes a choice for the life of the index.
 *
 * There is no memo domain to derive the threshold from the way
 * `RETAINED_MASK_WORDS` derives from the largest memoisable pattern. What it
 * buys is that every collection at or below it can never reach the release
 * branch, so their allocation behaviour is exactly what it was.
 */
const RETAINED_RESULT_SLOTS = 1 << 16

export class QueryState {
  readonly keys: (string | number)[] = []
  readonly counts: number[] = []
  readonly touched: number[] = []
  base = 0
  scannedAll = false
  ids: Uint32Array = new Uint32Array(0)
  scores: Float64Array = new Float64Array(0)

  reserve(needed: number): void {
    const held = this.ids.length
    if (
      held >= needed &&
      (held <= RETAINED_RESULT_SLOTS || needed > RETAINED_RESULT_SLOTS)
    ) {
      return
    }
    this.ids = new Uint32Array(needed)
    this.scores = new Float64Array(needed)
  }
}

export function roomFor(limit: number | null, choiceCount: number): number {
  return limit === null ? choiceCount : limit < choiceCount ? limit : choiceCount
}

export function gramlessResult(
  sealed: SealedIndex<Float64Array | null>,
  state: QueryState,
  elements: ArrayLike<unknown>,
  threshold: number | null,
  limit: number | null,
  ascending: boolean,
): SelectedChoices {
  const matched: number[] = []
  if (threshold === null || threshold <= 1) {
    for (const entry of sealed.gramless) {
      if (elementsEqual(elements, entry.elements)) matched.push(entry.id)
    }
  }
  const zeroes = zeroesQualify(threshold)
  const room = roomFor(limit, sealed.choiceCount)
  state.reserve(zeroes ? room : matched.length < room ? matched.length : room)
  const ids = state.ids
  const scores = state.scores
  let length = 0
  if (!zeroes) {
    for (const id of matched) {
      if (length === room) break
      ids[length] = id
      scores[length] = 1
      length++
    }
    return { ids, scores, length }
  }
  if (!ascending) {
    for (const id of matched) {
      if (length === room) break
      ids[length] = id
      scores[length] = 1
      length++
    }
  }
  let next = 0
  for (let id = 0; id < sealed.choiceCount && length < room; id++) {
    const isMatch = next < matched.length && matched[next] === id
    if (isMatch) next++
    if (isMatch && !ascending) continue
    ids[length] = id
    scores[length] = isMatch ? 1 : 0
    length++
  }
  return { ids, scores, length }
}

export function fillZeroes(
  sealed: SealedIndex<Float64Array | null>,
  state: QueryState,
  accumulator: Int32Array | Float64Array,
  length: number,
  threshold: number | null,
  room: number,
): number {
  if (!zeroesQualify(threshold)) return length
  if (state.scannedAll) return length
  const ids = state.ids
  const scores = state.scores
  let filled = length
  for (let id = 0; id < sealed.choiceCount && filled < room; id++) {
    if (accumulator[id] !== 0) continue
    ids[filled] = id
    scores[filled] = 0
    filled++
  }
  return filled
}

export function reachesDenseList(
  postings: Postings,
  dense: Uint8Array,
  keys: readonly (string | number)[],
): boolean {
  for (let index = 0; index < keys.length; index++) {
    const ordinal = postings.ordinals.get(keys[index])
    if (ordinal !== undefined && dense[ordinal] === 1) return true
  }
  return false
}

/** Returns the per-query scratch to its untouched state after an answer. */
export function resetQuery(
  state: QueryState,
  accumulator: Int32Array | Float64Array,
): void {
  const touched = state.touched
  if (state.scannedAll) accumulator.fill(0)
  else
    for (let index = 0; index < touched.length; index++) accumulator[touched[index]] = 0
  touched.length = 0
  state.keys.length = 0
  state.counts.length = 0
  state.scannedAll = false
  state.base = 0
}

export function rankSelected(found: SelectedChoices): SelectedChoices {
  const length = found.length
  const collectedIds = found.ids
  const collectedScores = found.scores
  const order = new Array<number>(length)
  for (let at = 0; at < length; at++) order[at] = at
  order.sort(
    (left, right) =>
      collectedScores[right] - collectedScores[left] ||
      collectedIds[left] - collectedIds[right],
  )
  const ids = new Uint32Array(length)
  const scores = new Float64Array(length)
  for (let at = 0; at < length; at++) {
    const from = order[at]
    ids[at] = collectedIds[from]
    scores[at] = collectedScores[from]
  }
  return { ids, scores, length }
}

export function sortTouched(state: QueryState): number[] {
  const touched = state.touched
  touched.sort((left, right) => left - right)
  return touched
}
