import type { SelectedChoices } from '../../../../core/scoring/choiceIndex.js'
import { elementsEqual } from '../../sequence.js'
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

export class QueryState {
  readonly keys: (string | number)[] = []
  readonly counts: number[] = []
  readonly touched: number[] = []
  base = 0
  scannedAll = false
  ids: Uint32Array = new Uint32Array(0)
  scores: Float64Array = new Float64Array(0)

  reserve(needed: number): void {
    if (this.ids.length >= needed) return
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
