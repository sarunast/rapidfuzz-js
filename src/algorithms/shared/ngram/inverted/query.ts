import type { SelectedChoices } from '../../../../core/scoring/choiceIndex.js'
import { elementsEqual } from '../../sequence.js'
import type { Postings, SealedIndex } from './builder.js'

/**
 * The search ranking rule as a predicate: a higher score wins, and a tie goes
 * to the earlier stored position.
 *
 * Load-bearing, and not derivable from the scan order: a dense scan does count
 * upward and each posting list is sorted, but the touched set is filled across
 * several lists, so a gram matching id 9 before another matches id 2 leaves it
 * out of order.
 */
export function outranks(
  score: number,
  id: number,
  otherScore: number,
  otherId: number,
): boolean {
  return score > otherScore || (score === otherScore && id < otherId)
}

/**
 * Whether a score of exactly `0` belongs in the result. When it does, every
 * choice the postings never reached has to be accounted for; when it does not,
 * they can simply vanish.
 */
export function zeroesQualify(threshold: number | null): boolean {
  return threshold === null || threshold <= 0
}

/**
 * What every query of either metric needs: the flattened query, the set of
 * choices accumulation reached, and the arrays a result is written into.
 *
 * Held for the index's lifetime and reused, so none of it is allocated per
 * query. What still is: the `seen` map inside `extractGrams`, a gramless
 * query's matches, and an unlimited call's sorted result — none of them on the
 * path this exists to keep cheap, and query preparation measured a few percent
 * of a query.
 */
export class QueryState {
  readonly keys: (string | number)[] = []
  readonly counts: number[] = []
  readonly touched: number[] = []
  /** What every choice scores before its own accumulator entry is added. */
  base = 0
  /** Set when a dense list has put every choice into the scan. */
  scannedAll = false
  ids: Uint32Array = new Uint32Array(0)
  scores: Float64Array = new Float64Array(0)

  /** Grown before anything is written, never after: this does not preserve. */
  reserve(needed: number): void {
    if (this.ids.length >= needed) return
    this.ids = new Uint32Array(needed)
    this.scores = new Float64Array(needed)
  }
}

/**
 * How many results a call can produce, which is what the result arrays are
 * grown to once, before selection writes anything into them.
 */
export function roomFor(limit: number | null, choiceCount: number): number {
  return limit === null ? choiceCount : limit < choiceCount ? limit : choiceCount
}

/**
 * A gramless query scores `1` against a choice that is gramless and equal, and
 * `0` against everything else — a zero-gram similarity is `1` only when both
 * sides have no grams. So this needs the short choices' elements and nothing
 * else, which is why they are the one thing an index retains besides postings.
 */
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
  // With no zeroes to place, every result scores `1` and the two orders agree:
  // ascending id is also `(score desc, id asc)`, and `matched` is already in it.
  if (!zeroes) {
    for (const id of matched) {
      if (length === room) break
      ids[length] = id
      scores[length] = 1
      length++
    }
    return { ids, scores, length }
  }
  // Otherwise they diverge: ranked order puts every `1` ahead of every `0`,
  // while ascending order interleaves them, so a match at id 7 and a zero at
  // id 3 come back 3 then 7.
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

/**
 * Ids no posting list named, at the score they all share, appended after the
 * ranked results — where they belong, because under a sparse scan every touched
 * choice shares at least one gram and so scores above zero.
 */
export function fillZeroes(
  sealed: SealedIndex<Float64Array | null>,
  state: QueryState,
  accumulator: Int32Array | Float64Array,
  length: number,
  threshold: number | null,
  room: number,
): number {
  if (!zeroesQualify(threshold)) return length
  // Every choice was already scored and offered, so there is nothing to fill —
  // walking for an untouched accumulator entry here would re-add choices the
  // dense base had already put in the result.
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

/**
 * Does this query reach a dense list? If it does, every choice is in play — a
 * default frequency applies to choices no posting entry names — so selection
 * runs over the whole corpus rather than over the touched set.
 */
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

/**
 * An unlimited call's results, ranked by sorting the collected set once.
 *
 * `top` places each qualifying choice by walking the results it already holds,
 * which is what makes a small limit cheap and what makes an unlimited one
 * quadratic: with room for the whole corpus every qualifying choice can shift
 * every earlier one. An unlimited call instead collects in whatever order is
 * cheapest — for a sparse query that is the touched set unordered, since an id
 * sort immediately before a score sort would be paid for nothing — and sorts
 * that set once, `O(k log k)`.
 *
 * The arrays come back freshly allocated rather than borrowed from the query
 * scratch, which is what the collected set already fills.
 */
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

/**
 * The ids a sparse query may still qualify, sorted ascending in place — a copy
 * would allocate per query for an array that exists to be reused.
 *
 * Only under a positive threshold: nothing untouched can clear one, so the walk
 * is confined to what accumulation reached — and that set arrives unordered,
 * because it is filled across several posting lists.
 */
export function sortTouched(state: QueryState): number[] {
  const touched = state.touched
  touched.sort((left, right) => left - right)
  return touched
}
