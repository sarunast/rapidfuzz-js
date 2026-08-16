import type { SealedIndex } from './builder.js'
import { type QueryState, reachesDenseList } from './query.js'

export function assertSharedAccumulatorExact(gramCount: number): void {
  if (gramCount > 0x7fff_ffff) {
    throw new RangeError('a query of more than 2147483647 grams cannot be indexed')
  }
}

/**
 * Accumulates `Σ min(query frequency, choice frequency)` per candidate — the
 * shared-gram count Dice and Tversky both score from. A dense list stores its
 * exceptions, so its members contribute through `state.base` and the stored
 * entries correct the difference.
 */
export function accumulateSharedFrequency(
  sealed: SealedIndex<Float64Array | null>,
  state: QueryState,
  accumulator: Int32Array,
): void {
  const postings = sealed.postings
  const touched = state.touched
  const keys = state.keys
  const queryCounts = state.counts
  const ids = postings.ids
  const postingCounts = postings.counts
  const offsets = postings.offsets
  const dense = postings.dense
  state.base = 0
  if (dense !== null && reachesDenseList(postings, dense, keys)) state.scannedAll = true
  const tracking = !state.scannedAll
  for (let index = 0; index < keys.length; index++) {
    const ordinal = postings.ordinals.get(keys[index])
    if (ordinal === undefined) continue
    const queryCount = queryCounts[index]
    const from = offsets[ordinal]
    const upto = offsets[ordinal + 1]
    if (dense !== null && dense[ordinal] === 1) {
      state.base += 1
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) accumulator[ids[at]] -= 1
        continue
      }
      for (let at = from; at < upto; at++) {
        const count = postingCounts[at]
        accumulator[ids[at]] += (queryCount < count ? queryCount : count) - 1
      }
      continue
    }
    if (!tracking) {
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) accumulator[ids[at]] += 1
        continue
      }
      for (let at = from; at < upto; at++) {
        const count = postingCounts[at]
        accumulator[ids[at]] += queryCount < count ? queryCount : count
      }
      continue
    }
    if (postingCounts === null) {
      for (let at = from; at < upto; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        accumulator[id] += 1
      }
      continue
    }
    for (let at = from; at < upto; at++) {
      const id = ids[at]
      if (accumulator[id] === 0) touched.push(id)
      const count = postingCounts[at]
      accumulator[id] += queryCount < count ? queryCount : count
    }
  }
}
