import type { Match, ScoredEntry } from '../results.js'
import { bestDistance } from './bestDistance.js'
import { pushHeap, replaceHeapRoot } from './heap.js'
import type { RawPreparedScore, StoredItem } from './types.js'

function worse<T, K>(left: ScoredEntry<T, K>, right: ScoredEntry<T, K>): boolean {
  return (
    left.score > right.score || (left.score === right.score && left.order > right.order)
  )
}

function result<T, K>(entries: ScoredEntry<T, K>[]): readonly Match<T, K>[] {
  entries.sort((a, b) => a.score - b.score || a.order - b.order)
  return entries.map(({ item, key, score }) => ({ item, key, score }))
}

export function topDistance<T, K>(
  items: readonly StoredItem<T, K>[],
  score: RawPreparedScore,
  threshold: number | null,
  limit: number | null,
  optimal: number | null,
): readonly Match<T, K>[] {
  if (limit === 0) return []
  if (limit === 1) {
    const found = bestDistance(items, score, threshold, optimal)
    return found === undefined ? [] : [found]
  }

  if (limit !== null) {
    const heap: ScoredEntry<T, K>[] = []
    let cutoff = threshold
    let order = 0
    for (let index = 0; index < items.length; index++) {
      const entry = items[index]
      const value = score(entry.prepared, cutoff)
      if (threshold !== null && value > threshold) {
        order++
        continue
      }
      if (heap.length < limit) {
        const candidate = { item: entry.item, key: entry.key, score: value, order }
        pushHeap(heap, candidate, worse)
        if (heap.length === limit) {
          cutoff = heap[0].score
          if (optimal !== null && cutoff === optimal) break
        }
      } else if (value < heap[0].score) {
        const candidate = { item: entry.item, key: entry.key, score: value, order }
        replaceHeapRoot(heap, candidate, worse)
        cutoff = heap[0].score
        if (optimal !== null && cutoff === optimal) break
      }
      order++
    }
    return result(heap)
  }

  const results: ScoredEntry<T, K>[] = []
  let order = 0
  for (let index = 0; index < items.length; index++) {
    const entry = items[index]
    const value = score(entry.prepared, threshold)
    if (threshold === null || value <= threshold) {
      results.push({ item: entry.item, key: entry.key, score: value, order })
    }
    order++
  }
  return result(results)
}
