import type { Match, ScoredEntry } from '../results.js'
import { bestDistance } from './bestDistance.js'
import { pushHeap, replaceHeapRoot } from './heap.js'
import type { RawPreparedScore, StoredItem } from './types.js'

function worse<TItem, TKey>(
  left: ScoredEntry<TItem, TKey>,
  right: ScoredEntry<TItem, TKey>,
): boolean {
  return (
    left.score > right.score || (left.score === right.score && left.order > right.order)
  )
}

function result<TItem, TKey>(
  entries: ScoredEntry<TItem, TKey>[],
): readonly Match<TItem, TKey>[] {
  entries.sort((a, b) => a.score - b.score || a.order - b.order)
  return entries.map(({ item, key, score }) => ({ item, key, score }))
}

export function topDistance<TItem, TKey>(
  items: readonly StoredItem<TItem, TKey>[],
  score: RawPreparedScore,
  threshold: number | null,
  limit: number | null,
  optimal: number | null,
): readonly Match<TItem, TKey>[] {
  if (limit === 0) return []
  if (limit === 1) {
    const found = bestDistance(items, score, threshold, optimal)
    return found === undefined ? [] : [found]
  }

  // `order` is the source position, which for stored items is the loop index:
  // nothing is skipped before scoring, so a separate counter tracked `index`
  // exactly.
  if (limit !== null) {
    const heap: ScoredEntry<TItem, TKey>[] = []
    let cutoff = threshold
    for (let index = 0; index < items.length; index++) {
      const entry = items[index]
      const value = score(entry.prepared, cutoff)
      if (threshold !== null && value > threshold) continue
      if (heap.length < limit) {
        const candidate = { item: entry.item, key: entry.key, score: value, order: index }
        pushHeap(heap, candidate, worse)
        if (heap.length === limit) {
          cutoff = heap[0].score
          if (optimal !== null && cutoff === optimal) break
        }
        continue
      }
      // A tie loses on order, and every later candidate has a later order, so
      // the numeric test alone decides admission — no comparator call.
      if (value >= heap[0].score) continue
      const candidate = { item: entry.item, key: entry.key, score: value, order: index }
      replaceHeapRoot(heap, candidate, worse)
      cutoff = heap[0].score
      if (optimal !== null && cutoff === optimal) break
    }
    return result(heap)
  }

  const results: ScoredEntry<TItem, TKey>[] = []
  for (let index = 0; index < items.length; index++) {
    const entry = items[index]
    const value = score(entry.prepared, threshold)
    if (threshold === null || value <= threshold) {
      results.push({ item: entry.item, key: entry.key, score: value, order: index })
    }
  }
  return result(results)
}
