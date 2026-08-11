import type { Match, ScoredEntry } from '../results.js'
import type { RawPreparedScore, StoredItem } from './types.js'

export function topDistance<T, K>(
  items: readonly StoredItem<T, K>[],
  score: RawPreparedScore,
  threshold: number | null,
  limit: number | null,
): readonly Match<T, K>[] {
  const results: ScoredEntry<T, K>[] = []
  let order = 0
  for (const entry of items) {
    const value = score(entry.prepared, threshold)
    if (threshold === null || value <= threshold) {
      results.push({ item: entry.item, key: entry.key, score: value, order })
    }
    order++
  }
  results.sort((a, b) => a.score - b.score || a.order - b.order)
  const selected = limit === null ? results : results.slice(0, limit)
  return selected.map(({ item, key, score: value }) => ({ item, key, score: value }))
}
