import type { DriverMatch, RawPreparedScore, StoredItem } from './types.js'

export function bestDistance<T, K>(
  items: readonly StoredItem<T, K>[],
  score: RawPreparedScore,
  threshold: number | null,
  optimal: number | null,
): DriverMatch<T, K> | undefined {
  let found: DriverMatch<T, K> | undefined
  let cutoff = threshold
  for (const entry of items) {
    const value = score(entry.prepared, cutoff)
    if (threshold !== null && value > threshold) continue
    if (found === undefined || value < found.score) {
      found = { item: entry.item, key: entry.key, score: value }
      cutoff = value
      if (optimal !== null && value === optimal) break
    }
  }
  return found
}
