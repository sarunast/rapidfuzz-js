import type { DriverMatch, RawPreparedScore, StoredItem } from './types.js'

export function bestDistance<T, K>(
  items: readonly StoredItem<T, K>[],
  score: RawPreparedScore,
  threshold: number | null,
  optimal: number | null,
): DriverMatch<T, K> | undefined {
  // The winner is carried as the stored entry it already is, so a run of
  // improvements allocates one result rather than one per improvement.
  //
  // What makes the first candidate a winner is `best`, not the sentinel: a
  // custom metric may return the sentinel as a real score, and a bare
  // `value < bestScore` would then select nothing. The sentinel is only here
  // to mirror `bestSimilarity`.
  let best: StoredItem<T, K> | undefined
  let bestScore = Number.POSITIVE_INFINITY
  let cutoff = threshold
  for (let index = 0; index < items.length; index++) {
    const entry = items[index]
    const value = score(entry.prepared, cutoff)
    if (threshold !== null && value > threshold) continue
    if (best === undefined || value < bestScore) {
      best = entry
      bestScore = value
      cutoff = value
      if (optimal !== null && value === optimal) break
    }
  }
  return best === undefined
    ? undefined
    : { item: best.item, key: best.key, score: bestScore }
}
