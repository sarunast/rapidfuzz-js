export interface StoredItem<T, K> {
  readonly item: T
  readonly key: K
  readonly prepared: unknown
}

export type DriverMatch<T, K> = Match<T, K>

export interface RawPreparedScore {
  /** Returns the actual score when it qualifies; pruning may return only a miss. */
  (choice: unknown, threshold: number | null): number
}
import type { Match } from '../results.js'
