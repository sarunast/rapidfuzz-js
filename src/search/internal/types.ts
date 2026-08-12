export interface StoredItem<TItem, TKey> {
  readonly item: TItem
  readonly key: TKey
  readonly prepared: unknown
}

export type DriverMatch<TItem, TKey> = Match<TItem, TKey>

export interface RawPreparedScore {
  /** Returns the actual score when it qualifies; pruning may return only a miss. */
  (choice: unknown, threshold: number | null): number
}
import type { Match } from '../results.js'
