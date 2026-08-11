import type { Sequence } from '../../core/types.js'

export interface StoredItem<T, K> {
  readonly item: T
  readonly key: K
  readonly sequence: Sequence
  readonly prepared: unknown
}

export interface DriverMatch<T, K> {
  readonly item: T
  readonly key: K
  readonly score: number
}

export interface RawPreparedScore {
  (choice: unknown, threshold: number | null): number
}
