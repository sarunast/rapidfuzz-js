export interface Match<T, K> {
  readonly item: T
  readonly key: K
  readonly score: number
}

export interface ScoredEntry<T, K> extends Match<T, K> {
  readonly order: number
}
