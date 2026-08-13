/** A choice id and what it scored — what a driver hands back to its caller. */
export interface ScoredId {
  readonly id: number
  readonly score: number
}

export interface RawPreparedScore {
  /** Returns the actual score when it qualifies; pruning may return only a miss. */
  (choice: unknown, threshold: number | null): number
}
