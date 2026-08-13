/** A choice id and what it scored — what a driver hands back to its caller. */
export interface ScoredId {
  readonly id: number
  readonly score: number
}
