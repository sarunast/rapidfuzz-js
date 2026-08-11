export type { MaybeSequence, Sequence } from '../../core/types.js'

/** Internal options understood by the algorithm implementations. */
export interface ScorerOptions {
  readonly scoreCutoff?: number | undefined
  readonly scoreHint?: number | undefined
}
