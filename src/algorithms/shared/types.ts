import type { Sequence } from '../../core/types.js'

export type { MaybeSequence, Sequence } from '../../core/types.js'

export type Processor = (sequence: Sequence) => Sequence

/** Internal options understood by the algorithm implementations. */
export interface ScorerOptions {
  readonly processor?: Processor | undefined
  readonly scoreCutoff?: number | undefined
  readonly scoreHint?: number | undefined
}

export interface EditopsOptions {
  readonly processor?: Processor | undefined
}
