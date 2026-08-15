export type { MaybeSequence, Sequence } from '../../types.js'

export interface ScorerOptions {
  readonly scoreCutoff?: number | undefined
  readonly scoreHint?: number | undefined
}
