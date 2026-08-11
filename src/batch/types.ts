import type { Scorer } from '../core/scorer.js'
import type { Direction, Normalizer } from '../core/types.js'
import type { ScoreArrayKind } from './scoreArray.js'

export interface BatchOptions<D extends Direction, K extends ScoreArrayKind = 'f64'> {
  readonly scorer: Scorer<D>
  readonly into?: K | undefined
  readonly normalize?: Normalizer | undefined
}
