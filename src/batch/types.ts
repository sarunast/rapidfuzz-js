import type { Scorer } from '../core/scorer.js'
import type { Direction, Normalizer } from '../core/types.js'
import type { ScoreArrayKind } from './scoreArray.js'

export interface BatchOptions<D extends Direction, K extends ScoreArrayKind = 'f64'> {
  readonly scorer: Scorer<D>

  /** Element type the scores are stored as. Defaults to `'f64'`. */
  readonly into?: K | undefined

  /** Applied to every query and every choice before scoring. */
  readonly normalize?: Normalizer | undefined

  /**
   * The score a pair has to reach, on the scorer's own scale — `0..100` for a
   * fuzz scorer, `0..1` for a normalized one, a count for a raw edit distance.
   * A pair that misses it is stored as the far end of the scorer's bounds:
   * the worst similarity, or the worst distance.
   *
   * Read before {@link scoreMultiplier}, so the threshold means the same thing
   * whatever the multiplier is.
   */
  readonly threshold?: number | undefined

  /**
   * Multiplies every score after thresholding and before storage. Defaults to
   * `1`. May be negative — `-1` turns a similarity into a rank key that sorts
   * ascending — and an integral `into` rounds the product half away from zero.
   */
  readonly scoreMultiplier?: number | undefined
}
